"""
Scale-hyperprior video compressor (Ballé 2018, adapted for driving frames).

Architecture
------------
  Main encoder g_a:    x  ->  y    (4 strided conv layers, /16 spatial, M channels)
  Hyperprior g_a':     y  ->  z    (3 strided conv layers, /4 spatial, N_hyp channels)
  Hyperprior g_s':     ẑ  ->  σ̂   (3 transposed conv layers, predicts scale of y)
  Main decoder g_s:    ŷ  ->  x̂   (4 transposed conv layers)

Quantization
------------
  Train:  ỹ = y + U(-0.5, 0.5),  ẑ = z + U(-0.5, 0.5)   (differentiable proxy)
  Eval :  ŷ = round(y),           ẑ = round(z)

Entropy
-------
  z is encoded under a learned factorized prior  -> EntropyBottleneck
  y is encoded under a Gaussian conditional with σ predicted from ẑ -> GaussianConditional

Both modules are from the `compressai` library and provide both a differentiable rate
estimate (bits) AND real arithmetic coding (compress/decompress to bytes).

Note on size
------------
The trained weights of THIS network must fit inside archive.zip and count toward
the rate. With N=64, M=128 the model is ~1.5M params (~6 MB fp32, ~3 MB fp16,
~1.5 MB int8). Tune N, M down if you need a smaller model.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install compressai
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN  # generalized divisive normalization


def _conv(in_ch, out_ch, k=5, s=2):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2)


def _deconv(in_ch, out_ch, k=5, s=2):
    return nn.ConvTranspose2d(
        in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2, output_padding=s - 1
    )


class MainEncoder(nn.Module):
    """RGB image (B,3,H,W) -> latent y (B,M,H/16,W/16)."""

    def __init__(self, N=64, M=128):
        super().__init__()
        self.net = nn.Sequential(
            _conv(3, N), GDN(N),
            _conv(N, N), GDN(N),
            _conv(N, N), GDN(N),
            _conv(N, M),
        )

    def forward(self, x):
        return self.net(x)


class MainDecoder(nn.Module):
    """Latent ŷ (B,M,H/16,W/16) -> RGB image (B,3,H,W) in [0,1]."""

    def __init__(self, N=64, M=128):
        super().__init__()
        self.net = nn.Sequential(
            _deconv(M, N), GDN(N, inverse=True),
            _deconv(N, N), GDN(N, inverse=True),
            _deconv(N, N), GDN(N, inverse=True),
            _deconv(N, 3),
        )

    def forward(self, y_hat):
        # decoder outputs unconstrained values; clamp at use-site for losses
        return self.net(y_hat)


class HyperEncoder(nn.Module):
    """y (B,M,H/16,W/16) -> z (B,N_hyp,H/64,W/64)."""

    def __init__(self, M=128, N_hyp=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(M, N_hyp, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            _conv(N_hyp, N_hyp), nn.ReLU(inplace=True),
            _conv(N_hyp, N_hyp),
        )

    def forward(self, y):
        return self.net(torch.abs(y))  # abs so prior models scale, not sign


class HyperDecoder(nn.Module):
    """ẑ (B,N_hyp,H/64,W/64) -> σ for y (B,M,H/16,W/16)."""

    def __init__(self, M=128, N_hyp=64):
        super().__init__()
        self.net = nn.Sequential(
            _deconv(N_hyp, N_hyp), nn.ReLU(inplace=True),
            _deconv(N_hyp, N_hyp), nn.ReLU(inplace=True),
            nn.Conv2d(N_hyp, M, 3, stride=1, padding=1),
        )

    def forward(self, z_hat):
        # softplus to enforce σ > 0
        return F.softplus(self.net(z_hat)) + 1e-6


class ScaleHyperpriorCompressor(nn.Module):
    """
    Full compressor. forward() is for training (returns recon + rate estimate).
    compress() / decompress() use real arithmetic coding for inference.

    The model takes RGB frames in [0, 255] uint8 range. It internally divides by 255
    (so reconstructions are roughly in [0,1]) and the loss can rescale to 0-255 to
    feed into the comma SegNet/PoseNet which expect 0-255 inputs.
    """

    def __init__(self, N=64, M=128, N_hyp=64):
        super().__init__()
        self.N, self.M, self.N_hyp = N, M, N_hyp

        self.g_a = MainEncoder(N=N, M=M)
        self.g_s = MainDecoder(N=N, M=M)
        self.h_a = HyperEncoder(M=M, N_hyp=N_hyp)
        self.h_s = HyperDecoder(M=M, N_hyp=N_hyp)

        # Entropy models from compressai
        self.entropy_bottleneck = EntropyBottleneck(N_hyp)  # for z
        self.gaussian_conditional = GaussianConditional(None)  # σ supplied at runtime

    # ---------- helpers ----------

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, mult: int = 64) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Pad H, W up to multiples of `mult`. Returns (padded_x, pad_tuple)."""
        _, _, h, w = x.shape
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        # F.pad order: (left, right, top, bottom)
        pad = (0, pad_w, 0, pad_h)
        return F.pad(x, pad, mode="replicate"), pad

    @staticmethod
    def _unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
        l, r, t, b = pad
        if r == 0 and b == 0:
            return x
        return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]

    # ---------- training forward ----------

    def forward(self, x_uint8: torch.Tensor):
        """
        x_uint8: (B, 3, H, W) uint8 or float in [0, 255]
        returns dict with keys: x_hat (in 0..255 range), likelihoods_y, likelihoods_z, num_pixels
        """
        x = x_uint8.float() / 255.0
        x, pad = self._pad_to_multiple(x, mult=64)

        y = self.g_a(x)
        z = self.h_a(y)

        # z: factorized prior with built-in noise/quantization
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # y: gaussian with σ from hyperdecoder
        sigma = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma)

        x_hat = torch.sigmoid(self.g_s(y_hat))  # back to [0,1]
        x_hat = self._unpad(x_hat, pad)
        x_hat = x_hat * 255.0  # back to 0-255 for feeding into SegNet/PoseNet

        B, _, H, W = x_uint8.shape
        return {
            "x_hat": x_hat,           # (B, 3, H, W) in [0, 255]
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": z_likelihoods,
            "num_pixels": B * H * W,
        }

    # ---------- inference: real bitstream ----------

    def update(self, force: bool = True) -> bool:
        """
        Build CDF tables for arithmetic coding. Call AFTER training, BEFORE compress/decompress.
        """
        updated = self.entropy_bottleneck.update(force=force)
        # gaussian_conditional needs scale_table; use compressai's default geometric range
        from compressai.entropy_models import GaussianConditional as _GC
        scale_table = torch.exp(torch.linspace(torch.log(torch.tensor(0.11)), torch.log(torch.tensor(256.0)), 64))
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    @torch.no_grad()
    def compress(self, x_uint8: torch.Tensor):
        """
        Real entropy coding. Returns:
            {"strings": [y_strings, z_strings], "shape": z.shape[-2:], "pad": pad, "orig_hw": (H,W)}
        """
        x = x_uint8.float() / 255.0
        x, pad = self._pad_to_multiple(x, mult=64)
        H, W = x_uint8.shape[-2:]

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        sigma = self.h_s(z_hat)

        indexes = self.gaussian_conditional.build_indexes(sigma)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {
            "strings": [y_strings, z_strings],
            "shape": tuple(z.size()[-2:]),
            "pad": pad,
            "orig_hw": (H, W),
        }

    @torch.no_grad()
    def decompress(self, strings, shape, pad, orig_hw):
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        sigma = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(sigma)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)
        x_hat = torch.sigmoid(self.g_s(y_hat))
        x_hat = self._unpad(x_hat, pad)
        # crop to exact original size in case of slight mismatch
        H, W = orig_hw
        x_hat = x_hat[..., :H, :W]
        return (x_hat * 255.0).round().clamp(0, 255).to(torch.uint8)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # smoke test
    m = ScaleHyperpriorCompressor(N=64, M=128, N_hyp=64)
    print(f"params: {num_params(m):,}  (~{num_params(m)*4/1e6:.1f} MB fp32, "
          f"~{num_params(m)*2/1e6:.1f} MB fp16)")
    x = torch.randint(0, 256, (1, 3, 874, 1164), dtype=torch.uint8)
    out = m(x)
    print(f"x_hat: {out['x_hat'].shape}  range=[{out['x_hat'].min():.1f}, {out['x_hat'].max():.1f}]")
    print(f"y_lik: {out['y_likelihoods'].shape}  z_lik: {out['z_likelihoods'].shape}")