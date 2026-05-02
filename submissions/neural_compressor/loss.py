"""
Differentiable approximation of the comma.ai compression challenge score.

  Score = 100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)

This module is wired up to the ACTUAL comma.ai networks defined in
`modules.py` of the challenge repo:

  - PoseNet: takes (B, T, C, H, W) float, returns dict with 'pose' shape (B, 12).
             Distortion = MSE on first 6 dims of 'pose'.
  - SegNet : takes (B, T, C, H, W) float, uses ONLY x[:, -1, ...] internally,
             returns (B, 5, 384, 512) logits.
             Distortion = mean(argmax disagreement) — we replace with a soft
             expected-disagreement that is differentiable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CommaScoreLoss(nn.Module):
    """
    Args:
        segnet, posenet:  pre-trained, frozen comma.ai networks (eval mode)
        lam_rate       :  scalar weight on the rate term during training.
                          The official score uses 25 * rate (where rate is
                          archive_bytes / original_bytes). Because our train
                          rate is bits-per-pixel from the entropy model — not
                          the archive ratio — we expose lam as a hyperparameter
                          and you sweep it (Lagrangian).
        use_official_weights:
                          If True, use the exact 100 / 25 / sqrt(10*) recipe.
                          If False, use lam_rate * R + segnet + posenet
                          (more stable for early training).
    """

    def __init__(
        self,
        segnet: nn.Module,
        posenet: nn.Module,
        lam_rate: float = 0.01,
        use_official_weights: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.segnet = segnet.eval()
        self.posenet = posenet.eval()
        for p in self.segnet.parameters():
            p.requires_grad_(False)
        for p in self.posenet.parameters():
            p.requires_grad_(False)

        self.lam_rate = lam_rate
        self.use_official_weights = use_official_weights
        self.eps = eps

    # ---------- distortion terms ----------

    def segnet_soft_disagreement(self, x_orig: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        x_orig, x_recon: (B, 2, 3, H, W) float in [0, 255]
        Returns scalar in [0, 1].

        SegNet preprocess uses x[:, -1, ...] internally — only the last frame.
        We replicate that to keep the gradient path correct.
        """
        with torch.no_grad():
            seg_in_orig = self.segnet.preprocess_input(x_orig)         # (B, 3, 384, 512)
            logits_orig = self.segnet(seg_in_orig)                      # (B, 5, h, w)
            P_orig = F.softmax(logits_orig, dim=1)

        seg_in_recon = self.segnet.preprocess_input(x_recon)
        logits_recon = self.segnet(seg_in_recon)
        P_recon = F.softmax(logits_recon, dim=1)

        agreement = (P_orig * P_recon).sum(dim=1)        # (B, h, w)
        return (1.0 - agreement).mean()

    def posenet_mse(self, x_orig: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        x_orig, x_recon: (B, 2, 3, H, W) float in [0, 255]

        We mirror the official `posenet.compute_distortion` which uses only
        the first half of the 12-dim pose head: out['pose'][..., :6].
        """
        with torch.no_grad():
            pose_in_orig = self.posenet.preprocess_input(x_orig)        # (B, 12, h, w)
            out_orig = self.posenet(pose_in_orig)                       # dict
            target = out_orig["pose"][..., :6]                          # (B, 6)

        pose_in_recon = self.posenet.preprocess_input(x_recon)
        out_recon = self.posenet(pose_in_recon)
        pred = out_recon["pose"][..., :6]                               # (B, 6)

        return F.mse_loss(pred, target)

    # ---------- rate ----------

    @staticmethod
    def rate_bpp(y_likelihoods: torch.Tensor, z_likelihoods: torch.Tensor, num_pixels: int) -> torch.Tensor:
        """
        Bits per pixel from compressai entropy modules.
        likelihoods are p(ỹ | ẑ) and p(ẑ), both in (0,1].
        bits = -log2(p), summed over all elements.
        """
        bits_y = (-torch.log2(y_likelihoods.clamp(min=1e-12))).sum()
        bits_z = (-torch.log2(z_likelihoods.clamp(min=1e-12))).sum()
        return (bits_y + bits_z) / num_pixels

    # ---------- forward ----------

    def forward(
        self,
        x_orig: torch.Tensor,         # (B, 2, 3, H, W) float in [0, 255]
        x_recon: torch.Tensor,        # (B, 2, 3, H, W) float in [0, 255]
        y_likelihoods: torch.Tensor,
        z_likelihoods: torch.Tensor,
        num_pixels: int,
    ):
        seg_loss = self.segnet_soft_disagreement(x_orig, x_recon)
        pose_loss = self.posenet_mse(x_orig, x_recon)
        bpp = self.rate_bpp(y_likelihoods, z_likelihoods, num_pixels)

        if self.use_official_weights:
            # Direct surrogate of the leaderboard score.
            # NOTE: bpp is bits/pixel; the leaderboard rate is archive/original ratio.
            # bpp / 24 ≈ rate-vs-raw. Real archive rate ≈ much smaller because the
            # 'original' MKV is already H.265 compressed. Treat this as a proxy.
            rate_term = bpp / 24.0
            total = (
                100.0 * seg_loss
                + 25.0 * rate_term
                + torch.sqrt(10.0 * pose_loss + self.eps)
            )
        else:
            # Stable training form. Sweep lam_rate.
            total = seg_loss + pose_loss + self.lam_rate * bpp

        return {
            "loss": total,
            "seg": seg_loss.detach(),
            "pose": pose_loss.detach(),
            "bpp": bpp.detach(),
        }