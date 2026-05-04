#!/usr/bin/env python
"""HPAC inflate: TokenRendererV62 master + ShrinkSingleNeRV slave + HPAC tokens.

Archive layout (in unzipped data_dir):
  master.pt      : TokenRendererV62 state_dict (FP16, SCN pre-applied)
  slave.pt       : ShrinkSingleNeRV state_dict (LSQ-INT4 pre-applied)
  tokens.bin     : HPAC arithmetic-coded bitstream
  hpac.pt        : HPACMini state_dict (FP16, SCN pre-applied)
  meta.pt        : dict {N, P, delta, ch, slave_channels, slave_d_lat, d_film}

Outputs [N*2, CAMERA_H, CAMERA_W, 3] uint8, interleaved (slave, master) per pair.
"""
from __future__ import annotations
import sys, io, gzip, time
from pathlib import Path
import numpy as np
try:
    import pyppmd
    HAVE_PPMD = True
except ImportError:
    HAVE_PPMD = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import constriction

CAMERA_H, CAMERA_W = 874, 1164
SEGNET_IN_H, SEGNET_IN_W = 384, 512
FEAT_H, FEAT_W = 6, 8
NUM_CLASSES = 5


# ============================================================================
# Master decoder: TokenRendererV62 (SCN pre-applied at build time)
# ============================================================================
class TokenRendererV62(nn.Module):
    def __init__(self, num_pairs=600, num_classes=NUM_CLASSES, d_film=8):
        super().__init__()
        self.num_classes = num_classes
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        self.film_gen = nn.Linear(d_film, 64)
        self.conv1 = nn.Conv2d(num_classes, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.act = nn.GELU()
        # Cross-hardware-deterministic FiLM: (frame_embed @ film_gen.W.T + bias)
        # is computed ONCE on CPU at bake_film_table() time. cuBLAS picks
        # different kernels for tiny (8x64) matmul on Ada vs Turing, which
        # causes master pixels to differ across GPUs. CPU FP32 is bit-portable.
        self.register_buffer("_film_table",
                             torch.zeros(num_pairs, 64), persistent=False)
        self._film_table_baked = False

    def bake_film_table(self):
        """Precompute (frame_embed @ film_gen.W.T + film_gen.b) on CPU FP32.
        Call AFTER load_state_dict. Result is cross-hardware bit-identical."""
        with torch.no_grad():
            emb = self.frame_embed.weight.detach().cpu().float()      # (N, d_film)
            w = self.film_gen.weight.detach().cpu().float()           # (64, d_film)
            b = self.film_gen.bias.detach().cpu().float()             # (64,)
            table = emb @ w.T + b                                      # (N, 64)
            self._film_table.copy_(table.to(self._film_table.device))
        self._film_table_baked = True

    def forward(self, tokens, idx):
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x = self.conv1(x)
        x = self.gn1(x)
        if self._film_table_baked:
            film = self._film_table[idx]
        else:
            emb = self.frame_embed(idx)
            film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)
        x = x * (1.0 + scale.view(-1, 32, 1, 1)) + shift.view(-1, 32, 1, 1)
        x = self.act(x)
        x = self.act(self.gn2(self.conv2(x)))
        x = self.out_conv(x)
        raw = torch.sigmoid(x) * 255.0
        return F.interpolate(raw, size=(CAMERA_H, CAMERA_W),
                             mode="bilinear", align_corners=False)


# ============================================================================
# Slave decoder: ShrinkSingleNeRV (INT4 pre-applied at build time)
# ============================================================================
class _NeRVBlock(nn.Module):
    def __init__(self, c_in, c_out, s=2):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out * s * s, kernel_size=1, bias=True)
        self.ps = nn.PixelShuffle(s)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.ps(self.pw(self.dw(x))))


class ShrinkSingleNeRV(nn.Module):
    def __init__(self, num_pairs=600, d_lat=6, channels=(24, 16, 12, 8, 8, 6, 6)):
        super().__init__()
        assert len(channels) == 7
        self.codes = nn.Embedding(num_pairs, d_lat)
        self.stem = nn.Linear(d_lat, channels[0] * FEAT_H * FEAT_W, bias=True)
        self.stem_act = nn.GELU()
        self.blocks = nn.ModuleList([_NeRVBlock(channels[i], channels[i + 1], s=2) for i in range(6)])
        self.head = nn.Conv2d(channels[-1], 3, kernel_size=1, bias=True)
        self.per_pair_bias = nn.Embedding(num_pairs, 3)
        self.channels = channels

    def forward(self, idx):
        z = self.codes(idx)
        x = self.stem(z).view(-1, self.channels[0], FEAT_H, FEAT_W)
        x = self.stem_act(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x) + self.per_pair_bias(idx).view(-1, 3, 1, 1)
        raw = torch.sigmoid(out) * 255.0
        return F.interpolate(raw, size=(CAMERA_H, CAMERA_W),
                             mode="bilinear", align_corners=False)


# ============================================================================
# HPAC compressor: HPACMini (SCN pre-applied at build time)
# ============================================================================
def _patch_group_mask(k, delta, type_):
    mask = torch.zeros(k, k, dtype=torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + delta * dr
            if type_ == 'A':
                if val < 0:
                    mask[dr_idx, dc_idx] = 1.0
            else:  # 'B'
                if val <= 0:
                    mask[dr_idx, dc_idx] = 1.0
    return mask


class _MaskedConv2dPG(nn.Module):
    """Plain masked conv (no SCN - quantization pre-applied at build time)."""
    def __init__(self, c_in, c_out, k, padding=0, dilation=1, groups=1, type_='B', delta=2, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(c_out, c_in // groups, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        m = _patch_group_mask(k, delta, type_)
        self.register_buffer("mask", m.view(1, 1, k, k), persistent=False)

    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


class _ChannelNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)


class _CausalSPM(nn.Module):
    """Decode-time CausalSPM (no SCN runtime). Mirrors hpac_mini.CausalSPM."""
    def __init__(self, ch, P=32):
        super().__init__()
        self.P = P
        self.norm = _ChannelNorm2d(ch)
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1)
    def forward(self, h_past):
        B, C, H, W = h_past.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x_p = h_past.view(B, C, NRp, P, NCp, P).mean(dim=(3, 5))
        x_p = self.norm(x_p)
        x_p = self.dw(x_p)
        x_p = F.gelu(x_p)
        x_p = self.pw(x_p)
        x_full = x_p.unsqueeze(3).unsqueeze(5).expand(B, C, NRp, P, NCp, P).contiguous()
        return x_full.view(B, C, NRp * P, NCp * P)


class HPACMini(nn.Module):
    def __init__(self, num_pairs=600, num_classes=NUM_CLASSES, P=32, delta=2,
                 d_film=32, ch=64, use_spm=False):
        super().__init__()
        self.num_classes = num_classes
        self.P = P
        self.delta = delta
        self.ch = ch
        self.use_spm = use_spm
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        self.film_gen = nn.Linear(d_film, ch * 2)
        self.conv_a  = _MaskedConv2dPG(num_classes + 2, ch, k=7, padding=3, type_='A', delta=delta)
        self.gn_a    = _ChannelNorm2d(ch)
        self.conv_b1 = _MaskedConv2dPG(ch, ch, k=5, padding=4, dilation=2, groups=ch, type_='B', delta=delta)
        self.gn_b1   = _ChannelNorm2d(ch)
        self.conv_b2 = _MaskedConv2dPG(ch, ch, k=3, padding=4, dilation=4, groups=ch, type_='B', delta=delta)
        self.gn_b2   = _ChannelNorm2d(ch)
        self.conv_past = nn.Conv2d(num_classes, ch, kernel_size=3, padding=1)
        self.spm = _CausalSPM(ch, P=P) if use_spm else None
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1, padding=0)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_P = -1

    def _patch_coord_grid(self, B, device):
        if self._cached_P != self.P or self._coord_cache.numel() == 0:
            P = self.P
            ys = torch.linspace(-1.0, 1.0, P, device=device).view(1, 1, P, 1).expand(1, 1, P, P)
            xs = torch.linspace(-1.0, 1.0, P, device=device).view(1, 1, 1, P).expand(1, 1, P, P)
            grid = torch.cat([ys, xs], dim=1)
            self._coord_cache = grid
            self._cached_P = self.P
        return self._coord_cache.expand(B, -1, -1, -1)

    def _to_patches(self, x):
        B, C, H, W = x.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x = x.view(B, C, NRp, P, NCp, P).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B * NRp * NCp, C, P, P)

    def _from_patches(self, x_p, B, NRp, NCp):
        P = self.P
        C = x_p.shape[1]
        x_p = x_p.view(B, NRp, NCp, C, P, P).permute(0, 3, 1, 4, 2, 5).contiguous()
        return x_p.view(B, C, NRp * P, NCp * P)

    def forward(self, tokens, idx, prev_tokens):
        B, H, W = tokens.shape
        P = self.P
        NRp, NCp = H // P, W // P
        Np = NRp * NCp
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x_p = self._to_patches(x)
        coord_p = self._patch_coord_grid(B * Np, x.device)
        x_in_p = torch.cat([x_p, coord_p], dim=1)
        h_p = self.gn_a(self.conv_a(x_in_p))
        emb = self.frame_embed(idx)
        film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)
        scale_p = scale.view(B, 1, self.ch, 1, 1).expand(B, Np, self.ch, 1, 1).reshape(B * Np, self.ch, 1, 1)
        shift_p = shift.view(B, 1, self.ch, 1, 1).expand(B, Np, self.ch, 1, 1).reshape(B * Np, self.ch, 1, 1)
        h_p = h_p * (1.0 + scale_p) + shift_p
        h_p = F.gelu(h_p)
        x_prev = F.one_hot(prev_tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        h_past_full = self.conv_past(x_prev)
        h_past_p = self._to_patches(h_past_full)
        h_p = h_p + h_past_p
        if self.spm is not None:
            h_p = h_p + self._to_patches(self.spm(h_past_full))
        h_p = F.gelu(self.gn_b1(self.conv_b1(h_p)))
        h_p = F.gelu(self.gn_b2(self.conv_b2(h_p)))
        logits_p = self.head(h_p)
        return self._from_patches(logits_p, B, NRp, NCp)


# ============================================================================
# HPAC token decoder
# ============================================================================
def _reconstruct_hpac_state_dict(packed_sd, device):
    """Reconstruct FP32 state_dict from INT8-packed HPAC state_dict.
    Mirrors `reconstruct_hpac_state_dict` in build_archive_hpac.py."""
    out = {}
    bases = sorted({k[:-len(".weight_q")] for k in packed_sd if k.endswith(".weight_q")})
    for base in bases:
        q = packed_sd[base + ".weight_q"].to(device).float()
        scale = packed_sd[base + ".weight_scale"].to(device).float()
        shape = [1] * q.ndim
        shape[0] = -1
        out[base + ".weight"] = (q * scale.view(*shape)).to(torch.float32)
    skip = set()
    for base in bases:
        skip.add(base + ".weight_q")
        skip.add(base + ".weight_scale")
    for k, v in packed_sd.items():
        if k in skip:
            continue
        out[k] = v.to(device).float() if torch.is_floating_point(v) else v.to(device)
    return out


@torch.no_grad()
def decompress_tokens_hpac(blob: bytes, N: int, H: int, W: int,
                            hpac_pt: Path, P: int, delta: int, ch: int,
                            device: str, use_spm: bool = False,
                            hpac_d_film: int = 32) -> np.ndarray:
    """Decode N frames of [H, W] tokens from HPAC arithmetic-coded bitstream."""
    if str(hpac_pt).endswith(".ppmd"):
        decoded = pyppmd.decompress(Path(hpac_pt).read_bytes(), max_order=4, mem_size=16<<20)
        packed_sd = torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=False)
    elif str(hpac_pt).endswith(".gz"):
        with gzip.open(hpac_pt, "rb") as f:
            packed_sd = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    else:
        packed_sd = torch.load(hpac_pt, map_location="cpu", weights_only=False)
    sd = _reconstruct_hpac_state_dict(packed_sd, device)
    gen = HPACMini(num_pairs=N, num_classes=NUM_CLASSES, P=P, delta=delta, ch=ch,
                   d_film=hpac_d_film, use_spm=use_spm).to(device).eval()
    gen.load_state_dict(sd, strict=False)

    NRp, NCp = H // P, W // P
    rs = torch.arange(P, device=device).view(P, 1).expand(P, P)
    cs = torch.arange(P, device=device).view(1, P).expand(P, P)
    s_grid = (cs + delta * rs)
    n_groups = int((1 + delta) * P - delta)
    # Pre-compute per-group full mask
    group_masks = []
    for s in range(n_groups):
        mp = (s_grid == s)  # [P, P]
        if not mp.any():
            group_masks.append(None)
            continue
        full = mp.unsqueeze(0).unsqueeze(0).expand(NRp, NCp, P, P).permute(0, 2, 1, 3).reshape(NRp * P, NCp * P)
        group_masks.append(full)

    tokens = np.empty((N, H, W), dtype=np.uint8)
    decoded_prev = torch.zeros((1, H, W), dtype=torch.long, device=device)
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(blob, dtype=np.uint32))

    print(f"[hpac-decode] decoding {N} frames (P={P} delta={delta} groups={n_groups})...", flush=True)
    t0 = time.time()
    for f in range(N):
        idx = torch.tensor([f], dtype=torch.long, device=device)
        cur = torch.zeros((1, H, W), dtype=torch.long, device=device)
        for s in range(n_groups):
            mask = group_masks[s]
            if mask is None:
                continue
            logits = gen(cur, idx, decoded_prev)
            probs = F.softmax(logits.float(), dim=1)
            probs_at_s = probs[0][:, mask].permute(1, 0).contiguous()
            n_pos = probs_at_s.shape[0]
            probs_np = probs_at_s.cpu().numpy().astype(np.float64)
            probs_np = np.clip(probs_np, 1e-7, 1.0)
            probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
            decoded = np.empty(n_pos, dtype=np.int64)
            for i in range(n_pos):
                cat = constriction.stream.model.Categorical(probabilities=probs_np[i], perfect=False)
                decoded[i] = decoder.decode(cat)
            cur[0, mask] = torch.from_numpy(decoded).to(device)
        tokens[f] = cur[0].cpu().numpy().astype(np.uint8)
        decoded_prev = cur.clone()
        if (f + 1) % 50 == 0 or f == 0:
            dt = time.time() - t0
            eta = dt * (N - f - 1) / max(f + 1, 1)
            print(f"[hpac-decode]   frame {f+1}/{N}  dt={dt:.0f}s  eta={eta:.0f}s", flush=True)
    print(f"[hpac-decode] done in {time.time()-t0:.0f}s", flush=True)
    return tokens


# ============================================================================
# Main
# ============================================================================
def main():
    if len(sys.argv) != 4:
        print("usage: inflate.py <data_dir> <base> <dst_raw>", file=sys.stderr)
        sys.exit(2)
    data_dir, base, dst_raw = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = torch.load(data_dir / "meta.pt", map_location="cpu", weights_only=False)
    N = meta["N"]
    P = meta.get("P", 32)
    delta = meta.get("delta", 2)
    ch = meta.get("ch", 64)
    use_spm = meta.get("use_spm", False)
    hpac_d_film = meta.get("hpac_d_film", 32)
    slave_channels = tuple(meta["slave_channels"])
    slave_d_lat = meta["slave_d_lat"]
    d_film = meta.get("d_film", 8)

    def _load_pt(path_no_gz: Path):
        gz_p = path_no_gz.with_suffix(path_no_gz.suffix + ".gz")
        if gz_p.exists():
            with gzip.open(gz_p, "rb") as f:
                return torch.load(io.BytesIO(f.read()), map_location=device, weights_only=False)
        return torch.load(path_no_gz, map_location=device, weights_only=False)

    # Master
    master_sd = _load_pt(data_dir / "master.pt")
    master = TokenRendererV62(num_pairs=N, d_film=d_film).to(device).eval()
    master_sd = {k: (v.to(device).float() if torch.is_floating_point(v) else v.to(device))
                 for k, v in master_sd.items()}
    master.load_state_dict(master_sd, strict=False)
    # Bake FiLM table on CPU so master pixels are deterministic across GPUs.
    master.bake_film_table()

    # Slave
    slave_sd = _load_pt(data_dir / "slave.pt")
    slave_sd = {k: (v.to(device).float() if torch.is_floating_point(v) else v.to(device))
                for k, v in slave_sd.items()}
    slave = ShrinkSingleNeRV(num_pairs=N, d_lat=slave_d_lat, channels=slave_channels).to(device).eval()
    slave.load_state_dict(slave_sd, strict=False)

    # Tokens (HPAC)
    blob = (data_dir / "tokens.bin").read_bytes()
    for cand in ("hpac.pt.ppmd", "hpac.pt.gz", "hpac.pt"):
        hpac_path = data_dir / cand
        if hpac_path.exists():
            break
    # Force HPAC decode onto CPU so it matches the CPU encoder bit-exactly.
    # Different GPUs (encoder ran on RTX 4090, eval runs on T4) produce slightly
    # different FP32 softmax → arithmetic decoder asserts. CPU FP32 is portable.
    tokens_np = decompress_tokens_hpac(
        blob, N, SEGNET_IN_H, SEGNET_IN_W,
        hpac_path, P, delta, ch, device, use_spm, hpac_d_film,
    )
    tokens = torch.from_numpy(tokens_np).long().to(device)

    # Render
    out = np.empty((N * 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    chunk = 8 if device == "cuda" else 2
    with torch.inference_mode():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            idx = torch.arange(s, e, device=device)
            tok_chunk = tokens[s:e]
            mst = master(tok_chunk, idx).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            slv = slave(idx).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(e - s):
                out[(s + i) * 2 + 0] = slv[i]
                out[(s + i) * 2 + 1] = mst[i]

    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    dst_raw.write_bytes(out.tobytes(order="C"))
    print(f"Wrote {dst_raw} shape={out.shape} bytes={dst_raw.stat().st_size}")


if __name__ == "__main__":
    main()
