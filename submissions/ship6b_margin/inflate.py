"""Ship 6b test-time refinement inflate (CE→margin swap over Ship 6).

Fork of Ship 6 (ship6_ttrefine). Identical archive + deployment scaffold.
Only change: seg refinement loss is boundary-gated logit-margin hinge instead
of global cross-entropy. See experiments/hr_v23_lane3_prep/flywheel_node_ce_vs_margin_empirical.md —
CE on Ship 5's saturated seg distribution regresses argmax-mismatch (91% worse),
while margin on a dilated class-boundary band flips 92% of init mismatches correct.

Training + accept gate both use margin_hinge_per_pair(seg_hat, argmax, band).
W_SEG=10 (raw-logit scale, not CE log-prob scale — 10× lower coefficient).
Accept gate surrogate: W_SEG*margin + sqrt(10*pose_mse), same shape as Ship 6.
"""
import io
import os
import sys
import tempfile
import time
import zlib
from pathlib import Path

import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments" / "seg_ctw"))

from pose_laplace_codec import decode_pose_laplace  # noqa: E402
from seg_sparse_m11_codec import decode_seg_split_m11  # noqa: E402
from flat_fp4_codec import FLAT_MAGIC, decode_flat  # noqa: E402

import frame_utils  # noqa: E402
import modules as _modules  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from frame_utils import camera_size  # noqa: E402


# -----------------------------
# Grad-preserving rgb_to_yuv6 monkey-patch (PoseNet autograd path)
# -----------------------------
def _rgb_to_yuv6_grad(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Grad-preserving rgb_to_yuv6: byte-identical to upstream on valid inputs."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_grad
_modules.rgb_to_yuv6 = _rgb_to_yuv6_grad


def ste_uint8(x: torch.Tensor) -> torch.Tensor:
    """Straight-through uint8 quant: forward rounds+clamps, backward identity."""
    clamped = x.clamp(0.0, 255.0)
    return clamped.detach().round() - clamped.detach() + clamped


# -----------------------------
# Ship 5 FP4 dequantization (vendored verbatim)
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles: torch.Tensor, scales: torch.Tensor, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()
        nibbles = nibbles.view(-1, block_size)
        signs = (nibbles >> 3).to(torch.int64)
        mag_idx = (nibbles & 0x7).to(torch.int64)
        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = levels[mag_idx]
        q = torch.where(signs.bool(), -q, q)
        dq = q * scales[:, None].to(torch.float32)
        return dq.view(-1)[:flat_n].reshape(orig_shape)


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    flat = packed.reshape(-1)
    hi = (flat >> 4) & 0x0F
    lo = flat & 0x0F
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


def get_decoded_state_dict(payload_data, device: torch.device):
    data = torch.load(io.BytesIO(payload_data), map_location=device)
    state_dict = {}
    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            padded_count = rec["packed_weight"].numel() * 2
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), padded_count)
            w = FP4Codebook.dequantize_from_nibbles(
                nibbles, rec["scales_fp16"].to(device), rec["weight_shape"]
            )
        else:
            w = rec["weight_fp16"].to(device).float()
        state_dict[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for name, tensor in data["dense_fp16"].items():
        state_dict[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)
    return state_dict


# -----------------------------
# Ship 5 architecture (inference-only, vendored verbatim)
# -----------------------------
class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)


class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)


class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)

    def forward(self, x):
        return self.pw(self.dw(x))


class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))


class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond_emb):
        residual = x
        x = self.norm2(self.conv2(self.conv1(x)))
        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta
        return self.act(residual + x)


def make_coord_grid(batch, height, width, device, dtype):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=40, c2=44, depth_mult=4):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )
        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2, coords):
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f


class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch, hidden=36, depth_mult=4):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat):
        x = self.block1(feat)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0


class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=32, hidden=36, depth_mult=4):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat, cond_emb):
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0


class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(
            num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head = FrameHead(
            in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)
        self.frame2_head = Frame2StaticHead(
            in_ch=56, hidden=52, depth_mult=depth_mult)

    def forward(self, mask2, pose6):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)
        shared_feat = self.shared_trunk(mask2, coords)
        pred_frame2 = self.frame2_head(shared_feat)
        cond_emb = self.pose_mlp(pose6)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)
        return pred_frame1, pred_frame2


def load_ctw_mask(path: Path) -> torch.Tensor:
    """Decode zlib-wrapped sparse-M11 CTW blob → (n_pairs, H, W) uint8 tensor."""
    raw = zlib.decompress(path.read_bytes())
    with tempfile.NamedTemporaryFile(suffix=".ctw.raw", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        seg = decode_seg_split_m11(tmp_path)
    finally:
        tmp_path.unlink()
    if seg.dtype != np.uint8:
        raise TypeError(f"CTW decode yielded {seg.dtype}, expected uint8")
    if seg.ndim != 3 or seg.shape[1:] != (384, 512):
        raise ValueError(f"CTW decode yielded shape {seg.shape}, expected (N, 384, 512)")
    if int(seg.max()) > 4:
        raise ValueError(f"CTW decode class max {seg.max()} > 4 — index corruption")
    return torch.from_numpy(seg).contiguous()


# -----------------------------
# Test-time refinement (Stage 3 core)
# -----------------------------
MODEL_H, MODEL_W = 384, 512
POSE_DIM = 6
N_CLASSES = 5

REFINE_BATCH = 4
REFINE_STEPS = 50
REFINE_STEPS_FALLBACK = 35
WATCHDOG_ELAPSED_S = 22 * 60
REFINE_LR = 0.05
W_SEG = 10.0  # margin raw-logit scale; 10× less than CE-equivalent 100.0
W_POSE = 1000.0
ACCEPT_GATE_EVERY = 10
BOUNDARY_DILATE_PX = 2


def _dilate_bool(mask: torch.Tensor, radius: int) -> torch.Tensor:
    m = mask.float().unsqueeze(0).unsqueeze(0)
    k = 2 * radius + 1
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=radius).squeeze(0).squeeze(0).bool()


def _class_boundary_mask(argmax: torch.Tensor, radius: int) -> torch.Tensor:
    h_edge = argmax[:, 1:] != argmax[:, :-1]
    v_edge = argmax[1:, :] != argmax[:-1, :]
    raw = torch.zeros_like(argmax, dtype=torch.bool)
    raw[:, :-1] |= h_edge
    raw[:, 1:] |= h_edge
    raw[:-1, :] |= v_edge
    raw[1:, :] |= v_edge
    return _dilate_bool(raw, radius)


def margin_hinge_per_pair(logits: torch.Tensor, target: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    B = logits.shape[0]
    lf = logits.float()
    gt = lf.gather(1, target.unsqueeze(1)).squeeze(1)
    masked = lf.scatter(1, target.unsqueeze(1), torch.finfo(lf.dtype).min)
    max_other = masked.max(dim=1).values
    violation = F.relu(max_other - gt)
    out = torch.zeros(B, device=logits.device, dtype=torch.float32)
    for i in range(B):
        sel = violation[i][band[i]]
        out[i] = sel.mean() if sel.numel() > 0 else torch.tensor(0.0, device=logits.device)
    return out


def _refine_chunk(
    net: nn.Module,
    f0_ship: torch.Tensor,
    f1_init: torch.Tensor,
    seg_target_argmax: torch.Tensor,
    pose_target: torch.Tensor,
    n_steps: int,
    use_amp: bool = True,
    loss_log: list | None = None,
) -> torch.Tensor:
    """Per-chunk Adam on f1_param against decoded-archive teacher targets.

    Returns per-pair best uint8 frames (B, H_cam, W_cam, 3) chosen by lowest
    surrogate W_SEG*margin_hinge + sqrt(10*pose_mse) observed at gate steps.
    """
    B = f0_ship.shape[0]
    device_type = f0_ship.device.type
    amp_enabled = use_amp and device_type == "cuda"
    f1_param = f1_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([f1_param], lr=REFINE_LR)
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)

    band = torch.stack(
        [_class_boundary_mask(seg_target_argmax[i], BOUNDARY_DILATE_PX) for i in range(B)],
        dim=0,
    )

    best_score = torch.full((B,), float("inf"), device=f0_ship.device)
    best_f1 = ste_uint8(f1_param).detach().clone()

    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        f1_scored = ste_uint8(f1_param)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
            hat_pair = torch.stack([f0_ship, f1_scored], dim=1)
            pose_hat, seg_hat = net(hat_pair)
            pose_diff_sq = (pose_hat["pose"][..., :POSE_DIM].float() - pose_target) ** 2
            pose_mse_per = pose_diff_sq.reshape(B, -1).mean(dim=1)
            seg_margin_per = margin_hinge_per_pair(seg_hat, seg_target_argmax, band)
            loss = (W_SEG * seg_margin_per + W_POSE * pose_mse_per).sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if loss_log is not None:
            loss_log.append(float(loss.detach()))

        if step % ACCEPT_GATE_EVERY == 0 or step == n_steps:
            with torch.no_grad():
                f1_scored_post = ste_uint8(f1_param)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                    pose_hat_p, seg_hat_p = net(torch.stack([f0_ship, f1_scored_post], dim=1))
                    pose_mse_p = ((pose_hat_p["pose"][..., :POSE_DIM].float() - pose_target) ** 2).reshape(B, -1).mean(dim=1)
                    seg_margin_p = margin_hinge_per_pair(seg_hat_p, seg_target_argmax, band)
                surrogate = W_SEG * seg_margin_p + torch.sqrt(torch.clamp(10.0 * pose_mse_p, min=0.0))
                nan_mask = torch.isnan(surrogate)
                surrogate = torch.where(nan_mask, torch.full_like(surrogate, float("inf")), surrogate)
                improved = surrogate < best_score
                best_score = torch.where(improved, surrogate, best_score)
                if improved.any():
                    idx = improved.nonzero(as_tuple=True)[0]
                    best_f1[idx] = f1_scored_post.detach()[idx]

    return best_f1.detach().round().clamp(0, 255).to(torch.uint8)


# -----------------------------
# Main: 3-arg inflate (data_dir, output_dir, file_list)
# -----------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    t0 = time.time()

    files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

    # Load Ship 5 generator weights (FP4 flat or legacy).
    generator = JointFrameGenerator().to(device)
    model_blob = (data_dir / "w").read_bytes()
    weights_data = brotli.decompress(model_blob)
    if weights_data[:4] == FLAT_MAGIC:
        state_dict = {k: v.to(device) for k, v in decode_flat(model_blob).items()}
    else:
        state_dict = get_decoded_state_dict(weights_data, device)
    generator.load_state_dict(state_dict, strict=True)
    generator.train(False)
    for p in generator.parameters():
        p.requires_grad_(False)

    mask_frames_all = load_ctw_mask(data_dir / "m").to(device)
    pose_vecs, _ = decode_pose_laplace(data_dir / "p")
    pose_frames_all = torch.from_numpy(pose_vecs).float().to(device)

    # DistortionNet for teacher loss (SegNet + PoseNet).
    net = DistortionNet().to(device)
    net.train(False)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"models ready at t={time.time() - t0:.1f}s", flush=True)

    out_h, out_w = camera_size[1], camera_size[0]  # (W, H) → (H, W)
    pairs_per_file = 600
    cursor = 0

    for file_name in files:
        base_name = os.path.splitext(file_name)[0]
        raw_out_path = out_dir / f"{base_name}.raw"

        file_masks = mask_frames_all[cursor:cursor + pairs_per_file]
        file_poses = pose_frames_all[cursor:cursor + pairs_per_file]
        cursor += pairs_per_file

        with open(raw_out_path, "wb") as f_out:
            for i in range(0, file_masks.shape[0], REFINE_BATCH):
                in_mask2 = file_masks[i:i + REFINE_BATCH].long()
                in_pose6 = file_poses[i:i + REFINE_BATCH].float()
                B = in_mask2.shape[0]

                elapsed = time.time() - t0
                n_steps = REFINE_STEPS if elapsed < WATCHDOG_ELAPSED_S else REFINE_STEPS_FALLBACK

                # Ship 5 decode: fake1 (pose-conditioned) → f0, fake2 (static) → f1.
                with torch.no_grad():
                    fake1, fake2 = generator(in_mask2, in_pose6)
                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    # (B, 3, H, W) → (B, H, W, 3) for DistortionNet input layout.
                    f0_ship = fake1_up.permute(0, 2, 3, 1).contiguous()
                    f1_init = fake2_up.permute(0, 2, 3, 1).contiguous()

                seg_target = in_mask2  # (B, 384, 512) long, SegNet output resolution.
                pose_target = in_pose6  # (B, 6) raw pose.

                best_f1 = _refine_chunk(
                    net, f0_ship, f1_init,
                    seg_target_argmax=seg_target,
                    pose_target=pose_target,
                    n_steps=n_steps,
                    use_amp=(device.type == "cuda"),
                )

                f0_uint8 = f0_ship.detach().round().clamp(0, 255).to(torch.uint8)
                for b in range(B):
                    f_out.write(f0_uint8[b].contiguous().cpu().numpy().tobytes())
                    f_out.write(best_f1[b].contiguous().cpu().numpy().tobytes())

                if ((i + REFINE_BATCH) % 40 == 0) or (i + B >= file_masks.shape[0]):
                    el = time.time() - t0
                    print(f"  {file_name} {i + B}/{file_masks.shape[0]} ({el:.1f}s, n_steps={n_steps})", flush=True)

    total = time.time() - t0
    print(f"done: {cursor} pairs in {total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
