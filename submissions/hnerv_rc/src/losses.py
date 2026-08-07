"""All loss functions across the 8 training stages, plus QAT helpers + EMA helpers.

Loss progression:
  Stage 1 (CE):                ce_seg
  Stage 2 (Softplus):          tau_softplus_seg(tau=0.3)
  Stage 3 (Smooth):            smooth_disagreement_seg(tau=0.3)
  Stage 4 (+QAT):              smooth_disagreement_seg + apply_qat() in forward
  Stage 5 (+L7+C1a):           l7_softplus_seg + cat_entropy_v2(sigma=0.2) + QAT
  Stage 6 (lambda=0.02):       same as 5, lambda=0.02
  Stage 7 (sigma=0.1):         same as 6, sigma=0.1
  Stage 8 (Muon):              same as 7, Muon optimizer

Pose loss is sqrt(10·MSE), constant across all stages. EMA decay 0.999.
Aggregation: loss = 100*seg + 1*pose + cat_lambda*c1a_entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Segmentation losses (one per stage family)
# ============================================================================

def ce_seg_loss(seg_logits, targets_hard):
    """Stage 1 CE seg loss."""
    return F.cross_entropy(seg_logits, targets_hard)


def tau_softplus_seg_loss(seg_logits, targets_hard, tau=0.3):
    """Stage 2 tau-Softplus surrogate. Smooth in margin space."""
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    return (tau * F.softplus(-margin / tau)).mean()


def smooth_disagreement_seg_loss(seg_logits, targets_hard, tau=0.3):
    """Stage 3+4 smooth disagreement loss (sigmoid-on-negative-margin).
    Bell-curve gradient peaks at margin=0 — pushes boundary pixels across the line."""
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    return torch.sigmoid(-margin / tau).mean()


def l7_softplus_seg_loss(seg_logits, targets_hard,
                         tau=0.3, l7_threshold=1.0, l7_mult=4.0):
    """Stage 5+ L7-weighted Softplus: pixels where margin<threshold get a
    (1 + l7_mult) boost (renormalized to mean 1). Concentrates gradient on
    hard-to-classify pixels."""
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    per_pixel = tau * F.softplus(-margin / tau)
    with torch.no_grad():
        weights = 1.0 + l7_mult * (margin < l7_threshold).float()
        weights = weights / weights.mean()
    return (per_pixel * weights).mean()


# ============================================================================
# Pose loss (constant across all stages)
# ============================================================================

def pose_loss(pose_pred, pose_target):
    """sqrt(10·MSE) — concave-in-MSE, emphasizes small errors more than plain MSE."""
    mse = F.mse_loss(pose_pred, pose_target)
    return torch.sqrt(10.0 * mse + 1e-12)


# ============================================================================
# C1a entropy regularizer (Stage 5+)
# ============================================================================

def cat_entropy_v2(decoder, sigma=0.2, sample_size=2000, device=None):
    """Size-weighted soft histogram entropy.
    For each Conv2d/Linear weight tensor:
      - quantize to {-127, ..., 127} via Gaussian soft-assignment with bandwidth sigma
      - compute categorical entropy
      - weight by tensor size (numel)
    Returns: weighted mean entropy in bits/weight, averaged across all weight tensors.

    Pushing this down (small sigma + big lambda) sharpens the post-INT8
    distribution at integer grid points.
    """
    if device is None:
        device = next(decoder.parameters()).device
    bins = torch.arange(-127, 128, device=device, dtype=torch.float32)
    total_numel = 0
    weighted_entropy = torch.zeros((), device=device)
    for name, mod in decoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, 'weight'):
            w = mod.weight
            numel = w.numel()
            ma = w.abs().max().detach()
            if ma.item() < 1e-12:
                continue
            wn = (w / (ma / 127.0)).flatten()
            if wn.numel() > sample_size:
                idx = torch.randperm(wn.numel(), device=wn.device)[:sample_size]
                wn = wn[idx]
            sa = torch.exp(-0.5 * ((wn.unsqueeze(1) - bins.unsqueeze(0)) / sigma).pow(2))
            sa = sa / (sa.sum(dim=1, keepdim=True) + 1e-12)
            bp = sa.mean(dim=0)
            bp = bp / (bp.sum() + 1e-12)
            entropy = -(bp * torch.log2(bp + 1e-12)).sum()
            weighted_entropy = weighted_entropy + numel * entropy
            total_numel += numel
    return weighted_entropy / max(total_numel, 1)


# ============================================================================
# QAT (INT8 fake-quant with straight-through estimator)
# ============================================================================

def fake_quantize(tensor, n_levels=127):
    """Per-tensor symmetric INT8 fake-quant. STE: forward rounds, backward passes through."""
    ma = tensor.abs().max()
    scale = ma / n_levels if ma > 0 else 1.0
    q = (tensor / scale).round().clamp(-n_levels, n_levels)
    return (q * scale - tensor).detach() + tensor


def apply_qat(decoder):
    """Replace Conv2d/Linear weights with fake-quantized versions IN PLACE.
    Returns dict of originals to restore after the forward pass.

    Pattern:
        originals = apply_qat(decoder)
        decoded = decoder(...)
        restore_qat(decoder, originals)
    """
    originals = {}
    for name, mod in decoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, 'weight'):
            originals[name] = mod.weight.data.clone()
            mod.weight.data = fake_quantize(mod.weight.data)
    return originals


def restore_qat(decoder, originals):
    for name, mod in decoder.named_modules():
        if name in originals:
            mod.weight.data = originals[name]


# ============================================================================
# EMA helpers
# ============================================================================

def ema_update(ema_decoder, decoder, ema_latents, latents, decay=0.999):
    """In-place EMA update."""
    with torch.no_grad():
        for ep, pv in zip(ema_decoder.parameters(), decoder.parameters()):
            ep.data.mul_(decay).add_(pv.data, alpha=1 - decay)
        if ema_latents is not None and latents is not None:
            ema_latents.mul_(decay).add_(latents.data, alpha=1 - decay)
