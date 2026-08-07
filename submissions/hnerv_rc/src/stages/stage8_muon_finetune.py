"""Stage 8: muon_finetune — switches optimizer to Muon (hidden convs) + AdamW (rest).

Source: Stage 7 output (canonical: exp4_sigma01_ep975, 0.2042).
Output canonical: `muon_ep250` at score 0.2009.

Same loss as Stage 7 (L7 Softplus + C1a@σ=0.1,λ=0.02 + QAT). Optimizer
changes: Muon on hidden Conv2d weights (11 tensors, ~177K params),
AdamW on stem Linear, RGB heads, biases, latents (~52K decoder + 16K latent).

LR drops dramatically: ADAMW_LR = 1e-5 (was 3e-5), MUON_LR = 2e-4 (new).

Researcher #24 tweak applied: Muon weight_decay = 5e-4 (not in canonical).
Theoretical justification: Chen-Li-Liu arXiv:2506.15054 — Muon's spectral-norm
KKT mechanism requires WD to be active.

Default canonical: 3000 epochs. Our extension: 5000 epochs.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import l7_softplus_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 5000,
                muon_weight_decay: float = 5e-4) -> StageConfig:
    return StageConfig(
        name="stage8_muon_finetune",
        seg_loss_fn=lambda logits, targets: l7_softplus_seg_loss(
            logits, targets, tau=0.3, l7_threshold=1.0, l7_mult=4.0),
        epochs=epochs,
        eval_every=25,
        batch_size=8,
        ema_decay=0.999,
        use_muon=True,                 # ← optimizer switch
        adamw_lr=1e-5,                 # canonical
        muon_lr=2e-4,                  # canonical
        muon_weight_decay=muon_weight_decay,  # researcher #24 idea 1
        latent_lr_mult=10.0,
        grad_clip=1.0,
        grad_clip_muon=1.0,            # canonical kept (researcher #24 idea 3 was SKIPPED)
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.02,
        cat_sigma=0.1,
        use_qat=True,
        resume_from=resume_from,
        output_dir=output_dir,
    )
