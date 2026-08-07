"""Stage 7: suite_9 exp4_sigma01 — sharpens C1a sigma 0.2 → 0.1.

Source: Stage 6 output (canonical: lambda_0.02_ep475).
Output canonical: `exp4_sigma01_ep975` at score 0.2042.

Same loss as Stage 6 except C1a sigma changes from 0.2 to 0.1, making
the soft histogram entropy more peaked around integer grid points.

Optimizer: AdamW only. LR = 3e-5 cosine.

Default canonical: 2000 epochs. Our extension: 3000 epochs.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import l7_softplus_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 3000) -> StageConfig:
    return StageConfig(
        name="stage7_sigma_sweep",
        seg_loss_fn=lambda logits, targets: l7_softplus_seg_loss(
            logits, targets, tau=0.3, l7_threshold=1.0, l7_mult=4.0),
        epochs=epochs,
        eval_every=25,
        batch_size=8,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=3e-5,
        latent_lr_mult=10.0,
        grad_clip=1.0,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.02,
        cat_sigma=0.1,      # ← Stage 7's change vs Stage 6 (was 0.2)
        use_qat=True,
        resume_from=resume_from,
        output_dir=output_dir,
    )
