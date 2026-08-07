"""Stage 6: lambda_sweep λ=0.02 branch — tightens C1a regularization.

Source: Stage 5 output (canonical: c1a_l7_ep2075).
Output canonical: `lambda_0.02_ep475` at score 0.2054.

Same loss as Stage 5 (L7 Softplus + C1a + QAT) but with C1a lambda 0.01 → 0.02.
Sigma stays at 0.2.

Optimizer: AdamW only. LR = 3e-5 cosine.

Default canonical: 1000 epochs. Our extension: 2000 epochs.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import l7_softplus_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 2000) -> StageConfig:
    return StageConfig(
        name="stage6_lambda_sweep",
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
        cat_lambda=0.02,    # ← Stage 6's change vs Stage 5
        cat_sigma=0.2,
        use_qat=True,
        resume_from=resume_from,
        output_dir=output_dir,
    )
