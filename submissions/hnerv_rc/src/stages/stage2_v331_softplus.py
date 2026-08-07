"""Stage 2: v3.31 Softplus L1 phase — switch CE → tau-Softplus seg loss.

Source: Stage 1 output (e2ev328_ep3000).
Loss: tau-Softplus seg (tau=0.3) + sqrt(10·MSE) pose.
Optimizer: AdamW (continues from Stage 1 — peak_lr=1e-3 cosine, but resumed
mid-schedule from ep3000 of 10K total). Effectively the LR is mid-cosine
when Stage 2 starts.

5650 epochs. Encoded for reproducibility — not re-run.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import tau_softplus_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 5650) -> StageConfig:
    return StageConfig(
        name="stage2_v331_softplus",
        seg_loss_fn=lambda logits, targets: tau_softplus_seg_loss(
            logits, targets, tau=0.3),
        epochs=epochs,
        eval_every=25,
        batch_size=8,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,           # peak (cosine schedule "continues" v3.28's)
        latent_lr_mult=10.0,
        grad_clip=1.0,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        resume_from=resume_from,
        output_dir=output_dir,
    )
