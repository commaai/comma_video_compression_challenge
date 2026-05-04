"""Stage 3: v3.32 smooth disagreement — switch Softplus → sigmoid(-margin/tau).

Source: Stage 2 output (e2ev331_ep8650).
Loss: smooth_disagreement_seg (sigmoid bell-curve, peaks at margin=0).
Optimizer: AdamW with **fresh cosine schedule** at peak_lr=1e-4 (NOT continuing
prior schedule).

1500 epochs. Encoded for reproducibility — not re-run.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import smooth_disagreement_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 1500) -> StageConfig:
    return StageConfig(
        name="stage3_v332_smooth",
        seg_loss_fn=lambda logits, targets: smooth_disagreement_seg_loss(
            logits, targets, tau=0.3),
        epochs=epochs,
        eval_every=25,
        batch_size=8,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-4,           # ← fresh cosine peak, lower than v3.28
        latent_lr_mult=10.0,
        grad_clip=1.0,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,           # QAT joins in Stage 4
        resume_from=resume_from,
        output_dir=output_dir,
    )
