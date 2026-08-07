"""Stage 4: v3.32 + QAT — same loss as Stage 3, INT8 fake-quant joins in forward.

Source: Stage 3 output (e2ev332_ep10150).
Loss: smooth_disagreement_seg (unchanged from Stage 3).
Optimizer: AdamW continuing fresh cosine from Stage 3.

500 epochs. Output canonical: `e2ev332_ep10650` (saved as
`e2ev332_d28_c36_e10650_bs8_ep10200`). This is the input for Stage 5.

Encoded for reproducibility — not re-run.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import smooth_disagreement_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 500) -> StageConfig:
    return StageConfig(
        name="stage4_v332_qat",
        seg_loss_fn=lambda logits, targets: smooth_disagreement_seg_loss(
            logits, targets, tau=0.3),
        epochs=epochs,
        eval_every=25,
        batch_size=8,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-4,           # continues Stage 3
        latent_lr_mult=10.0,
        grad_clip=1.0,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=True,            # ← QAT joins
        resume_from=resume_from,
        output_dir=output_dir,
    )
