"""Stage 5: c1a_l7_combined — adds L7-weighted Softplus seg + C1a entropy regularizer.

Source: `e2ev332_ep10200` (saved Stage 4 output).
Output canonical: `c1a_l7_ep2075` at score 0.2071.

Loss change: Stage 4's smooth-disagreement seg → L7-weighted Softplus seg
(weights = 1 + 4·𝟙[margin<1]) + C1a entropy regularizer (cat_entropy_v2,
sigma=0.2, lambda=0.01). QAT remains active (inherited from Stage 4).

Optimizer: AdamW only. LR = 3e-5 cosine.

Default canonical: 6000 epochs. Our extension: 9000 epochs.
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import l7_softplus_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 9000) -> StageConfig:
    return StageConfig(
        name="stage5_c1a_l7",
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
        cat_lambda=0.01,    # canonical
        cat_sigma=0.2,      # canonical
        use_qat=True,
        resume_from=resume_from,
        output_dir=output_dir,
    )
