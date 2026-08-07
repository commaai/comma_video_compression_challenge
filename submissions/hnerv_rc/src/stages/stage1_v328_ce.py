"""Stage 1: v3.28 CE phase — bulk calibration from random init with Cross-Entropy seg loss.

Source: random init (the only stage with init_latents_random=True).
Output canonical: ckpt at ep3000 of v3.28 (full v3.28 ran 10K ep, but Stage 2 resumes
from ep3000 because that's where the loss switches from CE to Softplus).

Loss: F.cross_entropy(seg_logits, hard_targets) + sqrt(10·MSE) pose.
Optimizer: AdamW only, peak_lr=1e-3, latent_lr=1e-2 (10×). 20-ep linear warmup, then
cosine to 5e-6.

3000 epochs. **Encoded for reproducibility — not re-run for this submission**;
we resume from canonical Stage 4 output (`e2ev332_ep10200`).
"""
from pathlib import Path

from .common import StageConfig, train_stage
from losses import ce_seg_loss


def make_config(output_dir: Path, epochs: int = 3000) -> StageConfig:
    return StageConfig(
        name="stage1_v328_ce",
        seg_loss_fn=lambda logits, targets: ce_seg_loss(logits, targets),
        epochs=epochs,
        eval_every=25,
        batch_size=8,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,           # v3.28 peak_lr
        latent_lr_mult=10.0,
        grad_clip=1.0,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,          # no C1a yet
        cat_sigma=0.2,           # unused (lambda=0)
        use_qat=False,           # no QAT yet
        resume_from=None,
        init_latents_random=True,
        output_dir=output_dir,
    )
