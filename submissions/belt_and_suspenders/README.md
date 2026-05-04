# belt_and_suspenders

GPU-evaluated submission. 186 KB archive: HNeRV decoder + 28-d per-frame-pair latents + per-pair single-dim latent sidecar perturbation chosen against DALI cu128 ground truth.

Lineage: HNeRV decoder weights and architecture from PR #95 (hnerv_muon, AaronLeslie138).

Verified score on cu128/DALI 1.52: 0.20946.

GPU-only: inflate.py asserts torch.cuda.is_available() so leaderboard auto-routes to T4 instance.
