# semantic_joint_ctxmix

Reproduction package for the 180,002-byte challenge-video archive that scored
0.14797617125559104 on `[contest-CUDA]` Tesla T4, n600 at evaluated commit
`1c9fbbf58716eb0f26bcdf2a91e3c89d0e4efdde` (recomputed from the evaluator's
printed components; the report displays 0.15 at two decimals).
This is a competitive submission on its claimed contest-CUDA axis: it is below
the public PR #135 result of 0.162 on that same axis.

## Credits and lineage

The learned vehicle descends from public contest work, and I do not claim it:

- **PR #130** (Fesal Fayed, @fesalfayed) — the semantic-token renderer family: a dense
  per-pair semantic token grid decoded by a small learned renderer, integer HPAC
  arithmetic coding of the token stream, and the pose-carrier format.
- **PR #135** (Shreyan Mohanty, @codexblack) — the CPR1-polished form of that vehicle
  (0.162 on contest-CUDA), whose receiver structure this submission's runtime carries
  forward.
- **PR #133** (@JasonMo123) — a refinement of PR #130's vehicle that PR #135
  carries forward.

My contribution is the decision and lossless-representation layer on top:

1. **Joint edit admission** — per-pair segmentation-token edits admitted only when the
   measured joint delta (seg + pose + rate together) improves the score; 455 of 573
   proposed edits admitted.
2. **Pose-carrier re-solve** — damped Gauss–Newton re-fit of the existing stored pose
   coefficients against PoseNet on the edited rendered frames.
3. **Five lossless coder/container stages** — integer log-odds context mixing, group-
   and tile-conditioned token re-encodes, a joint re-encode collecting banked wins.
   Decoded output is byte-identical across all five.

## Reproduce

- **Inflate:** `./inflate.sh archive_dir output_dir file_list` — requires CUDA;
  the prior contest-tested target was `linux-nvidia-t4`. The prior contest-CPU
  attempt timed out at the 1,800-second inflation limit.
- **Compress:** use CPython 3.13.12 with NumPy 1.26.4, Torch 2.12.1, and Brotli
  1.2.0, then run
  `PYTHONDONTWRITEBYTECODE=1 python compress.py --base-archive /path/to/base_archive.zip --store /durable/ssd/path --resume`.
  The encoder performs no network requests, verifies the complete source-tree
  manifest, retains every payload and checkpoint outside this public tree, and
  refuses unless the rebuilt archive matches the pinned SHA-256 exactly
  (180,002 bytes). The required base archive is pinned the same way: the script
  refuses unless `base_archive.zip` matches SHA-256
  `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`
  (180,456 bytes). The base is not included in this tree; ask and I will
  provide it. One run takes about an hour of CPU; individual stages also
  self-check with in-memory determinism repeats. Pass `--repeats 2` to add a
  full second rebuild as an end-to-end determinism demonstration.
- **Local process, stated plainly:** the stages that decided the archive's content —
  training, the segmentation edit solve, and the pose re-solve — did not run on CUDA.
  They ran on a local Apple-silicon machine, using MLX/Metal ports of the contest's
  frozen scoring networks, with every accepted change verified against the standard
  CPU PyTorch scorers. Re-running those stages takes days of compute and would not
  reproduce the exact bytes on different hardware, so `compress.py` replays their
  recorded lossless output instead. The code, configs, and logs live in the research
  repository linked below.
- **TODO (planned after this submission):** a `--device` flag (cuda / mps / cpu) so
  the solve stages can run on CUDA as well as on the Apple-silicon stack; a
  full-pipeline mode that starts from the raw video by running the credited
  PR #130/#135 training scripts and then these solve and packaging stages
  (equivalent quality, not identical bytes — neural training does not reproduce
  bit-for-bit across different GPUs); and per-video auto-configuration so the
  pipeline can be pointed at any clip, not just this one.

## Verify

- Archive SHA-256: `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`,
  180,002 bytes.
- Native/Python receiver identity: 600/600 pairs, 3,662,409,600 raw output bytes,
  0 differing.
- Research-receipt repository: <https://github.com/adpena/comma-lab> (evaluated
  commit `1c9fbbf58716eb0f26bcdf2a91e3c89d0e4efdde`, verified publicly
  accessible 2026-09-03).

The score above belongs to the exact archive bytes. On 2026-09-03 this tree's
public CUDA entrypoint was rerun end-to-end on a 1:1 Tesla T4 against those
bytes and reproduced the row exactly: PoseNet 0.00000637, SegNet 0.00020139,
180,002 bytes, recomputed score 0.14797617125559104 (532.3 s inflation +
40.6 s evaluation).

`[contest-CPU]` is RECORD-WITH-REASON: no exact AFR1 CPU score was run, and no
older CPU score is inherited. The prior contest-CPU attempt timed out at the
1,800-second inflation limit, so this package makes no CPU score claim.

Corrections welcome — if any attribution is incomplete, tell me and I will fix it.
