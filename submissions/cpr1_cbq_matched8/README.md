# CPR1-CBQ matched-8 submission

This submission inherits Fesal Fayed's public CPR1 semantic-pose codec, which
continues jas0xf's earlier challenge work, and keeps its inflater and payload
topology.

Our delta is compensability-based quantization (CBQ): basis atoms 2, 5, and 9
use 4-bit quantization instead of 5-bit quantization, followed by exact
PoseNet-guided coordinate recoding of the existing int12 coefficient stream.
The selected artifact includes eight full passes over all 600 public pairs.
No additional side stream or runtime model was added.

Charged archive:

- size: `190,212` bytes
- SHA-256:
  `051baf408f57fae3b343d6ee218ab963d070b3935ceb0b2f412c93a53cf3fab0`

See `CREDITS.md` and `LICENSE` for inheritance and licensing details.

## Public-test result

From the unmodified official evaluator on Tesla T4 at challenge commit
`5387a097398ec6581c7e4e428231e1821fc62670`, in the pinned `cu128` environment
(Python 3.11.15, torch 2.9.0+cu128): Pose `0.00000896`, Seg `0.00029660`,
190,212 bytes, which places the score in `[0.16577695, 0.16578323]`.

In the same session, on the same GPU and with the same copy of the decoder, the
unmodified public CPR1 archive reproduced the maintainer-triggered workflow's
published Pose `0.00002331` and Seg `0.00029660` for that archive to all eight
printed decimals, calibrating the rig against the official runner. CPR1 scores
`0.172141` there; this submission scores `0.165780`.

## Attribution boundary

The effort-matched control keeps the unchanged CPR1 basis and receives the
identical eight-pass search budget. At that endpoint the CBQ basis change
accounts for only 2.55% lower Pose distortion, so CBQ's defensible contribution
is the 828-byte rate saving; almost all of the Pose improvement over public
CPR1 comes from the inherited exact coefficient search. See `CREDITS.md`.

This is verification evidence, not a guarantee of leaderboard position. The
maintainer-triggered workflow remains authoritative.
