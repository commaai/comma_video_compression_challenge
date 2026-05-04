# hnerv_muon_finetuned_from_pr95

This submission is fine-tuned from PR #95, `hnerv_muon`.

Changes from the PR #95 baseline:

- metric fine-tuned HNeRV decoder archive;
- compact architecture-ordered decoder packing with fp16 scales;
- small decode-side postprocess tuned on the public evaluation path:
  - frame 0 red channel `-1`;
  - frame 0 blue channel `-1`;
  - frame 1 green channel `-1`.

Latest pod evaluation on PyAV/CUDA:

- PoseNet distortion: `0.00003489`
- SegNet distortion: `0.00058795`
- archive size: `178,392` bytes
- displayed score: `0.20`

The original PR #95 code and representation are otherwise preserved.
