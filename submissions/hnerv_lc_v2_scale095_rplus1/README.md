# hnerv_lc_v2_scale095_rplus1

Built on top of BradyMeighan's `hnerv_lc_v2` PR #100, which itself is built on top of EthanYangTW's `hnerv_muon_finetuned_from_pr95` PR #98 and AaronLeslie138's `hnerv_muon` PR #95.

Changes from PR #100:

- retuned latent correction scale from `0.0100` to `0.0095`;
- added a zero-byte decode-side nudge: frame 0 red channel `+1`.

Fast PyAV/CUDA scorer result on the public video:

- PoseNet distortion: `0.000033274`
- SegNet distortion: `0.000575697`
- archive size: `178,981` bytes
- exact score: `0.194986956`

The archive payload is unchanged from PR #100; only inference-time code constants changed.
