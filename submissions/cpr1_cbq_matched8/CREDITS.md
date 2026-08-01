# Lineage, attribution, and originality boundary

This submission is a **derived, incremental payload optimization**. It does
not claim a new video-compression architecture.

## Direct code and artifact ancestor

The direct ancestor is Fesal Fayed's
[`semantic-pose-HPAC_CPR1` challenge PR #130][pr130] and its
[`comma-ai-semantic-pose-hpac-cpr1` reproducibility repository][cpr1-repro].
The inflater modules in this directory are copied from that MIT-licensed
repository; its license is retained as `LICENSE`.

Inherited unchanged from CPR1:

- the semantic-master / pose-slave frame decomposition;
- the semantic token renderer and its quantized weights;
- the integer-lattice HPAC entropy decoder and token stream;
- the standalone neutral-gray, 12-dimensional learned pose carrier;
- per-pair signed int12 pose coefficients;
- compact carrier coding, canonical Huffman/Rice streams, and payload framing;
- camera-resolution bilinear/bicubic rendering;
- the deployed CUDA inflater;
- exact per-row coefficient rescue as an optimization technique.

The archive retains CPR1's semantic and HPAC state. The changed data are the
quantized carrier basis and per-pair coefficient rows.

## Earlier architectural ancestor

CPR1 explicitly continues jas0xf's
[`jas0xf_adversarial_neural_representation` PR #86][pr86]. PR #86 already
combined:

- semantic class tokens rendered into RGB;
- asymmetric master/slave reconstruction roles;
- patch-group HPAC entropy modeling;
- Type-A and Type-B masked convolutions;
- frame/FiLM conditioning;
- previous-frame token context;
- arithmetic coding.

None of those mechanisms are claimed here.

## Research and challenge prior art

The HPAC architecture derives from Li, Bai, Wang, Zhao, Jiang, and Liu,
“Rethinking Autoregressive Models for Lossless Image Compression via
Hierarchical Parallelism and Progressive Adaptation,” arXiv:2511.10991
([paper][hpac-paper]).

Low-rank spatial pose actuation and per-frame coefficient/basis reconstruction
have earlier challenge precedent in EthanYangTW's qpose submissions
[`#67`][pr67] and [`#79`][pr79]. This submission does not claim to invent
low-rank pose actuation, basis/coefficients, or pose-conditioned generation.

## What this experiment adds

The narrow contribution is **Compensability-aware Basis Quantization (CBQ)**:

1. measure a shared basis atom's bit value after the already-transmitted
   per-pair coefficient channel is allowed to compensate;
2. use PoseNet-Jacobian active-set steps as a screening oracle;
3. coarsen carrier basis atoms 2, 5, and 9 from five-bit to four-bit signed
   support;
4. accept changes against exact forward PoseNet evaluation and actual packaged
   bytes;
5. apply inherited exact coefficient rescue to selected difficult rows.

This changes offline encoder-side rate allocation, not the deployed inference
architecture. The final archive is 190,212 bytes, 840 bytes smaller than the
191,052-byte CPR1 artifact.

## Measured attribution limit

The control for this release is effort-matched: it keeps the unchanged CPR1
basis and receives the identical eight-pass, full-600-row coefficient-search
budget. On the exact-batch local audit:

| Endpoint | Pose | Archive bytes | Pass-8 accepts |
|---|---:|---:|---:|
| unchanged CPR1 basis | `0.0000080833270` | `191,040` | 205 |
| CBQ atoms 2/5/9 | `0.0000078771227` | `190,212` | 140 |

At matched effort the CBQ basis change accounts for only **2.55%** lower Pose
distortion. The defensible contribution of CBQ is therefore the **828-byte**
rate saving, which follows directly from four-bit atom support and is
independent of any coefficient search. Almost all of the Pose improvement over
public CPR1 comes from the inherited exact coefficient search, not from the new
basis.

Neither branch had converged — both still accepted changes on pass 8 — so this
is a matched-budget comparison, not a convergence claim.

Consequently:

- additional coefficient search is the dominant source of improvement;
- exact coefficient search is inherited and is not claimed as new;
- CBQ's own contribution is the rate saving, and its Pose contribution is
  small at matched effort;
- no claim of worldwide novelty, architectural invention, or a generally
  useful learned video codec is made.

Measured on one Tesla T4 in one session with the unmodified official
evaluator, one copy of the decoder, and the archive as the only variable:

| Archive | Pose | Seg | Bytes | Score |
|---|---:|---:|---:|---:|
| public CPR1 (calibration) | `0.00002331` | `0.00029660` | `191,052` | `0.172141` |
| superseded CBQ candidate | `0.00001220` | `0.00029660` | `190,216` | `0.167362` |
| this release | `0.00000896` | `0.00029660` | `190,212` | `0.165780` |

The calibration row reproduces the maintainer-triggered workflow's published
result for that archive to all eight printed decimals.

“New” here means only that compensability-aware shared-basis bit allocation
was not identified in the audited public challenge lineage. This is not a
patent search or legal priority opinion.

[pr130]: https://github.com/commaai/comma_video_compression_challenge/pull/130
[cpr1-repro]: https://github.com/fesalfayed/comma-ai-semantic-pose-hpac-cpr1
[pr86]: https://github.com/commaai/comma_video_compression_challenge/pull/86
[pr67]: https://github.com/commaai/comma_video_compression_challenge/pull/67
[pr79]: https://github.com/commaai/comma_video_compression_challenge/pull/79
[hpac-paper]: https://arxiv.org/abs/2511.10991
