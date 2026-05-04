# optimization_qpose_josema — Technical Write-up

## Executive summary

This submission is a compact semantic/neural reconstruction entry for the comma video compression challenge.

The final local MPS evaluation result was:

```text
Average PoseNet Distortion: 0.00061985
Average SegNet Distortion: 0.00071020
Submission file size: 277,087 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate: 0.00738003
Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.33The submission is based on a qpose-style reconstruction pipeline with a compact semantic mask stream, a neural generator, pose/latent side information, and localized tile correction actions.

The main engineering lesson was that the semantic mask stream is the most important compression target, but it is also extremely sensitive. Several approaches reduced size or changed the mask representation, but they damaged temporal consistency and produced worse PoseNet distortion. The final version therefore prioritizes stability over aggressive byte reduction.

1. Challenge metric

The challenge metric combines semantic distortion, temporal distortion, and compression rate:

For this submission, the score contribution is:

The important observation is that the metric is not only a rate-distortion metric in the usual RGB sense. It rewards preserving:

semantic content through SegNet;
temporal dynamics through PoseNet;
compressed archive size through the rate term.

The experiments showed that PoseNet is the easiest term to damage. A smaller archive is not useful if the reconstructed sequence loses temporal consistency.

2. Reconstruction pipeline

The final submission uses a semantic/neural reconstruction pipeline rather than a direct RGB video codec.

The pipeline contains four compact components:

Semantic mask stream
Provides high-level scene structure.
Pose / latent side channel
Provides temporal and motion-related conditioning.
Neural frame generator
Reconstructs frames from semantic structure and pose information.
Tile correction actions
Applies small localized corrections in regions that can improve semantic agreement.

The goal is not pixel-perfect reconstruction. The goal is to preserve the information that the evaluation networks care about.

3. Source video diagnostics

The following contact sheet samples representative frames from the source video:

This helped inspect scene variation across the clip and reason about where semantic errors were likely to matter: road boundaries, vehicles, lane-like structures, sky/road transitions, and high-contrast regions.

4. Tile-action optimization

The tile-action component searches for small localized corrections.

Each candidate action is defined by:

frame index;
tile index;
action ID / correction type;
estimated metric gain;
encoded byte cost.

The action search is score-aware. A correction is only useful if the expected metric improvement is larger than the rate penalty from storing it.

This was important because many local actions improve a small region visually or semantically, but are not worth their compressed payload cost.

5. Action subset pruning

After generating candidate actions, I used subset optimization to keep only net-positive actions.

The decision rule was approximately:

keep action if metric improvement > compressed-size penalty

This prevented the archive from growing too much. The final action set is conservative: it makes small corrections without destabilizing the reconstruction.

6. Experiment log

The table below summarizes the main experiments and decisions.

The score comparison shows how the rejected approaches performed when they were evaluated:

PoseNet sensitivity was the main failure mode:

7. Failed direction: simple lossless mask codecs

I tested multiple simple lossless alternatives for the semantic mask stream:

temporal delta coding;
run-length encoding;
changed-pixel streams;
simple contextual arithmetic coding.

These approaches were larger than the original representation.

Observed examples:

Temporal-delta lossless mask codec:
  archive size: 648,559 bytes
  result: rejected

Simple arithmetic mask codec:
  archive size: 305,334 bytes
  result: rejected

This showed that the existing mask representation was already strong. Simple hand-written lossless codecs were not enough to beat it.

A better approach would likely require a learned or highly adaptive entropy model that understands spatial and temporal mask structure.

8. Failed direction: lossy mask re-encoding

I also tested aggressive lossy re-encoding of the semantic mask stream.

This reduced archive size in some cases, but it badly damaged temporal dynamics.

CRF50 mask re-encode:
  archive size: 273,630 bytes
  PoseNet: 0.00529962
  SegNet: 0.00082431
  score: 0.49

CRF60 mask re-encode:
  archive size: 181,142 bytes
  PoseNet: 0.04886454
  SegNet: 0.00201768
  score: 1.02

The CRF60 experiment was especially informative. It substantially reduced the archive size, but PoseNet distortion increased by orders of magnitude. The final score became much worse.

The conclusion was clear:

Reducing bytes is not enough. The mask stream must preserve temporal structure.

9. Failed direction: aggressive keyframe-style mask re-encoding

I also tested a more aggressive keyframe/g=1 style mask re-encode.

Aggressive g=1 mask re-encode:
  archive size: 340,225 bytes
  PoseNet: 0.26924199
  SegNet: 0.00188160
  score: 2.06

This failed on both rate and distortion. It produced a larger archive and severely damaged temporal consistency.

This reinforced that the mask stream is tightly coupled to the neural renderer. It should not be modified independently without checking PoseNet.

10. What worked

The stable strategy was conservative:

preserve the qpose-style neural reconstruction pipeline;
preserve the semantic mask stream;
preserve the pose side channel;
add only small localized tile corrections;
prune actions by net score contribution;
avoid lossy mask changes.

This kept PoseNet stable and avoided catastrophic failures.

11. What did not work

The main failed assumption was:

If the archive becomes smaller, the score should improve.

That assumption was false.

The score depends on compressed size, semantic preservation, and temporal preservation. Reducing size while damaging temporal dynamics gives a worse final score.

In this challenge, byte savings must be earned without changing the temporal behavior that PoseNet measures.

12. Main technical insight

The most promising compression target is the semantic mask stream, but it cannot be compressed naively.

Simple lossless codecs were larger. Simple lossy codecs broke PoseNet. This points toward a more advanced mask representation:

learned entropy coding;
stronger spatial/temporal contexts;
class-transition modeling;
PoseNet-aware optimization;
joint optimization of mask stream and renderer.

The next major score improvement would likely come from a learned or highly adaptive semantic mask compressor, not from generic video-codec parameter tuning.

13. Future work

The highest-value future directions are:

learned entropy coding for semantic masks;
context-adaptive arithmetic coding with richer temporal context;
direct PoseNet-aware mask compression;
action search that evaluates both SegNet and PoseNet effects;
joint training of the renderer and mask representation;
multi-window GPU tile-action search with global subset pruning.

A stronger approach would compress semantic masks using a probability model that understands road scenes, temporal continuity, and class transitions.

14. Final result
Average PoseNet Distortion: 0.00061985
Average SegNet Distortion: 0.00071020
Submission file size: 277,087 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate: 0.00738003
Final score: 0.33

The final submission prioritizes metric stability. It avoids the large PoseNet failures observed in aggressive mask-compression experiments and keeps the reconstruction pipeline reliable.
