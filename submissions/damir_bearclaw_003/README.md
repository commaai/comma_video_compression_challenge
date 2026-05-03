# damir_bearclaw_003

This submission is a single-stream representation-oriented experiment.

Instead of preserving the whole frame uniformly, it:

- keeps only the middle 50% of the frame as the stored signal
- applies boundary-preserving smoothing to that middle band
- reconstructs synthetic top and bottom filler bands during inflate

The motivation came from direct inspection of the fixed scoring models.

SegNet appears to derive most of its useful semantic structure from the central driving corridor, while the upper and lower filler regions are heavily position-biased:

![SegNet class map](images/segnet_classes_frame_0171_mask.png)

PoseNet error analysis suggested strong sensitivity to the central road corridor and near-field anchor regions, but increasingly diminishing practical value from chasing ever smaller scalar deviations:

![PoseNet occlusion example](images/pose0_occlusion_rank_04_pair_0252.png)

The broader point is not that this submission beats the best previous score. It does not. The point is that once proxy errors are within a practically acceptable band, additional optimization pressure should shift toward rate rather than continuing to reward uniform proxy fidelity.

This submission is therefore best read as a probe of that limit:

- preserve the task-relevant middle-region structure
- accept small proxy-error increases once they are practically tolerable
- then push hard on stored size
