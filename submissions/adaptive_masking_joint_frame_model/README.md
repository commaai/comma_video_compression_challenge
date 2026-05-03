This builds on the Quantizr/qpose/r55-style approach of storing evaluator-relevant latent signals rather than a conventional compressed video.

The main difference is that this stores the SegNet semantic mask stream directly as a 5-class tensor compressed with an adaptive range coder, instead of storing the mask as an AV1/Brotli video. A compact QZS3 neural renderer then uses the decoded masks, pose data, and small correction streams to regenerate RGB frames at inflate time.

## Result

Verified local CUDA result:

```text
archive.zip: 215,960 bytes
PoseNet:     0.00058400
SegNet:      0.00060989
score:       0.2812077922144353
````

Score breakdown:

```text
SegNet term:  ~0.060989
PoseNet term: ~0.076420
Rate term:    ~0.143799
```

## Archive layout

`archive.zip` contains one stored ZIP member named `p`.

| Component                         |   Bytes | Purpose                                            |
| --------------------------------- | ------: | -------------------------------------------------- |
| Adaptive range-coded SegNet masks | 159,011 | Stores the 600-frame, 5-class semantic mask tensor |
| Compact QZS3 renderer payload     |  55,725 | Stores quantized neural-renderer weights           |
| Pose stream                       |     899 | Preserves short-horizon dynamics for PoseNet       |
| Router actions                    |     225 | Applies small per-pair color/action corrections    |

Total ZIP size: `215,960` bytes.

## Key files

| File                   | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `inflate.sh`           | entrypoint                                                               |
| `inflate.py`           | Main inflater, payload parser, renderer reconstruction, and frame writer |
| `range_mask_codec.cpp` | Adaptive range decoder for the semantic mask tensor                      |
| `archive.zip`          | Counted compressed payload                                               |


## Key changes

Most related neural-renderer submissions store a compressed mask video and then render frames from that video plus pose data. This instead stores the semantic masks as class IDs and compresses them with a custom adaptive range codec.

Main differences:

* Stores exact SegNet class IDs rather than video-coded masks
* Uses spatial and temporal context in `range_mask_codec.cpp`
* Packs all reconstruction data into one ZIP member
* Reconstructs a compact QZS3 model payload at inflate time
* Uses tiny pose and router side channels for motion and correction

Based on 
* `quantizr` #55
* `fp4_mask_gen` #62
* `unified_brotli` #64
