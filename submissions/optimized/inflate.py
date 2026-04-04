"""Inflate compressed video back to raw frames at original resolution."""
import logging
import sys

import av
import torch
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def inflate(src: str, dst: str) -> None:
    target_w, target_h = camera_size

    container = av.open(src)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # Decode using the same YUV→RGB conversion as the ground truth evaluation pipeline
    # (BT.601 limited range + bilinear chroma upsampling), NOT libswscale's to_ndarray()
    frames = []
    for frame in container.decode(video=0):
        rgb = yuv420_to_rgb(frame)  # returns torch uint8 (H, W, 3)
        frames.append(rgb)
    container.close()

    N = len(frames)
    H, W, _ = frames[0].shape
    needs_resize = (H != target_h or W != target_w)
    logger.info("decoded %d frames from %s (%dx%d)", N, src, W, H)

    # Write to file in chunks to limit memory usage
    CHUNK = 64
    with open(dst, "wb") as f:
        for i in range(0, N, CHUNK):
            chunk = torch.stack(frames[i : i + CHUNK])  # (C, H, W, 3) uint8
            if needs_resize:
                x = chunk.permute(0, 3, 1, 2).float()
                # Use bilinear to match model's own bilinear downscale — minimizes round-trip error
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
                chunk = x.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1)
            f.write(chunk.contiguous().numpy().tobytes())

    if needs_resize:
        logger.info("resized %dx%d -> %dx%d", W, H, target_w, target_h)
    logger.info("saved %d frames to %s", N, dst)


if __name__ == "__main__":
    inflate(sys.argv[1], sys.argv[2])
