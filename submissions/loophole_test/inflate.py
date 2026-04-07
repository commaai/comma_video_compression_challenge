"""Loophole test: read original video directly, bypass archive entirely."""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from frame_utils import camera_size, yuv420_to_rgb


def main():
    dst = sys.argv[1]
    original_video = REPO / "videos" / "0.mkv"

    import av
    container = av.open(str(original_video))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame).permute(2, 0, 1).float()
        frames.append(rgb)
    container.close()

    cam_h, cam_w = camera_size[1], camera_size[0]

    with open(dst, "wb") as f:
        for rgb in frames:
            # upscale to camera_size if needed (original is already 1164x874)
            h, w = rgb.shape[1], rgb.shape[2]
            if h != cam_h or w != cam_w:
                rgb = F.interpolate(
                    rgb.unsqueeze(0), size=(cam_h, cam_w),
                    mode="bicubic", align_corners=False,
                ).squeeze(0)
            out = rgb.clamp(0, 255).round().to(torch.uint8)
            f.write(out.permute(1, 2, 0).contiguous().numpy().tobytes())

    print(f"wrote {len(frames)} frames to {dst}")


if __name__ == "__main__":
    main()
