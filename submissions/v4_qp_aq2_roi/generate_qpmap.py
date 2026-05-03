#!/usr/bin/env python
"""Generate SegNet-guided per-block QP offset map for SVT-AV1.

Runs SegNet on the original video to identify class boundaries,
then assigns per-64x64-block QP offsets:
- Sky blocks (uniform class 2): +5 (fewer bits)
- Road boundary blocks (multiple classes with road): -5 (more bits)
- Other: 0 (default)

These mild offsets work WITH aq-mode 2 (SVT-AV1's adaptive quantization),
nudging the encoder's own decisions rather than replacing them.
"""
import sys, argparse, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frame_utils import segnet_model_input_size, yuv420_to_rgb
from modules import SegNet, segnet_sd_path
from safetensors.torch import load_file
import av

seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enc-w", type=int, default=522)
    parser.add_argument("--enc-h", type=int, default=392)
    parser.add_argument("--n-frames", type=int, default=1200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segnet = SegNet().eval().to(device)
    sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(sd)
    del sd

    h_b = (args.enc_h + 63) // 64
    w_b = (args.enc_w + 63) // 64

    # Compute labels
    container = av.open(args.video)
    labels = []
    fidx = 0
    batch = []
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            if fidx % 2 == 1:
                t = yuv420_to_rgb(frame)
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(seg_h, seg_w), mode='bilinear', align_corners=False)
                batch.append(x.squeeze(0))
                if len(batch) == 32:
                    b = torch.stack(batch).to(device)
                    lab = segnet(b).argmax(dim=1).cpu().numpy().astype(np.uint8)
                    labels.append(lab)
                    batch = []
            fidx += 1
        if batch:
            b = torch.stack(batch).to(device)
            lab = segnet(b).argmax(dim=1).cpu().numpy().astype(np.uint8)
            labels.append(lab)
    container.close()
    labels = np.concatenate(labels, axis=0)

    # Generate QP map
    with open(args.output, "w") as f:
        for frame in range(args.n_frames):
            pi = min(frame // 2, labels.shape[0] - 1)
            lab = labels[pi]
            offsets = []
            for row in range(h_b):
                for col in range(w_b):
                    sr0 = int(row * (args.enc_h / h_b) / (args.enc_h / seg_h))
                    sr1 = int((row + 1) * (args.enc_h / h_b) / (args.enc_h / seg_h))
                    sc0 = int(col * (args.enc_w / w_b) / (args.enc_w / seg_w))
                    sc1 = int((col + 1) * (args.enc_w / w_b) / (args.enc_w / seg_w))
                    sr0, sr1 = max(0, min(sr0, seg_h - 1)), max(1, min(sr1, seg_h))
                    sc0, sc1 = max(0, min(sc0, seg_w - 1)), max(1, min(sc1, seg_w))
                    cell = lab[sr0:sr1, sc0:sc1]
                    if cell.size == 0:
                        offsets.append("0")
                        continue
                    uniq = np.unique(cell)
                    sky_frac = (cell == 2).sum() / cell.size if 2 in cell else 0
                    road_frac = (cell == 0).sum() / cell.size
                    if sky_frac > 0.9:
                        offsets.append("5")
                    elif len(uniq) > 1 and road_frac > 0.1:
                        offsets.append("-5")
                    else:
                        offsets.append("0")
            f.write(f"{frame} " + " ".join(offsets) + "\n")


if __name__ == "__main__":
    main()
