import av
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    container = av.open(args.input)
    stream = container.streams.video[0]
    
    # Target size required by evaluate.py
    target_w, target_h = 1164, 874
    
    # Simple sharpening kernel
    sharpen_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=torch.float32).view(1, 1, 3, 3)

    with open(args.output, "wb") as f:
        for frame in container.decode(stream):
            # 1. Convert to RGB Tensor
            img = torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1).float().unsqueeze(0)
            
            # 2. Upscale back to original resolution
            img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # 3. Apply Sharpening (Helps SegNet classification)
            # Apply per channel
            sharpened = []
            for c in range(3):
                chan = img[:, c:c+1, :, :]
                chan = F.conv2d(chan, sharpen_kernel, padding=1)
                sharpened.append(chan)
            img = torch.cat(sharpened, dim=1).clamp(0, 255)
            
            # 4. Save as raw uint8 RGB
            raw_frame = img.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            f.write(raw_frame.tobytes())
            
    container.close()

if __name__ == "__main__":
    main()