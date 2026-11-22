"""
Ask comma's judges what they see in the original video.

1. Read video 0 into 1,200 frames -> 600 pairs.
2. Show every pair to SegNet and PoseNet.
3. Save their answers to out/targets.npz.
"""

import sys
import time
from pathlib import Path
import numpy as np

import av
import torch

# "judges" = comma's SegNet and PoseNet, the two networks that score the video

# comma's files (frame_utils.py, modules.py, videos/) are two folders up
CHALLENGE_FOLDER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHALLENGE_FOLDER))

from frame_utils import yuv420_to_rgb
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

VIDEO_PATH = CHALLENGE_FOLDER / "videos" / "0.mkv"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"

NUMBER_OF_PAIRS = 600
MAP_HEIGHT, MAP_WIDTH = 384, 512   # SegNet's map size
PAIRS_PER_BATCH = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TF32 off, so the answers match comma's evaluation
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_video(video_path):
    """Read every frame of the video as RGB.

    Input:
        video_path: path to 0.mkv.

    Output:
        frames: list of 1,200 frames, each (height, width, rgb: 0-255).
    """
    video_file = av.open(str(video_path))
    video_stream = video_file.streams.video[0]

    frames = []
    for frame in video_file.decode(video_stream):
        frames.append(yuv420_to_rgb(frame))
    video_file.close()

    if len(frames) != 2 * NUMBER_OF_PAIRS:
        raise ValueError("Expected 1,200 frames but got " + str(len(frames)))

    return frames


def load_judges():
    """Load comma's SegNet + PoseNet, with their trained weights."""
    judges = DistortionNet().to(DEVICE)
    judges.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    judges.eval()
    return judges


def judge_all_pairs(frames, judges):
    """Show the judges every pair and keep their answers.

    Input:
        frames: list of 1,200 frames, each (height, width, rgb: 0-255).
        judges: comma's SegNet + PoseNet.

    Output:
        seg_arr: (600, 384, 512) uint8, SegNet's label for every pixel.
        pose_arr: (600, 6) float32, PoseNet's motion for every pair.
    """
    seg_arr = np.zeros((NUMBER_OF_PAIRS, MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)
    pose_arr = np.zeros((NUMBER_OF_PAIRS, 6), dtype=np.float32)

    with torch.inference_mode():
        for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            batch_end = batch_start + PAIRS_PER_BATCH

            # frames -> pairs -> one batch of shape (pair, frame, height, width, rgb)
            batch = []
            for pair in range(batch_start, batch_end):
                batch.append(torch.stack([frames[2 * pair], frames[2 * pair + 1]]))
            batch = torch.stack(batch).to(DEVICE)

            posenet_answer, segnet_answer = judges(batch)
            
            # SegNet gives 5 scores per pixel, keep the class with the highest score
            labels = segnet_answer.argmax(dim=1)
            seg_arr[batch_start:batch_end] = labels.to(torch.uint8).cpu().numpy()

            # PoseNet per pair -> (pair, (forward, sideways, up/down, roll, pitch, yaw))
            motion = posenet_answer["pose"][:, :6]
            pose_arr[batch_start:batch_end] = motion.float().cpu().numpy()

    return seg_arr, pose_arr


def save_targets(seg_arr, pose_arr):
    """Save the maps and poses to out/targets.npz."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    targets_path = OUTPUT_FOLDER / "targets.npz"
    np.savez_compressed(targets_path, maps=seg_arr, poses=pose_arr)
    return targets_path


def main():
    start_time = time.time()

    frames = load_video(VIDEO_PATH)
    judges = load_judges()
    seg_arr, pose_arr = judge_all_pairs(frames, judges)
    targets_path = save_targets(seg_arr, pose_arr)

    print("Saved", targets_path, "in", round(time.time() - start_time), "s")


if __name__ == "__main__":
    main()
