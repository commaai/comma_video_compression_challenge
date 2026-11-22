"""
Train the frame model until SegNet and PoseNet give the same answers on its frames as on the real video.

1. Seg stage: the whole model learns until SegNet's label for every pixel of frame 2 matches.
2. Pose stage: only the frame 1 head learns, until PoseNet's motion for every pair matches.
3. Every step trains on the same rounded weights the archive stores.
"""

import sys
import math
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

# comma's files (frame_utils.py, modules.py) are two folders up
CHALLENGE_FOLDER = Path(__file__).resolve().parents[2]
sys.path.append(str(CHALLENGE_FOLDER))

import frame_utils
import modules
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

import frame_model
OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"
NUMBER_OF_PAIRS = 600
PAIRS_PER_BATCH = 4
DEVICE = "cuda"

# per stage: [epochs, learning rate]
STAGES = {
    "seg": [600, 1e-3],
    "pose": [500, 5e-4],
}

WRONG_PIXEL_WEIGHT = 10.0    # pixels SegNet gets wrong count 10x in the loss
WEIGHT_STEPS = 15            # 5-bit: -15 to 15
FRAME1_PARTS = ("pose_to_code", "pose_correction", "draw_frame1")   # the frame 1 head stays 8-bit

# TF32 off, so the answers match comma's evaluation
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# remove comma's no-gradient lock so PoseNet's feedback reaches our model (training only)
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__


def load_targets():
    """Load the judges' maps and motion values from out/targets.npz.

    Output:
        seg_arr: (600, 384, 512) long, SegNet's label for every pixel.
        pose_arr: (600, 6) float32, PoseNet's motion for every pair.
    """
    data = np.load(OUTPUT_FOLDER / "targets.npz")
    seg_arr = torch.from_numpy(data["maps"]).long().to(DEVICE)
    pose_arr = torch.from_numpy(data["poses"]).float().to(DEVICE)
    return seg_arr, pose_arr


def load_judges():
    """Load comma's SegNet + PoseNet, with frozen weights."""
    judges = DistortionNet().eval().to(DEVICE)
    judges.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for weight in judges.parameters():
        weight.requires_grad_(False)
    return judges


def quantize_pose(pose_arr):
    """Round each motion value to 1 byte with one scale per column, the same as pose.bin."""
    scale = pose_arr.abs().max(dim=0).values / 127

    stored_bytes = (pose_arr / scale).round().clamp(-127, 127)
    return stored_bytes * scale


class RoundLikeShipped(nn.Module):
    """Trains with the same rounded weights the archive stores, while learning updates the real weights underneath."""

    def __init__(self, eight_bit):
        super().__init__()
        self.eight_bit = eight_bit

    def forward(self, weight):
        if self.eight_bit:
            steps = 127
        else:
            steps = WEIGHT_STEPS

        if weight.dim() >= 2:
            # one scale per row, the smallest that fits the row's largest weight within 15 (or 127 for 8-bit)
            real = weight.detach().double()
            biggest = real.abs().flatten(1).max(dim=1).values
            biggest = biggest.clamp_min(1e-30)
            k = torch.ceil(torch.log2(biggest / steps) * 8)
            k = k.clamp(-200, 55)
            scale = 2.0 ** (k / 8)
            scale = scale.view(-1, *[1] * (weight.dim() - 1))

            rounded = (real / scale).round().clamp(-steps, steps)
            rounded = (rounded * scale).float()
        else:
            # biases and norms are tiny, so they stay 8-bit
            scale = weight.detach().abs().max().clamp(min=1e-12) / 127
            rounded = (weight.detach() / scale).round().clamp(-127, 127)
            rounded = rounded * scale

        # forward uses the rounded weight, the gradient goes to the real one
        return weight + (rounded - weight).detach()


def rounding_on(model, only_frame1=False):
    """Make every layer use its rounded weights."""
    for name, layer in list(model.named_modules()):
        # skip the rounding's own helper layers
        if type(layer).__name__ in ("ParametrizationList", "RoundLikeShipped"):
            continue
        if only_frame1 and not name.startswith(FRAME1_PARTS):
            continue

        # the frame 1 head stays 8-bit because PoseNet notices small errors
        eight_bit = name.startswith(FRAME1_PARTS)
        for weight_name, _ in list(layer.named_parameters(recurse=False)):
            parametrize.register_parametrization(layer, weight_name, RoundLikeShipped(eight_bit))


def rounding_off(model):
    """Take the rounding off but keep the real weights."""
    for _, layer in list(model.named_modules()):
        if parametrize.is_parametrized(layer):
            for weight_name in list(layer.parametrizations.keys()):
                # keep the real weights, not the rounded copy
                parametrize.remove_parametrizations(layer, weight_name, leave_parametrized=False)


def to_camera_keep_gradient(frame):
    """Stretch a frame to camera size like to_camera, but let the gradient pass through the rounding."""
    camera_frame = F.interpolate(
        frame,
        size=(frame_model.CAMERA_HEIGHT, frame_model.CAMERA_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    camera_frame = camera_frame.clamp(0, 255)

    # use the rounded value, but let the gradient pass as if nothing was rounded
    rounding = (camera_frame.round() - camera_frame).detach()
    return camera_frame + rounding


def ask_segnet(judges, frame):
    """Ask SegNet what it sees in one of our frames.

    Input:
        judges: comma's SegNet + PoseNet, with frozen weights.
        frame: (pair, 3, 384, 512) float32, our drawn frame, RGB pixels.

    Output:
        scores: (pair, 5, 384, 512) float32, 5 class scores per pixel (highest score wins).
    """
    camera_frame = to_camera_keep_gradient(frame)
    camera_frame = camera_frame[:, None]      # SegNet wants a "frames" slot, and 1 frame is enough
    segnet_input = judges.segnet.preprocess_input(camera_frame)
    return judges.segnet(segnet_input)


def ask_posenet(judges, camera1, camera2):
    """Ask PoseNet how the camera moved between two camera-size frames.

    Input:
        judges: comma's SegNet + PoseNet, with frozen weights.
        camera1: (pair, 3, 874, 1164) float32, frame 1 at camera size, RGB pixels.
        camera2: same shape, frame 2.

    Output:
        motion: (pair, 6) float32, PoseNet's motion for every pair.
    """
    pair = torch.stack([camera1, camera2], dim=1)
    posenet_input = judges.posenet.preprocess_input(pair)
    answer = judges.posenet(posenet_input)
    return answer["pose"][:, :6]


def segnet_loss(segnet_scores, target_map):
    """How wrong SegNet is about our frame, with wrong pixels counting 10x."""
    per_pixel = F.cross_entropy(segnet_scores, target_map, reduction="none")

    with torch.no_grad():
        wrong = segnet_scores.argmax(dim=1) != target_map
        weight = torch.where(wrong, WRONG_PIXEL_WEIGHT, 1.0)

    return (per_pixel * weight).mean()


def posenet_loss(posenet_motion, target_motion):
    """How wrong PoseNet is about the motion, as a log."""
    squared_error = (posenet_motion - target_motion).pow(2).mean()
    return torch.log(squared_error + 1e-7)


def check_score(model, judges, seg_arr, quantized_pose, pose_arr):
    """Score all 600 pairs the way comma does.

    Input:
        model: our model mid-training.
        judges: comma's SegNet + PoseNet, with frozen weights.
        seg_arr: (600, 384, 512) long, SegNet's label for every pixel.
        quantized_pose: (600, 6) float32, PoseNet's motion for every pair, rounded to 1 byte.
        pose_arr: (600, 6) float32, PoseNet's real motion for every pair.

    Output:
        segnet_error: fraction of pixels SegNet labels wrong, averaged over the pairs.
        posenet_error: PoseNet's squared motion error, averaged over the pairs.
    """
    model.eval()
    total_wrong_pixels = 0.0
    total_pose_error = 0.0

    with torch.no_grad():
        for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            batch_end = batch_start + PAIRS_PER_BATCH
            maps = seg_arr[batch_start:batch_end]
            motions = quantized_pose[batch_start:batch_end]
            real_motions = pose_arr[batch_start:batch_end]

            frame1, frame2 = model(maps, motions)

            # how many pixels SegNet gets wrong in each pair
            labels = ask_segnet(judges, frame2).argmax(dim=1)
            wrong_per_pair = (labels != maps).float().mean(dim=(1, 2))
            total_wrong_pixels += wrong_per_pair.sum().item()

            # how far PoseNet's motion is from the real motion in each pair
            camera1 = to_camera_keep_gradient(frame1)
            camera2 = to_camera_keep_gradient(frame2)
            motion = ask_posenet(judges, camera1, camera2)
            error_per_pair = (motion - real_motions).pow(2).mean(dim=1)
            total_pose_error += error_per_pair.sum().item()

    model.train()

    segnet_error = total_wrong_pixels / NUMBER_OF_PAIRS
    posenet_error = total_pose_error / NUMBER_OF_PAIRS
    return segnet_error, posenet_error


def train(stage, epochs):
    """Train one stage and save its weights to out/stage_<stage>.pt."""
    torch.manual_seed(0)    # same random start and order every run

    seg_arr, pose_arr = load_targets()
    quantized_pose = quantize_pose(pose_arr)
    judges = load_judges()

    # seg stage starts from scratch, pose stage starts from the seg result and only trains the frame 1 head
    model = frame_model.FrameModel().to(DEVICE)
    if stage == "pose":
        weights = torch.load(OUTPUT_FOLDER / "stage_seg.pt", map_location=DEVICE, weights_only=True)
        model.load_state_dict(weights)
        for name, weight in model.named_parameters():
            if not name.startswith(FRAME1_PARTS):
                weight.requires_grad_(False)

    rounding_on(model)

    trainable = []
    for weight in model.parameters():
        if weight.requires_grad:
            trainable.append(weight)

    number_trainable = 0
    for weight in trainable:
        number_trainable += weight.numel()

    number_total = 0
    for weight in model.parameters():
        number_total += weight.numel()

    learning_rate = STAGES[stage][1]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0)

    steps_per_epoch = NUMBER_OF_PAIRS // PAIRS_PER_BATCH
    total_steps = epochs * steps_per_epoch
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    print(f"stage {stage}: {epochs} epochs, learning rate {learning_rate}, training {number_trainable:,} of {number_total:,} weights")

    fade_epochs = max(1, epochs // 3)
    start_time = time.time()
    for epoch in range(epochs):
        # early on, also train frame 1 to look like the map, so the pose stage starts from a real picture
        fade = 1.0 - epoch / fade_epochs
        if fade < 0.0:
            fade = 0.0

        order = torch.randperm(NUMBER_OF_PAIRS, device=DEVICE)

        for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            pairs = order[batch_start:batch_start + PAIRS_PER_BATCH]
            maps = seg_arr[pairs]
            motions = quantized_pose[pairs]
            real_motions = pose_arr[pairs]

            frame1, frame2 = model(maps, motions)

            # seg stage: SegNet's error on frame 2 (plus frame 1 early on)
            # pose stage: PoseNet's error on the pair, with frame 2 held fixed so only frame 1 learns
            if stage == "seg":
                loss = segnet_loss(ask_segnet(judges, frame2), maps)
                loss = loss + fade * segnet_loss(ask_segnet(judges, frame1), maps)
            else:
                camera1 = to_camera_keep_gradient(frame1)
                camera2 = to_camera_keep_gradient(frame2.detach())
                motion = ask_posenet(judges, camera1, camera2)
                loss = posenet_loss(motion, real_motions)
                loss = loss + 0.1 * fade * segnet_loss(ask_segnet(judges, frame1), maps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)    # no single step too big
            optimizer.step()
            schedule.step()

        if epoch % 5 == 4 or epoch == epochs - 1:
            segnet_error, posenet_error = check_score(model, judges, seg_arr, quantized_pose, pose_arr)
            seg_part = 100 * segnet_error
            pose_part = math.sqrt(10 * posenet_error)
            seconds = time.time() - start_time
            print(f"epoch {epoch + 1:3d}  {seconds:5.0f}s  100*seg {seg_part:.4f}  sqrt(10*pose) {pose_part:.4f}", flush=True)

    rounding_off(model)
    save_file = OUTPUT_FOLDER / f"stage_{stage}.pt"
    torch.save(model.state_dict(), save_file)
    print("Saved", save_file)


if __name__ == "__main__":
    stage = sys.argv[1]    # "seg" or "pose"

    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    else:
        epochs = STAGES[stage][0]

    train(stage, epochs)