"""
Fix frame 1 so PoseNet reads the right motion. Frame 2 never changes, so SegNet can't get worse.

1. Head: retrain the frame 1 head on the shipped model, since rounding the rest of the model shifts frame 1 a little.
2. Correction: add 4 faint patterns to frame 1, with different amounts for every pair.
3. Fit the correction again with 8-bit rounding on, the way the archive stores it.
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_frame_model
import frame_model
import pack

OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"
CLEAN_MAPS_FILE = OUTPUT_FOLDER / "clean_maps.npz"
DEVICE = train_frame_model.DEVICE
NUMBER_OF_PAIRS = 600
PAIRS_PER_BATCH = 8

PATTERNS = 4                                    # patterns shared by every pair
PATTERN_HEIGHT, PATTERN_WIDTH = 12, 16          # tiny, scaled up to 384 x 512
STRENGTH = 10.0                                 # how strong the correction can be


def load_clean_maps():
    """Load the cleaned maps, the ones that get shipped."""
    maps = np.load(CLEAN_MAPS_FILE)["maps"]
    return torch.from_numpy(maps).long().to(DEVICE)


def frame2_at_camera_size(model, seg_arr, quantized_pose):
    """Draw frame 2 once at camera size.

    Output:
        frame2_camera: (600, 3, 874, 1164) uint8, frame 2 exactly as inflate writes it.
    """
    frames = torch.empty(NUMBER_OF_PAIRS, 3, frame_model.CAMERA_HEIGHT, frame_model.CAMERA_WIDTH, dtype=torch.uint8, device=DEVICE)
    with torch.no_grad():
        for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            batch_end = batch_start + PAIRS_PER_BATCH
            _, frame2 = model(seg_arr[batch_start:batch_end], quantized_pose[batch_start:batch_end])
            frames[batch_start:batch_end] = frame_model.to_camera(frame2).to(torch.uint8)
    return frames


def frame1_weights_only(model):
    """Get the frame 1 head's real weights, named like the saved model."""
    weights = {}
    for key, weight in model.state_dict().items():
        if ".parametrizations." in key and key.endswith(".original"):
            name = key.replace(".parametrizations.", ".").replace(".original", "")
            weights[name] = weight.detach().cpu().clone()
    return weights


def retrain_head(epochs, learning_rate=5e-4, pairs_per_batch=4, check_every=10):
    """Retrain the frame 1 head on the shipped model and save the best to out/stage_head.pt."""
    torch.manual_seed(0)
    _, pose_arr = train_frame_model.load_targets()
    seg_arr = load_clean_maps()
    quantized_pose = train_frame_model.quantize_pose(pose_arr)
    judges = train_frame_model.load_judges()

    # frame 1 starts from its trained weights, the rest uses its shipped weights
    trained = torch.load(OUTPUT_FOLDER / "stage_pose.pt", map_location="cpu", weights_only=True)
    shipped = pack.shipped_weights(OUTPUT_FOLDER / "stage_pose.pt")
    start_weights = {}
    for name in shipped:
        if name.startswith(train_frame_model.FRAME1_PARTS):
            start_weights[name] = trained[name]
        else:
            start_weights[name] = shipped[name]

    model = frame_model.FrameModel()
    model.load_state_dict(start_weights)
    model = model.to(DEVICE)
    train_frame_model.rounding_on(model, only_frame1=True)

    trainable = []
    for name, weight in model.named_parameters():
        is_frame1 = name.startswith(train_frame_model.FRAME1_PARTS)
        weight.requires_grad_(is_frame1)
        if is_frame1:
            trainable.append(weight)

    print("drawing frame 2 once...", flush=True)
    model.eval()
    frame2_camera = frame2_at_camera_size(model, seg_arr, quantized_pose)

    def pose_error(pairs):
        """Score PoseNet's error on these pairs."""
        frame1, _ = model(seg_arr[pairs], quantized_pose[pairs])
        camera1 = train_frame_model.to_camera_keep_gradient(frame1)
        motion = train_frame_model.ask_posenet(judges, camera1, frame2_camera[pairs].float())
        return (motion - pose_arr[pairs]).pow(2).mean(dim=1)

    def check():
        """Score all 600 pairs."""
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
                pairs = torch.arange(batch_start, batch_start + PAIRS_PER_BATCH, device=DEVICE)
                total += pose_error(pairs).sum().item()
        model.train()
        return total / NUMBER_OF_PAIRS

    def save():
        """Save the frame 1 head's new weights with the rest of the model."""
        weights = dict(trained)
        weights.update(frame1_weights_only(model))
        torch.save(weights, OUTPUT_FOLDER / "stage_head.pt")

    number_trainable = 0
    for weight in trainable:
        number_trainable += weight.numel()
    best = check()
    save()
    print(f"start: pose part {math.sqrt(10 * best):.4f}  (training {number_trainable:,} frame 1 weights)", flush=True)

    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    warm_up_epochs = max(1, epochs // 30)
    start_time = time.time()

    for epoch in range(epochs):
        warm_up = min(1.0, (epoch + 1) / warm_up_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * warm_up * cosine

        order = torch.randperm(NUMBER_OF_PAIRS, device=DEVICE)
        for batch_start in range(0, NUMBER_OF_PAIRS - pairs_per_batch + 1, pairs_per_batch):
            pairs = order[batch_start:batch_start + pairs_per_batch]
            loss = torch.log(pose_error(pairs).mean() + 1e-9)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        if epoch % check_every == check_every - 1 or epoch == epochs - 1:
            error = check()
            seconds = time.time() - start_time
            print(f"epoch {epoch + 1}  {seconds:.0f}s  pose part {math.sqrt(10 * error):.4f}", flush=True)
            if error < best:
                best = error
                save()
                print("  saved best", flush=True)

    print(f"best pose part {math.sqrt(10 * best):.4f} -> out/stage_head.pt", flush=True)


def round_keep_gradient(values):
    """Round to whole numbers, but let the gradient through as if nothing was rounded."""
    return values + (values.round() - values).detach()


def fit_correction(epochs, rounded_epochs=200, learning_rate=0.01):
    """Fit the pose correction against PoseNet and save it to out/correction.npz."""
    torch.manual_seed(0)
    _, pose_arr = train_frame_model.load_targets()
    seg_arr = load_clean_maps()
    quantized_pose = train_frame_model.quantize_pose(pose_arr)
    judges = train_frame_model.load_judges()

    # draw both frames once with the shipped model
    model = frame_model.FrameModel()
    model.load_state_dict(pack.shipped_weights(OUTPUT_FOLDER / "stage_head.pt"))
    model = model.to(DEVICE).eval()
    print("drawing both frames once...", flush=True)

    frame1_base = torch.empty(NUMBER_OF_PAIRS, 3, frame_model.MAP_HEIGHT, frame_model.MAP_WIDTH, dtype=torch.float16, device=DEVICE)
    with torch.no_grad():
        for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            batch_end = batch_start + PAIRS_PER_BATCH
            frame1, _ = model(seg_arr[batch_start:batch_end], quantized_pose[batch_start:batch_end])
            frame1_base[batch_start:batch_end] = frame1.half()
    frame2_camera = frame2_at_camera_size(model, seg_arr, quantized_pose)
    del model
    torch.cuda.empty_cache()

    # amounts start near 0, so there's almost no correction at first
    patterns = torch.nn.Parameter(torch.randn(PATTERNS, 3, PATTERN_HEIGHT, PATTERN_WIDTH, device=DEVICE))
    amounts = torch.nn.Parameter(torch.randn(NUMBER_OF_PAIRS, PATTERNS, device=DEVICE) * 1e-3)

    def as_stored(rounded):
        """Round the patterns and amounts to 8-bit, the way the archive stores them."""
        if not rounded:
            return patterns, amounts
        pattern_scale = patterns.detach().abs().max() / 127
        amount_scales = amounts.detach().abs().max(dim=0).values.clamp_min(1e-8) / 127
        pattern_values = round_keep_gradient(patterns / pattern_scale).clamp(-127, 127) * pattern_scale
        amount_values = round_keep_gradient(amounts / amount_scales).clamp(-127, 127) * amount_scales
        return pattern_values, amount_values

    def motion_of(pairs, pattern_values, amount_values):
        """Add the pose correction to frame 1 and ask PoseNet for the motion of these pairs."""
        correction = torch.einsum("nk,kchw->nchw", amount_values[pairs], pattern_values)
        correction = F.interpolate(correction, size=(frame_model.MAP_HEIGHT, frame_model.MAP_WIDTH), mode="bilinear", align_corners=False)
        frame1 = frame1_base[pairs].float() + STRENGTH * correction
        camera1 = train_frame_model.to_camera_keep_gradient(frame1)
        return train_frame_model.ask_posenet(judges, camera1, frame2_camera[pairs].float())

    def check(rounded):
        """Score all 600 pairs, with or without the rounding."""
        total = 0.0
        with torch.no_grad():
            pattern_values, amount_values = as_stored(rounded)
            for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
                pairs = torch.arange(batch_start, min(NUMBER_OF_PAIRS, batch_start + PAIRS_PER_BATCH), device=DEVICE)
                motion = motion_of(pairs, pattern_values, amount_values)
                total += (motion - pose_arr[pairs]).pow(2).mean(dim=1).sum().item()
        return total / NUMBER_OF_PAIRS

    def fit(epochs, start_lr, rounded, label):
        """Fit the patterns and amounts, with or without the rounding."""
        optimizer = torch.optim.Adam([patterns, amounts], start_lr)
        start_time = time.time()
        for epoch in range(epochs):
            for group in optimizer.param_groups:
                group["lr"] = start_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))

            order = torch.randperm(NUMBER_OF_PAIRS, device=DEVICE)
            for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
                pairs = order[batch_start:batch_start + PAIRS_PER_BATCH]
                pattern_values, amount_values = as_stored(rounded)
                motion = motion_of(pairs, pattern_values, amount_values)
                loss = (motion - pose_arr[pairs]).pow(2).mean()
                optimizer.zero_grad(set_to_none=True)
                torch.log(loss + 1e-9).backward()
                optimizer.step()

            if epoch % 10 == 9 or epoch == epochs - 1:
                seconds = time.time() - start_time
                print(f"{label} epoch {epoch + 1}  {seconds:.0f}s  pose part {math.sqrt(10 * check(rounded)):.4f}", flush=True)

    print(f"no correction: pose part {math.sqrt(10 * check(False)):.4f}", flush=True)
    fit(epochs, learning_rate, False, "normal")
    fit(rounded_epochs, learning_rate / 5, True, "8-bit")

    # save what gets packed
    with torch.no_grad():
        pattern_scale = (patterns.abs().max() / 127).item()
        amount_scales = amounts.abs().max(dim=0).values.clamp_min(1e-8) / 127
        pattern_whole = (patterns / pattern_scale).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
        amount_whole = (amounts / amount_scales[None]).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
    np.savez(OUTPUT_FOLDER / "correction.npz",
             patterns=pattern_whole, pattern_scale=np.float32(pattern_scale),
             amounts=amount_whole, amount_scales=amount_scales.cpu().numpy().astype(np.float32),
             strength=np.float32(STRENGTH))
    print(f"saved out/correction.npz   pose part {math.sqrt(10 * check(True)):.4f}", flush=True)


if __name__ == "__main__":
    part = sys.argv[1]       # "head" or "correction"

    if part == "head":
        epochs = 600
    else:
        epochs = 4800
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])

    if part == "head":
        retrain_head(epochs)
    else:
        rounded_epochs = 200
        if len(sys.argv) > 3:
            rounded_epochs = int(sys.argv[3])
        fit_correction(epochs, rounded_epochs)
