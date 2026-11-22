"""
Pack everything comma needs into archive.zip, as small as possible.

1. maps.bin: the cleaned maps, coded by the map encoder.
2. weights.bin: the frame model's weights, rounded the same way they were trained.
3. pose.bin: the 6 motion values of every pair, 1 byte each.
4. correction.bin: frame 1's pose correction.

Inflate reads every piece back in the same order.
"""

import struct
import zipfile
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch

import frame_model
import map_encoder

SUBMISSION_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = SUBMISSION_FOLDER / "out"
MODEL_FILE = OUTPUT_FOLDER / "stage_head.pt"
CLEAN_MAPS_FILE = OUTPUT_FOLDER / "clean_maps.npz"
TARGETS_FILE = OUTPUT_FOLDER / "targets.npz"
CORRECTION_FILE = OUTPUT_FOLDER / "correction.npz"
ARCHIVE_FILE = SUBMISSION_FOLDER / "archive.zip"

# must match the training rounding exactly
WEIGHT_STEPS = 15                  # 5-bit: -15 to 15
SCALE_STEPS_PER_DOUBLING = 8       # row scales are 2^(k/8): 8 steps between each doubling
FRAME1_PARTS = ("pose_to_code", "pose_correction", "draw_frame1")   # the frame 1 head stays 8-bit
K_OFFSET = 200                     # k is stored as k + 200 so it fits in 1 byte


def row_scale_powers(rows, steps):
    """Pick the smallest scale for each row that still fits its largest weight."""
    biggest = np.abs(rows).max(axis=1)
    biggest = np.maximum(biggest, 1e-30)
    k = np.ceil(np.log2(biggest / steps) * SCALE_STEPS_PER_DOUBLING)
    k = np.clip(k, -200, 55)
    return k.astype(np.int64)


def round_rows(rows, k, steps):
    """Divide each row by its scale and round to whole numbers."""
    scale = 2.0 ** (k / SCALE_STEPS_PER_DOUBLING)
    whole_numbers = np.round(rows / scale[:, None])
    whole_numbers = np.clip(whole_numbers, -steps, steps)
    return whole_numbers


def quantize_weights(state):
    """Round every weight to whole numbers.

    Input:
        state: the model's weights, name -> tensor.

    Output:
        row_powers: uint8, one k + 200 per row (5-bit layers and the 8-bit frame 1 head).
        list_scales: float32, one scale per bias / norm list.
        eight_bit_values: int8, the frame 1 head + every bias / norm.
        five_bit_values: int64, every other layer.
    """
    row_powers = []
    list_scales = []
    eight_bit_values = []
    five_bit_values = []

    # always sorted by name, so inflate reads the layers back in the same order
    for name in sorted(state):
        weight = state[name].detach().double().cpu().numpy()

        if weight.ndim >= 2:
            rows = weight.reshape(weight.shape[0], -1)

            if name.startswith(FRAME1_PARTS):
                k = row_scale_powers(rows, 127)
                whole_numbers = round_rows(rows, k, 127)
                eight_bit_values.append(whole_numbers.astype(np.int8).ravel())
            else:
                k = row_scale_powers(rows, WEIGHT_STEPS)
                whole_numbers = round_rows(rows, k, WEIGHT_STEPS)
                five_bit_values.append(whole_numbers.astype(np.int64).ravel())
            row_powers.append(k)

        else:
            # biases and norms are tiny, so they stay 8-bit
            scale = np.float32(max(np.abs(weight).max() / 127, 1e-12))
            whole_numbers = np.clip(np.round(weight / float(scale)), -127, 127)
            list_scales.append(scale)
            eight_bit_values.append(whole_numbers.astype(np.int8))

    row_powers = (np.concatenate(row_powers) + K_OFFSET).astype(np.uint8)
    list_scales = np.array(list_scales, np.float32)
    eight_bit_values = np.concatenate(eight_bit_values)
    five_bit_values = np.concatenate(five_bit_values)
    return row_powers, list_scales, eight_bit_values, five_bit_values


def dequantize_weights(row_powers, list_scales, eight_bit_values, five_bit_values, template):
    """Turn the whole numbers back into weights."""
    state = {}
    row_offset = 0
    list_offset = 0
    eight_bit_offset = 0
    five_bit_offset = 0

    for name in sorted(template):
        shape_of = template[name]
        count = shape_of.numel()

        if shape_of.dim() >= 2:
            number_of_rows = shape_of.shape[0]
            k = row_powers[row_offset:row_offset + number_of_rows].astype(np.float64) - K_OFFSET
            scale = 2.0 ** (k / SCALE_STEPS_PER_DOUBLING)
            row_offset += number_of_rows

            if name.startswith(FRAME1_PARTS):
                whole_numbers = eight_bit_values[eight_bit_offset:eight_bit_offset + count].astype(np.float64)
                eight_bit_offset += count
            else:
                whole_numbers = five_bit_values[five_bit_offset:five_bit_offset + count].astype(np.float64)
                five_bit_offset += count
            values = whole_numbers.reshape(number_of_rows, -1) * scale[:, None]

        else:
            whole_numbers = eight_bit_values[eight_bit_offset:eight_bit_offset + count].astype(np.float64)
            values = whole_numbers * float(list_scales[list_offset])
            eight_bit_offset += count
            list_offset += 1

        state[name] = torch.from_numpy(values.astype(np.float32)).view_as(shape_of)

    return state


def shipped_weights(model_file):
    """Load a saved model with its weights rounded like the archive."""
    state = torch.load(model_file, map_location="cpu", weights_only=True)
    template = frame_model.FrameModel().state_dict()
    row_powers, list_scales, eight_bit_values, five_bit_values = quantize_weights(state)
    return dequantize_weights(row_powers, list_scales, eight_bit_values, five_bit_values, template)


def histogram_to_odds(histogram):
    """Turn a histogram into odds for the range coder."""
    odds = histogram.astype(np.float64) + 1e-3
    return odds / odds.sum()


def pack_weights():
    """Pack the trained model's weights into weights.bin."""
    state = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    row_powers, list_scales, eight_bit_values, five_bit_values = quantize_weights(state)

    # count each 5-bit value, so common ones cost fewer bits
    symbols = five_bit_values + WEIGHT_STEPS
    histogram = np.bincount(symbols, minlength=2 * WEIGHT_STEPS + 1).astype(np.uint32)
    odds = constriction.stream.model.Categorical(histogram_to_odds(histogram), perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols.astype(np.int32), odds)
    five_bit_bytes = encoder.get_compressed().tobytes()

    side_part = list_scales.tobytes() + row_powers.tobytes() + eight_bit_values.tobytes() + histogram.tobytes()
    side_part = brotli.compress(side_part, quality=11, lgwin=22)

    # weights.bin = [side part size] + [side part] + [5-bit values]
    return struct.pack("<I", len(side_part)) + side_part + five_bit_bytes


def pack_maps():
    """Code the cleaned maps into maps.bin."""
    maps = np.load(CLEAN_MAPS_FILE)["maps"]
    return map_encoder.encode(maps)


def pack_pose():
    """Pack the motion values of all 600 pairs into pose.bin."""
    pose_arr = np.load(TARGETS_FILE)["poses"].astype(np.float32)

    # round each motion value to 1 byte, with one scale per column
    scales = (np.abs(pose_arr).max(axis=0) / 127).astype(np.float32)
    whole_numbers = np.clip(np.round(pose_arr / scales), -127, 127).astype(np.int8)

    return brotli.compress(scales.tobytes() + whole_numbers.tobytes(), quality=11)


def pack_correction():
    """Pack frame 1's pose correction into correction.bin."""
    correction = np.load(CORRECTION_FILE)
    patterns = correction["patterns"].astype(np.int8)
    amounts = correction["amounts"].astype(np.int8)
    number_of_patterns, _, height, width = patterns.shape

    header = struct.pack("<HHH", number_of_patterns, height, width)
    strength_and_scale = np.array([correction["strength"], correction["pattern_scale"]], np.float32).tobytes()
    amount_scales = correction["amount_scales"].astype(np.float32).tobytes()

    return brotli.compress(header + strength_and_scale + amount_scales + patterns.tobytes() + amounts.tobytes(), quality=11)


def main():
    """Pack all 4 pieces into archive.zip and print their sizes."""
    pieces = {
        "maps.bin": pack_maps(),
        "weights.bin": pack_weights(),
        "pose.bin": pack_pose(),
        "correction.bin": pack_correction(),
    }

    # fixed dates so the same pieces always give the same zip
    with zipfile.ZipFile(ARCHIVE_FILE, "w", zipfile.ZIP_STORED) as archive:
        for name, data in pieces.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)

    for name, data in pieces.items():
        print(f"{name:15s} {len(data):>9,} bytes")

    total = ARCHIVE_FILE.stat().st_size
    size_part = 25 * total / 37_545_489
    print(f"archive.zip     {total:>9,} bytes   size part of the score: {size_part:.4f}")


if __name__ == "__main__":
    main()
