"""
Rebuild the video from archive.zip on comma's computer.

1. Unpack the maps, the weights, the motion values and the pose correction.
2. Draw every pair with the frame model, then add the pose correction to frame 1.
3. Stretch both frames to camera size and write them to 0.raw.
"""

import sys
import time
import struct
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch
import torch.nn.functional as F

SUBMISSION_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_FOLDER))
import frame_model
import pack
import map_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUMBER_OF_PAIRS = 600
PAIRS_PER_BATCH = 8


def unpack_model(archive_folder):
    """Rebuild the frame model from weights.bin."""
    data = (archive_folder / "weights.bin").read_bytes()
    side_size = struct.unpack("<I", data[:4])[0]
    side_part = brotli.decompress(data[4:4 + side_size])
    five_bit_bytes = data[4 + side_size:]

    # count how many values of each kind were packed
    model = frame_model.FrameModel()
    template = model.state_dict()
    number_of_lists = 0
    number_of_rows = 0
    number_eight_bit = 0
    number_total = 0
    for name, weight in template.items():
        number_total += weight.numel()
        if weight.dim() >= 2:
            number_of_rows += weight.shape[0]
            if name.startswith(pack.FRAME1_PARTS):
                number_eight_bit += weight.numel()
        else:
            number_of_lists += 1
            number_eight_bit += weight.numel()
    number_five_bit = number_total - number_eight_bit

    # the side part holds, in order: list scales, row powers, 8-bit values, histogram
    offset = 0
    list_scales = np.frombuffer(side_part, np.float32, number_of_lists, offset)
    offset += 4 * number_of_lists
    row_powers = np.frombuffer(side_part, np.uint8, number_of_rows, offset)
    offset += number_of_rows
    eight_bit_values = np.frombuffer(side_part, np.int8, number_eight_bit, offset)
    offset += number_eight_bit
    histogram = np.frombuffer(side_part, np.uint32, 2 * pack.WEIGHT_STEPS + 1, offset)

    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(five_bit_bytes, dtype=np.uint32))
    odds = constriction.stream.model.Categorical(pack.histogram_to_odds(histogram), perfect=False)
    symbols = np.asarray(decoder.decode(odds, number_five_bit), np.int64)
    five_bit_values = symbols - pack.WEIGHT_STEPS

    weights = pack.dequantize_weights(row_powers, list_scales, eight_bit_values, five_bit_values, template)
    model.load_state_dict(weights)
    return model.to(DEVICE).eval()


def unpack_maps(archive_folder):
    """Decode the 600 maps from maps.bin."""
    data = (archive_folder / "maps.bin").read_bytes()
    return map_encoder.decode(data, NUMBER_OF_PAIRS)


def unpack_pose(archive_folder):
    """Unpack the motion values of all 600 pairs from pose.bin."""
    data = brotli.decompress((archive_folder / "pose.bin").read_bytes())
    scales = np.frombuffer(data, np.float32, 6, 0)
    whole_numbers = np.frombuffer(data, np.int8, NUMBER_OF_PAIRS * 6, 6 * 4).reshape(NUMBER_OF_PAIRS, 6)
    return whole_numbers.astype(np.float32) * scales


def unpack_correction(archive_folder):
    """Unpack frame 1's pose correction from correction.bin.

    Output:
        patterns: (4, 3, 12, 16) float32, the patterns shared by every pair.
        amounts: (600, 4) float32, how much of each pattern every pair gets.
        strength: float, the correction's strength.
    """
    data = brotli.decompress((archive_folder / "correction.bin").read_bytes())
    number_of_patterns, height, width = struct.unpack_from("<HHH", data, 0)
    offset = 6

    strength, pattern_scale = np.frombuffer(data, np.float32, 2, offset)
    offset += 8
    amount_scales = np.frombuffer(data, np.float32, number_of_patterns, offset)
    offset += 4 * number_of_patterns

    pattern_count = number_of_patterns * 3 * height * width
    patterns = np.frombuffer(data, np.int8, pattern_count, offset).astype(np.float32)
    patterns = patterns.reshape(number_of_patterns, 3, height, width) * pattern_scale
    offset += pattern_count

    amounts = np.frombuffer(data, np.int8, NUMBER_OF_PAIRS * number_of_patterns, offset).astype(np.float32)
    amounts = amounts.reshape(NUMBER_OF_PAIRS, number_of_patterns) * amount_scales

    return torch.from_numpy(patterns), torch.from_numpy(amounts), float(strength)


def draw_video(model, maps, poses, correction, raw_path):
    """Draw all 600 pairs at camera size and write them to raw_path."""
    video = np.memmap(raw_path, np.uint8, "w+", shape=(2 * NUMBER_OF_PAIRS, frame_model.CAMERA_HEIGHT, frame_model.CAMERA_WIDTH, 3))
    patterns, amounts, strength = correction
    patterns = patterns.to(DEVICE)

    with torch.no_grad():
        for batch_start in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            batch_end = batch_start + PAIRS_PER_BATCH
            batch_maps = torch.from_numpy(maps[batch_start:batch_end].astype(np.int64)).to(DEVICE)
            batch_poses = torch.from_numpy(poses[batch_start:batch_end]).to(DEVICE)
            frame1, frame2 = model(batch_maps, batch_poses)

            # the pose correction, each pair's own amount of every pattern
            correction_image = torch.einsum("nk,kchw->nchw", amounts[batch_start:batch_end].to(DEVICE), patterns)
            correction_image = F.interpolate(correction_image, size=frame1.shape[-2:], mode="bilinear", align_corners=False)
            frame1 = frame1 + strength * correction_image

            # (pair, height, width, rgb) for the raw file
            frame1 = frame_model.to_camera(frame1).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            frame2 = frame_model.to_camera(frame2).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            video[2 * batch_start:2 * batch_end:2] = frame1
            video[2 * batch_start + 1:2 * batch_end:2] = frame2

    video.flush()


def main(archive_folder, output_folder, video_names_file):
    """Unpack the archive and write one .raw video per name."""
    start_time = time.time()
    archive_folder = Path(archive_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    maps = unpack_maps(archive_folder)
    model = unpack_model(archive_folder)
    poses = unpack_pose(archive_folder)
    correction = unpack_correction(archive_folder)
    print(f"unpacked in {time.time() - start_time:.0f}s on {DEVICE}", flush=True)

    for video_name in Path(video_names_file).read_text().split():
        raw_path = output_folder / (Path(video_name).stem + ".raw")
        draw_video(model, maps, poses, correction, raw_path)
        print("Wrote", raw_path, f"({time.time() - start_time:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
