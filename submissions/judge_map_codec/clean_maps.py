"""
Clean the maps, removing pixels that cost bytes but don't change what SegNet sees.

1. Try changing expensive pixels to the guesser's own guess.
2. Keep a change only if the bytes it saves are worth more than the SegNet pixels it breaks.
3. 1 wrong SegNet pixel costs as much score as 10.18 bits.
4. Edit every 6th map per round, so edits never change each other's cost.
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
import map_encoder

OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"
DEVICE = train_frame_model.DEVICE
NUMBER_OF_PAIRS = 600
MAP_HEIGHT, MAP_WIDTH = 384, 512
PAIRS_PER_BATCH = 8

ROUNDS = 60
WINDOW = 24                    # how far one edit can reach, in pixels
EDITS_PER_MAP = 80
MIN_BITS = 2.0                 # only try pixels that cost more than this

# 1 wrong SegNet pixel and this many bits of archive cost the same score
BITS_PER_WRONG_PIXEL = 8 * (100 / (600 * 384 * 512)) / (25 / 37_545_489)    # = 10.18


def load_models():
    """Load the shipped model, the map guesser and the judges."""
    model = frame_model.FrameModel()
    model.load_state_dict(pack.shipped_weights(OUTPUT_FOLDER / "stage_pose.pt"))
    model = model.to(DEVICE).eval()

    guesser = map_encoder.load_guesser(OUTPUT_FOLDER / "map_guesser.pt")
    guesser.rounding = True
    guesser = guesser.to(DEVICE).eval()

    for weight in model.parameters():
        weight.requires_grad_(False)
    for weight in guesser.parameters():
        weight.requires_grad_(False)

    judges = train_frame_model.load_judges()
    return model, guesser, judges


def segnet_wrong(model, judges, stored, real, motions):
    """Find the pixels SegNet gets wrong on frame 2.

    Input:
        stored: (maps, 384, 512) uint8, the maps the frame model draws from.
        real: (maps, 384, 512) uint8, SegNet's real maps.
        motions: (maps, 6) float32, the rounded motion values.

    Output:
        wrong: (maps, 384, 512) bool, True where SegNet gets the pixel wrong.
    """
    with torch.no_grad():
        _, frame2 = model(stored.long(), motions)
        camera_frame = frame_model.to_camera(frame2)
        segnet_input = judges.segnet.preprocess_input(camera_frame[:, None])
        labels = judges.segnet(segnet_input).argmax(dim=1)
        return labels != real.long()


def bits_per_pixel(guesser, current, maps_back):
    """Work out the bits spent on each pixel.

    Input:
        current: (maps, 384, 512) uint8, the maps being coded.
        maps_back: 5 x (maps, 384, 512) uint8, their last 5 maps.

    Output:
        bits: (maps, 384, 512) float16, bits per pixel.
    """
    old_maps = []
    for one_back in maps_back:
        old_maps.append(one_back.long())

    with torch.no_grad():
        scores = guesser(current.long(), *old_maps)
        log_odds = F.log_softmax(scores.float(), dim=1)
        real_label_log_odds = log_odds.gather(1, current.long()[:, None])[:, 0]
        return (-real_label_log_odds / math.log(2)).half()


class MapCleaner:
    """Keeps the maps, their wrong SegNet pixels and their bits up to date while editing."""

    def __init__(self):
        targets = np.load(OUTPUT_FOLDER / "targets.npz")
        self.real = torch.from_numpy(targets["maps"]).to(DEVICE)          # SegNet's real maps, never change
        self.stored = self.real.clone()                                   # what gets coded, edited here
        pose_arr = torch.from_numpy(targets["poses"]).float().to(DEVICE)
        self.motions = train_frame_model.quantize_pose(pose_arr)
        self.model, self.guesser, self.judges = load_models()
        self.blank = torch.zeros(MAP_HEIGHT, MAP_WIDTH, dtype=torch.uint8, device=DEVICE)

        self.wrong = torch.zeros(NUMBER_OF_PAIRS, MAP_HEIGHT, MAP_WIDTH, dtype=torch.bool, device=DEVICE)
        self.bits = torch.zeros(NUMBER_OF_PAIRS, MAP_HEIGHT, MAP_WIDTH, dtype=torch.float16, device=DEVICE)
        for first in range(0, NUMBER_OF_PAIRS, PAIRS_PER_BATCH):
            maps = list(range(first, min(NUMBER_OF_PAIRS, first + PAIRS_PER_BATCH)))
            self.wrong[maps] = self.segnet_wrong(maps, self.stored[maps])
            self.bits[maps] = self.bits_for(maps)

    def segnet_wrong(self, maps, stored):
        """Find SegNet's wrong pixels for these maps."""
        return segnet_wrong(self.model, self.judges, stored, self.real[maps], self.motions[maps])

    def history(self, map_number, swap=None):
        """Get the 5 maps before this one, with an optional edited map swapped in."""
        old_maps = []
        for steps_back in range(1, map_encoder.MAPS_BACK + 1):
            earlier_map = map_number - steps_back
            if earlier_map < 0:
                old_maps.append(self.blank)
            elif swap is not None and earlier_map == swap[0]:
                old_maps.append(swap[1])
            else:
                old_maps.append(self.stored[earlier_map])
        return old_maps

    def bits_for(self, maps, swap=None):
        """Work out the bits per pixel for these maps."""
        current = []
        histories = []
        for map_number in maps:
            if swap is not None and map_number == swap[0]:
                current.append(swap[1])
            else:
                current.append(self.stored[map_number])
            histories.append(self.history(map_number, swap))

        maps_back = []
        for steps_back in range(map_encoder.MAPS_BACK):
            one_back = []
            for history in histories:
                one_back.append(history[steps_back])
            maps_back.append(torch.stack(one_back))
        return bits_per_pixel(self.guesser, torch.stack(current), maps_back)

    def maps_affected_by(self, map_number):
        """List this map and the 5 after it."""
        return list(range(map_number, min(NUMBER_OF_PAIRS, map_number + map_encoder.MAPS_BACK + 1)))

    def total_bits(self):
        return self.bits.float().sum().item()

    def total_wrong(self):
        return self.wrong.sum().item()

    def clean_batch(self, maps):
        """Try single-pixel edits on these maps and keep the ones that pay off."""
        before = self.stored[maps].clone()

        # try expensive pixels where the guesser guessed something else
        maps_back = []
        histories = []
        for map_number in maps:
            histories.append(self.history(map_number))
        for steps_back in range(map_encoder.MAPS_BACK):
            one_back = []
            for history in histories:
                one_back.append(history[steps_back].long())
            maps_back.append(torch.stack(one_back))
        with torch.no_grad():
            guess = self.guesser(before.long(), *maps_back).argmax(dim=1)

        cost = self.bits[maps].float()
        is_candidate = (guess != before.long()) & (cost > MIN_BITS)
        never = torch.full_like(cost, float("inf"))
        priority = torch.where(is_candidate, -cost, never)          # most expensive first

        # up to 80 edits per map, at least 2 x WINDOW apart so their effects don't overlap
        edits = []
        remaining = priority.clone()
        for _ in range(EDITS_PER_MAP):
            best, where = remaining.view(len(maps), -1).min(dim=1)
            for index in range(len(maps)):
                if not math.isfinite(best[index].item()):
                    continue
                y, x = divmod(where[index].item(), MAP_WIDTH)
                edits.append((index, y, x, guess[index, y, x].item()))
                top = max(0, y - 2 * WINDOW)
                left = max(0, x - 2 * WINDOW)
                remaining[index, top:y + 2 * WINDOW + 1, left:x + 2 * WINDOW + 1] = float("inf")
        if len(edits) == 0:
            return 0

        edited = before.clone()
        for index, y, x, label in edits:
            edited[index, y, x] = label
        wrong_after = self.segnet_wrong(maps, edited)

        bits_change = torch.zeros(len(maps), MAP_HEIGHT, MAP_WIDTH, device=DEVICE)
        for index, map_number in enumerate(maps):
            affected = self.maps_affected_by(map_number)
            bits_after = self.bits_for(affected, (map_number, edited[index])).float()
            bits_change[index] = (bits_after - self.bits[affected].float()).sum(dim=0)

        # keep an edit only if the bits it saves are worth more than the SegNet pixels it breaks
        result = before.clone()
        changed = [False] * len(maps)
        for index, y, x, label in edits:
            rows = slice(max(0, y - WINDOW), min(MAP_HEIGHT, y + WINDOW + 1))
            columns = slice(max(0, x - WINDOW), min(MAP_WIDTH, x + WINDOW + 1))
            extra_wrong = (wrong_after[index, rows, columns].sum() - self.wrong[maps[index], rows, columns].sum()).item()
            bits_saved = bits_change[index, rows, columns].sum().item()
            if BITS_PER_WRONG_PIXEL * extra_wrong + bits_saved < -0.5:
                result[index, y, x] = label
                changed[index] = True

        kept = 0
        for index, map_number in enumerate(maps):
            if not changed[index]:
                continue
            kept += int((result[index] != before[index]).sum().item())
            affected = self.maps_affected_by(map_number)
            new_bits = self.bits_for(affected, (map_number, result[index]))
            self.stored[map_number] = result[index]
            self.wrong[map_number] = self.segnet_wrong([map_number], result[index:index + 1])[0]
            self.bits[affected] = new_bits
        return kept


def main(rounds):
    """Clean the maps for some rounds and save them to out/clean_maps.npz."""
    start_time = time.time()
    cleaner = MapCleaner()
    wrong_start = cleaner.total_wrong()
    bits_start = cleaner.total_bits()
    print(f"start: SegNet wrong pixels {wrong_start:,}   maps ~{bits_start / 8:,.0f} B   {time.time() - start_time:.0f}s", flush=True)

    for round_number in range(rounds):
        # every 6th map, a different sixth each round
        phase = round_number % 6
        maps = []
        for map_number in range(NUMBER_OF_PAIRS):
            if map_number % 6 == phase:
                maps.append(map_number)

        kept = 0
        for first in range(0, len(maps), PAIRS_PER_BATCH):
            kept += cleaner.clean_batch(maps[first:first + PAIRS_PER_BATCH])

        # score gain so far, from fewer wrong SegNet pixels minus the extra bytes
        seg_gain = 100 * (wrong_start - cleaner.total_wrong()) / (NUMBER_OF_PAIRS * MAP_HEIGHT * MAP_WIDTH)
        size_cost = 25 * (cleaner.total_bits() - bits_start) / 8 / 37_545_489
        print(f"round {round_number:2d}: kept {kept:4d} edits   maps ~{cleaner.total_bits() / 8:,.0f} B   "
              f"score gain {seg_gain - size_cost:+.4f}   {time.time() - start_time:.0f}s", flush=True)

    np.savez_compressed(OUTPUT_FOLDER / "clean_maps.npz", maps=cleaner.stored.cpu().numpy())
    changed = (cleaner.stored != cleaner.real).sum().item()
    print(f"saved out/clean_maps.npz   pixels changed {changed:,}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
    else:
        rounds = ROUNDS
    main(rounds)
