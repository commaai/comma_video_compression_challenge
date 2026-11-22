"""
Train the map guesser.

1. Guess each pixel from the last 5 maps and its 30 known neighbours.
2. Use the size in bits as the loss, since that's what the range coder spends.
3. Train normally, then with whole-number rounding on, like at coding time.
4. Save the guesser to out/map_guesser.pt and its packed copy to out/map_guesser.bin.
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import map_encoder

OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"
GUESSER_FILE = OUTPUT_FOLDER / "map_guesser.pt"
DEVICE = "cuda"
MAPS_PER_BATCH = 8

# options from the command line, like epochs=200
options = {}
for argument in sys.argv[1:]:
    key, value = argument.split("=", 1)
    options[key] = value

EPOCHS = int(options.get("epochs", 600))
ROUNDING_EPOCHS = int(options.get("rounding_epochs", 100))
LEARNING_RATE = float(options.get("lr", 3e-3))
MAPS_FILE = options.get("maps", "targets.npz")         # which maps to learn, in out/
CONTINUE = options.get("continue", "0") == "1"         # 1 means start from the saved guesser


def load_maps():
    """Load the maps, plus the 5 old maps for each one.

    Output:
        maps: (600, 384, 512) int64, a label 0-4 for every pixel.
        maps_back: 5 x (600, 384, 512) int64, the 5 maps before each map, newest first, all label 0 before map 0.
    """
    maps = np.load(OUTPUT_FOLDER / MAPS_FILE)["maps"]
    maps = torch.from_numpy(maps.astype(np.int64))
    blank = torch.zeros(1, map_encoder.MAP_HEIGHT, map_encoder.MAP_WIDTH, dtype=torch.int64)

    maps_back = []
    for steps_back in range(1, map_encoder.MAPS_BACK + 1):
        maps_back.append(torch.cat([blank] * steps_back + [maps[:-steps_back]]))
    return maps, maps_back


def train(guesser, maps, maps_back, epochs, learning_rate, label):
    """Train the guesser so its guesses cost fewer bits."""
    number_of_maps = len(maps)
    steps_per_epoch = math.ceil(number_of_maps / MAPS_PER_BATCH)
    optimizer = torch.optim.AdamW(guesser.parameters(), learning_rate, weight_decay=0)

    if label == "rounding":
        warm_up_share = 0.1
    else:
        warm_up_share = 0.3
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, total_steps=epochs * steps_per_epoch, pct_start=warm_up_share)

    for epoch in range(epochs):
        order = torch.randperm(number_of_maps)
        total_bits = 0.0
        start_time = time.time()

        for batch_start in range(0, number_of_maps, MAPS_PER_BATCH):
            batch = order[batch_start:batch_start + MAPS_PER_BATCH]
            current = maps[batch].to(DEVICE)
            old_maps = []
            for one_back in maps_back:
                old_maps.append(one_back[batch].to(DEVICE))

            # the cost of the real labels is the bits the coder would spend
            scores = guesser(current, *old_maps)
            cost = F.cross_entropy(scores, current, reduction="sum")
            pixels = len(batch) * map_encoder.MAP_HEIGHT * map_encoder.MAP_WIDTH

            optimizer.zero_grad()
            (cost / pixels).backward()
            torch.nn.utils.clip_grad_norm_(guesser.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total_bits += cost.item() / math.log(2)

        if epoch % 20 == 0 or epoch == epochs - 1:
            seconds = time.time() - start_time
            print(f"{label} epoch {epoch}: maps would be {total_bits / 8:,.0f} bytes  ({seconds:.1f}s)", flush=True)


def main():
    """Train the guesser and save both copies."""
    maps, maps_back = load_maps()

    if CONTINUE:
        guesser = map_encoder.load_guesser(GUESSER_FILE, DEVICE)
    else:
        guesser = map_encoder.MapGuesser().to(DEVICE)

    number_of_weights = 0
    for weight in guesser.parameters():
        number_of_weights += weight.numel()
    print("guesser weights:", number_of_weights, flush=True)

    if EPOCHS > 0:
        train(guesser, maps, maps_back, EPOCHS, LEARNING_RATE, "normal")
    guesser.rounding = True
    train(guesser, maps, maps_back, ROUNDING_EPOCHS, LEARNING_RATE / 10, "rounding")

    torch.save(guesser.state_dict(), GUESSER_FILE)
    guesser_bytes = map_encoder.pack_layers(map_encoder.export_layers(guesser.cpu()))
    map_encoder.GUESSER_FILE.write_bytes(guesser_bytes)
    print("saved out/map_guesser.bin:", len(guesser_bytes), "bytes", flush=True)


if __name__ == "__main__":
    main()
