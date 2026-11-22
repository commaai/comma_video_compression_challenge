"""
Code the 600 maps into maps.bin by guessing each pixel and only paying for surprises.

1. Guess each pixel from the last 5 maps and its 30 known neighbours.
2. Code the real label with a range coder, cheap when the guess is right.
3. Go one diagonal at a time, so every neighbour is already decoded.
4. Guess with whole numbers only, so encode and decode match on any computer.
"""

import sys
import time
import struct
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"
GUESSER_FILE = OUTPUT_FOLDER / "map_guesser.bin"

MAP_HEIGHT, MAP_WIDTH = 384, 512
NUMBER_OF_CLASSES = 5          # road, lane, sky, car, hood
OUTSIDE = 5                    # label for a neighbour outside the picture

MAPS_BACK = 5                  # how many old maps the guesser sees
CHANNELS = 16
HIDDEN = 32

WEIGHT_STEPS = 127             # 8-bit weights: -127 to 127
VALUE_STEP = 16.0              # values inside the guesser are rounded to 1/16
SCORE_STEP = 16.0              # scores are rounded to 1/16

# the 30 known neighbours as (up, left) offsets, all on earlier diagonals so they're already decoded
NEIGHBOURS = []
for up in range(5):
    for left in range(-4, 5):
        if up + left >= 1:
            NEIGHBOURS.append((up, left))


def make_diagonals():
    """List the pixel positions on each diagonal, in the order they're coded."""
    rows, columns = np.mgrid[0:MAP_HEIGHT, 0:MAP_WIDTH]
    diagonal_of_each_pixel = (rows + columns).ravel()
    order = np.argsort(diagonal_of_each_pixel, kind="stable")
    starts = np.searchsorted(diagonal_of_each_pixel[order], np.arange(MAP_HEIGHT + MAP_WIDTH))

    diagonals = []
    number_of_diagonals = MAP_HEIGHT + MAP_WIDTH - 1
    for number in range(number_of_diagonals):
        if number + 1 < number_of_diagonals:
            end = starts[number + 1]
        else:
            end = None
        diagonals.append(order[starts[number]:end])
    return diagonals


DIAGONALS = make_diagonals()
ROW_OF = np.arange(MAP_HEIGHT * MAP_WIDTH) // MAP_WIDTH
COLUMN_OF = np.arange(MAP_HEIGHT * MAP_WIDTH) % MAP_WIDTH


def round_weights(weight):
    """Train on 8-bit weights, but let the gradient reach the real weight."""
    scale = weight.detach().abs().max().clamp_min(1e-8) / WEIGHT_STEPS
    rounded = (weight / scale).round().clamp(-WEIGHT_STEPS, WEIGHT_STEPS) * scale
    return weight + (rounded - weight).detach()


def round_values(values):
    """Train on values rounded to 1/16, but let the gradient through."""
    rounded = (values * VALUE_STEP).round() / VALUE_STEP
    return values + (rounded - values).detach()


def one_hot(maps):
    """Turn (maps, 384, 512) labels into (maps, 5, 384, 512) yes/no grids, one per class."""
    return F.one_hot(maps, NUMBER_OF_CLASSES).permute(0, 3, 1, 2).float()


def position_grids(device):
    """Make row and column grids so the guesser knows where each pixel is."""
    rows = torch.linspace(-1, 1, MAP_HEIGHT, device=device).view(1, 1, MAP_HEIGHT, 1).expand(1, 1, MAP_HEIGHT, MAP_WIDTH)
    columns = torch.linspace(-1, 1, MAP_WIDTH, device=device).view(1, 1, 1, MAP_WIDTH).expand(1, 1, MAP_HEIGHT, MAP_WIDTH)
    return torch.cat([rows, columns], dim=1)


def shift(maps, up, left):
    """Shift the maps so each pixel sees its neighbour at (up, left)."""
    shifted = torch.zeros_like(maps)

    if up >= 0:
        to_rows = slice(up, None)
        from_rows = slice(None, MAP_HEIGHT - up)
    else:
        to_rows = slice(None, MAP_HEIGHT + up)
        from_rows = slice(-up, None)

    if left >= 0:
        to_columns = slice(left, None)
        from_columns = slice(None, MAP_WIDTH - left)
    else:
        to_columns = slice(None, MAP_WIDTH + left)
        from_columns = slice(-left, None)

    shifted[..., to_rows, to_columns] = maps[..., from_rows, from_columns]
    return shifted


class MapGuesser(nn.Module):
    """Guesses every pixel's label from the last 5 maps and its 30 known neighbours."""

    def __init__(self):
        super().__init__()
        map_inputs = NUMBER_OF_CLASSES * MAPS_BACK + 2
        pixel_inputs = CHANNELS + NUMBER_OF_CLASSES * MAPS_BACK + NUMBER_OF_CLASSES * len(NEIGHBOURS)

        # shrink, then 3 layers that see 1, 2 and 4 pixels apart
        self.shrink = nn.Conv2d(map_inputs, CHANNELS, kernel_size=3, stride=2, padding=1)
        self.near = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.wider = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, stride=1, padding=2, dilation=2)
        self.widest = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, stride=1, padding=4, dilation=4)

        self.pixel_hidden = nn.Conv2d(pixel_inputs, HIDDEN, kernel_size=1)
        self.pixel_scores = nn.Conv2d(HIDDEN, NUMBER_OF_CLASSES, kernel_size=1)

        # on at the end of training so the guesser learns the whole-number math
        self.rounding = False

    def run_layer(self, conv, x):
        """Run one conv, rounded like the whole-number guesser when rounding is on."""
        if not self.rounding:
            return conv(x)
        out = F.conv2d(x, round_weights(conv.weight), conv.bias, conv.stride, conv.padding, conv.dilation)
        return round_values(out)

    def forward(self, current, *maps_back):
        """Score every pixel of the current map.

        Input:
            current: (maps, 384, 512) long, the map being guessed.
            maps_back: 5 x (maps, 384, 512) long, the last 5 maps, newest first.

        Output:
            scores: (maps, 5, 384, 512) float32, 5 class scores per pixel.
        """
        number_of_maps = current.shape[0]

        old_maps = []
        for one_map in maps_back:
            old_maps.append(one_hot(one_map))
        positions = position_grids(current.device).expand(number_of_maps, -1, -1, -1)
        x = torch.cat(old_maps + [positions], dim=1)
        if self.rounding:
            x = round_values(x)

        features = F.relu(self.run_layer(self.shrink, x))
        features = features + F.relu(self.run_layer(self.near, features))
        features = features + F.relu(self.run_layer(self.wider, features))
        features = features + F.relu(self.run_layer(self.widest, features))
        features = F.interpolate(features, scale_factor=2, mode="nearest")

        current_grids = one_hot(current)
        neighbours = []
        for up, left in NEIGHBOURS:
            neighbours.append(shift(current_grids, up, left))
        pixel_inputs = torch.cat([features] + old_maps + neighbours, dim=1)

        hidden = F.relu(self.run_layer(self.pixel_hidden, pixel_inputs))
        return self.run_layer(self.pixel_scores, hidden)


def load_guesser(path, device="cpu"):
    """Load a trained MapGuesser from a .pt file."""
    state = torch.load(path, map_location=device, weights_only=True)
    guesser = MapGuesser()
    guesser.load_state_dict(state)
    return guesser.to(device)


# the guesser's layers in packing order, with (stride, padding, dilation)
LAYER_NAMES = ["shrink", "near", "wider", "widest", "pixel_hidden", "pixel_scores"]
LAYER_SETTINGS = {
    "shrink": (2, 1, 1),
    "near": (1, 1, 1),
    "wider": (1, 2, 2),
    "widest": (1, 4, 4),
    "pixel_hidden": (1, 0, 1),
    "pixel_scores": (1, 0, 1),
}


def export_layers(guesser):
    """Round each layer's weights to 8-bit."""
    layers = []
    for name in LAYER_NAMES:
        conv = getattr(guesser, name)
        weight = conv.weight.detach().double().cpu()
        scale = np.float32(max(weight.abs().max().item(), 1e-8) / WEIGHT_STEPS)
        whole_numbers = (weight / float(scale)).round().clamp(-WEIGHT_STEPS, WEIGHT_STEPS).to(torch.int8).numpy()
        bias = conv.bias.detach().float().cpu().numpy().astype(np.float64)
        layers.append((whole_numbers, bias, scale))
    return layers


def pack_layers(layers):
    """Pack the rounded layers into bytes."""
    all_weights = []
    all_biases = []
    all_scales = []
    for whole_numbers, bias, scale in layers:
        all_weights.append(whole_numbers.ravel())
        all_biases.append(bias.ravel())
        all_scales.append(scale)

    weights = np.concatenate(all_weights).astype(np.int8)
    biases = np.concatenate(all_biases).astype(np.float32)
    scales = np.array(all_scales, np.float32)

    # brotli of [all 8-bit weights] + [all biases] + [6 scales]
    return brotli.compress(weights.tobytes() + biases.tobytes() + scales.tobytes(), quality=11)


def unpack_layers(packed):
    """Undo pack_layers, taking the shapes from an empty MapGuesser."""
    data = brotli.decompress(packed)
    empty = MapGuesser()

    shapes = []
    number_of_weights = 0
    number_of_biases = 0
    for name in LAYER_NAMES:
        conv = getattr(empty, name)
        shapes.append((tuple(conv.weight.shape), conv.bias.numel()))
        number_of_weights += conv.weight.numel()
        number_of_biases += conv.bias.numel()

    weights = np.frombuffer(data, np.int8, number_of_weights, 0)
    biases = np.frombuffer(data, np.float32, number_of_biases, number_of_weights)
    scales = np.frombuffer(data, np.float32, len(LAYER_NAMES), number_of_weights + 4 * number_of_biases)

    layers = []
    weight_offset = 0
    bias_offset = 0
    for (shape, bias_count), scale in zip(shapes, scales):
        size = int(np.prod(shape))
        whole_numbers = weights[weight_offset:weight_offset + size].reshape(shape).copy()
        bias = biases[bias_offset:bias_offset + bias_count].astype(np.float64)
        layers.append((whole_numbers, bias, np.float32(scale)))
        weight_offset += size
        bias_offset += bias_count
    return layers


def bias_in_steps(bias, scale):
    """Convert a layer's bias into whole-number units."""
    return np.floor(bias * VALUE_STEP / float(scale) + 0.5)


class WholeNumberGuesser:
    """Runs the map guesser in whole numbers only, so encode and decode guess the same on any computer."""

    def __init__(self, layers):
        self.layers = {}
        for name, (whole_numbers, bias, scale) in zip(LAYER_NAMES, layers):
            weights = torch.from_numpy(whole_numbers.astype(np.float64))
            bias = torch.from_numpy(bias_in_steps(bias, scale))
            self.layers[name] = (weights, bias, float(np.float64(scale)))

        hidden_weights, hidden_bias, self.hidden_scale = self.layers["pixel_hidden"]
        hidden_whole = hidden_weights[:, :, 0, 0].numpy().astype(np.int64)
        old_maps_end = CHANNELS + NUMBER_OF_CLASSES * MAPS_BACK

        # the old-maps part of the per-pixel layer, done once per map
        self.hidden_from_map = hidden_weights[:, :old_maps_end]

        # the neighbours part: one lookup table per neighbour
        self.from_neighbour = []
        for number in range(len(NEIGHBOURS)):
            first_column = old_maps_end + NUMBER_OF_CLASSES * number
            columns = hidden_whole[:, first_column:first_column + NUMBER_OF_CLASSES].T * int(VALUE_STEP)
            outside_row = np.zeros((1, columns.shape[1]), np.int64)
            self.from_neighbour.append(np.vstack([columns, outside_row]))

        score_weights, score_bias, self.score_scale = self.layers["pixel_scores"]
        self.score_weights = score_weights[:, :, 0, 0].numpy().astype(np.int64).T
        self.score_bias = score_bias.numpy().astype(np.int64)

        position = position_grids("cpu").double().numpy()
        self.position = torch.from_numpy(np.floor(position * VALUE_STEP + 0.5))

    def run_layer(self, name, x):
        """Run one layer in whole numbers."""
        weights, bias, scale = self.layers[name]
        stride, padding, dilation = LAYER_SETTINGS[name]
        total = F.conv2d(x, weights, bias, stride, padding, dilation)
        return torch.floor(total * scale + 0.5)

    def old_maps_part(self, *maps_back):
        """Compute everything that only needs the old maps, once per map.

        Input:
            maps_back: 5 x (384, 512) int64, the last 5 maps, newest first.

        Output:
            partial_hidden: (384*512, 32) int64, the per-pixel layer before the neighbours are added.
        """
        old_maps = []
        for one_map in maps_back:
            labels = torch.from_numpy(one_map).view(1, MAP_HEIGHT, MAP_WIDTH).long()
            grids = F.one_hot(labels, NUMBER_OF_CLASSES).permute(0, 3, 1, 2).double() * VALUE_STEP
            old_maps.append(grids)
        x = torch.cat(old_maps + [self.position], dim=1)

        features = torch.relu(self.run_layer("shrink", x))
        for name in ("near", "wider", "widest"):
            features = features + torch.relu(self.run_layer(name, features))
        features = features.repeat_interleave(2, 2).repeat_interleave(2, 3)

        pixel_inputs = torch.cat([features] + old_maps, dim=1)
        total = F.conv2d(pixel_inputs, self.hidden_from_map, self.layers["pixel_hidden"][1])
        return total[0].reshape(total.shape[1], -1).T.numpy().astype(np.int64)

    def guess(self, partial_hidden, neighbour_labels):
        """Work out the odds for every pixel on one diagonal.

        Input:
            partial_hidden: (pixels, 32) int64, old_maps_part at these pixels.
            neighbour_labels: 30 x (pixels,) labels 0-5, with 5 for outside the picture.

        Output:
            odds: (pixels, 5) float64, each row sums to 1.
        """
        total = partial_hidden.copy()
        for lookup, labels in zip(self.from_neighbour, neighbour_labels):
            total += lookup[labels]
        hidden = np.maximum(np.floor(total * self.hidden_scale + 0.5), 0).astype(np.int64)

        scores = hidden @ self.score_weights + self.score_bias
        scores = np.floor(scores * (self.score_scale * SCORE_STEP / VALUE_STEP) + 0.5).astype(np.int64)

        # turn scores into odds with the fixed table
        below_best = np.minimum(scores.max(axis=1, keepdims=True) - scores, len(EXP_TABLE) - 1)
        exp_values = EXP_TABLE[below_best]
        return exp_values / exp_values.sum(axis=1, keepdims=True)


# exp() as a fixed table of whole numbers, so every computer gets the same odds
EXP_TABLE = np.array([
    16777216, 15760736, 14805841, 13908801, 13066109, 12274473, 11530801, 10832185, 10175896, 9559370, 8980197, 8436114,
    7924996, 7444844, 6993784, 6570052, 6171993, 5798051, 5446765, 5116762, 4806753, 4515526, 4241945, 3984938,
    3743503, 3516696, 3303630, 3103473, 2915443, 2738805, 2572869, 2416987, 2270549, 2132984, 2003753, 1882351,
    1768306, 1661169, 1560524, 1465977, 1377158, 1293720, 1215337, 1141704, 1072532, 1007550, 946506, 889160,
    835288, 784681, 737139, 692478, 650523, 611110, 574085, 539303, 506628, 475933, 447098, 420009,
    394562, 370657, 348200, 327104, 307285, 288668, 271178, 254749, 239314, 224815, 211194, 198398,
    186378, 175086, 164478, 154513, 145151, 136357, 128096, 120335, 113044, 106195, 99761, 93717,
    88039, 82705, 77694, 72987, 68565, 64411, 60508, 56842, 53398, 50163, 47124, 44269,
    41587, 39067, 36700, 34476, 32388, 30425, 28582, 26850, 25224, 23695, 22260, 20911,
    19644, 18454, 17336, 16286, 15299, 14372, 13501, 12683, 11915, 11193, 10515, 9878,
    9279, 8717, 8189, 7693, 7227, 6789, 6378, 5991, 5628, 5287, 4967, 4666,
    4383, 4118, 3868, 3634, 3414, 3207, 3013, 2830, 2659, 2497, 2346, 2204,
    2070, 1945, 1827, 1716, 1612, 1515, 1423, 1337, 1256, 1180, 1108, 1041,
    978, 919, 863, 811, 762, 716, 672, 631, 593, 557, 523, 492,
    462, 434, 408, 383, 360, 338, 318, 298, 280, 263, 247, 232,
    218, 205, 193, 181, 170, 160, 150, 141, 132, 124, 117, 110,
    103, 97, 91, 85, 80, 75, 71, 67, 63, 59, 55, 52,
    49, 46, 43, 40, 38, 36, 33, 31, 30, 28, 26, 24,
    23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12,
    11, 10, 10, 9, 8, 8, 7, 7, 7, 6, 6, 5,
    5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1], np.int64)


def known_neighbours(current, positions):
    """Look up the 30 neighbours' labels for the pixels on one diagonal."""
    rows = ROW_OF[positions]
    columns = COLUMN_OF[positions]

    labels = []
    for up, left in NEIGHBOURS:
        inside = (rows - up >= 0) & (columns - left >= 0) & (columns - left < MAP_WIDTH)
        neighbour = np.clip(positions - up * MAP_WIDTH - left, 0, MAP_HEIGHT * MAP_WIDTH - 1)
        labels.append(np.where(inside, current[neighbour], OUTSIDE))
    return labels


def code_maps(guesser, number_of_maps, maps=None, encoder=None, decoder=None):
    """Run the loop shared by encode and decode."""
    odds_model = constriction.stream.model.Categorical(perfect=False)
    decoded = np.zeros((number_of_maps, MAP_HEIGHT * MAP_WIDTH), np.uint8)

    # before map 0 there are no old maps, so use all label 0
    maps_back = []
    for _ in range(MAPS_BACK):
        maps_back.append(np.zeros(MAP_HEIGHT * MAP_WIDTH, np.int64))

    for map_number in range(number_of_maps):
        old_maps = []
        for one_map in maps_back:
            old_maps.append(one_map.reshape(MAP_HEIGHT, MAP_WIDTH))
        partial_hidden = guesser.old_maps_part(*old_maps)

        if maps is not None:
            current = maps[map_number].reshape(-1).astype(np.int64)
        else:
            current = np.zeros(MAP_HEIGHT * MAP_WIDTH, np.int64)       # filled in diagonal by diagonal

        for positions in DIAGONALS:
            odds = guesser.guess(partial_hidden[positions], known_neighbours(current, positions))
            if encoder is not None:
                encoder.encode(current[positions].astype(np.int32), odds_model, odds)
            else:
                current[positions] = np.asarray(decoder.decode(odds_model, odds), np.int64)

        decoded[map_number] = current
        maps_back = [current] + maps_back[:-1]                         # newest first
    return decoded


def encode(maps):
    """Code the maps into maps.bin.

    Input:
        maps: (600, 384, 512) uint8, a label 0-4 for every pixel.

    Output:
        data: bytes, the 8-bit map guesser followed by the coded maps.
    """
    guesser_bytes = GUESSER_FILE.read_bytes()
    guesser = WholeNumberGuesser(unpack_layers(guesser_bytes))
    encoder = constriction.stream.queue.RangeEncoder()
    code_maps(guesser, len(maps), maps=maps, encoder=encoder)
    return struct.pack("<I", len(guesser_bytes)) + guesser_bytes + encoder.get_compressed().tobytes()


def decode(data, number_of_maps):
    """Decode maps.bin back into the exact maps.

    Input:
        data: bytes from encode.
        number_of_maps: 600.

    Output:
        maps: (600, 384, 512) uint8, a label 0-4 for every pixel.
    """
    network_size = struct.unpack("<I", data[:4])[0]
    guesser = WholeNumberGuesser(unpack_layers(data[4:4 + network_size]))
    coded = np.frombuffer(data[4 + network_size:], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(coded)
    decoded = code_maps(guesser, number_of_maps, decoder=decoder)
    return decoded.reshape(number_of_maps, MAP_HEIGHT, MAP_WIDTH)


if __name__ == "__main__":
    # test the exact round trip, size and speed
    maps = np.load(OUTPUT_FOLDER / "clean_maps.npz")["maps"]
    if len(sys.argv) > 1:
        maps = maps[:int(sys.argv[1])]

    start_time = time.time()
    data = encode(maps)
    encode_seconds = time.time() - start_time

    start_time = time.time()
    decoded = decode(data, len(maps))
    decode_seconds = time.time() - start_time

    print(f"maps: {len(maps)}   size: {len(data):,} bytes")
    print(f"encode: {encode_seconds:.0f}s   decode: {decode_seconds:.0f}s")
    print("exact round trip:", np.array_equal(decoded, maps))
