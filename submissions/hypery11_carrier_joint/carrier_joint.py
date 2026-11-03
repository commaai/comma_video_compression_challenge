"""Lossless carrier recoding with joint color prediction/adaptation search.

Inherits RR5 layout and arithmetic coding from size_matters. Unlike its fixed
global adaptation rate, this stores an independently selected rate per plane.
Color weights are jointly selected for that rate and packed by column ranges.
All selected parameters occupy archive bytes; the original carrier is restored.
"""
import json
import math
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import carrier_pack as original
import packing
from renderer_pack import Reader, Writer

INCREMENTS = (2, 3, 4, 5, 6, 8, 12, 16)
HEADER = struct.Struct("<4sHBB")
N = 24 * 32
# Carrier models do not rescale their counts: this is the exact ideal log cost
# of the same adaptive counts, computed from their final symbol histograms.
PARTIAL = np.asarray([[math.lgamma(1 / k + n) - math.lgamma(1 / k) for n in range(N + 1)]
                      for k in INCREMENTS])
TOTAL = np.asarray([math.lgamma(32 / k + N) - math.lgamma(32 / k) for k in INCREMENTS])


def symbols(atom, plane, weights, fraction):
    if weights:
        x = np.asarray(weights) @ atom[:plane]
        half = 1 << (fraction - 1)
        prediction = np.where(x >= 0, (x + half) >> fraction, -((-x + half) >> fraction))
    else:
        prediction = 0
    residual = ((atom[plane] - prediction + 16) & 31) - 16
    return np.where(residual >= 0, residual * 2, -residual * 2 - 1)


def choose(atom, plane, fraction, prior=None):
    if plane == 0:
        candidates = [()]
    else:
        center = original.fitted_weights(atom.tolist(), plane, fraction)
        radius = 16 if plane == 1 else 8
        bounds = [range(max(-128, v - radius), min(127, v + radius) + 1) for v in center]
        if plane == 1:
            candidates = [(x,) for x in bounds[0]]
        else:
            candidates = [(x, y) for x in bounds[0] for y in bounds[1]]
    if prior is None or max(prior) == 1:
        partial = None
        total_cost = TOTAL
    else:
        partial = np.asarray([[[math.lgamma(c / k + n) - math.lgamma(c / k)
                                for n in range(N + 1)] for c in prior] for k in INCREMENTS])
        total_cost = np.asarray([math.lgamma(sum(prior) / k + N) - math.lgamma(sum(prior) / k)
                                 for k in INCREMENTS])
    best = None
    for weights in candidates:
        values = symbols(atom, plane, weights, fraction)
        histogram = np.bincount(values, minlength=32)
        if partial is None:
            costs = total_cost - PARTIAL[:, histogram].sum(axis=1)
        else:
            costs = total_cost - partial[:, np.arange(32), histogram].sum(axis=1)
        index = int(costs.argmin())
        rank = (float(costs[index]), weights, index)
        if best is None or rank < best[0]:
            best = rank, values
    rank, values = best
    return rank[1], rank[2], values


def pack_fields(fields):
    value, used, output = 0, 0, bytearray()
    for number, width in fields:
        if not 0 <= number < (1 << width):
            raise ValueError("Field exceeds its declared width")
        value |= int(number) << used
        used += width
        while used >= 8:
            output.append(value & 255)
            value >>= 8
            used -= 8
    if used:
        output.append(value)
    return bytes(output)


def unpack_fields(data, widths):
    value = int.from_bytes(data, "little")
    output = []
    for width in widths:
        output.append(value & ((1 << width) - 1))
        value >>= width
    if value:
        raise ValueError("Nonzero padding")
    return output


def prior_counts(seen, prior_bits):
    if not prior_bits or not sum(seen):
        return [1] * 32
    total = sum(seen)
    return [1 + (n * (1 << prior_bits)) // total for n in seen]


def encode(body, fraction=6, prior_bits=0):
    atoms, old_size = original.read_original(body, 12, 3, 24, 32, 5)
    weights, descriptors, writer = [], [], Writer()
    history = [[0] * 32 for _ in range(3)]
    for atom in atoms:
        array = np.asarray(atom, dtype=np.int64)
        for plane in range(3):
            prior = prior_counts(history[plane], prior_bits)
            selected, descriptor, values = choose(array, plane, fraction, prior)
            weights.extend(selected)
            descriptors.append(descriptor)
            counts = prior[:]
            for value in values.tolist():
                cumulative = original.cumulative(counts)
                writer.put(cumulative[value], cumulative[value + 1], cumulative[-1])
                counts[value] += INCREMENTS[descriptor]
                history[plane][value] += 1
    columns = [weights[c::3] for c in range(3)]
    minimum = [min(x) for x in columns]
    widths = [(max(x) - low).bit_length() for x, low in zip(columns, minimum)]
    metadata = struct.pack("<3bH", *minimum, widths[0] | (widths[1] << 4) | (widths[2] << 8))
    weight_data = pack_fields([(w - minimum[i % 3], widths[i % 3]) for i, w in enumerate(weights)])
    descriptors_data = pack_fields([(x, 3) for x in descriptors])
    stream = metadata + weight_data + descriptors_data + writer.finish()
    result = (HEADER.pack(b"CGA2" if prior_bits else b"CGA1", len(stream), fraction, prior_bits) + body[:original.PREFIX]
              + stream + body[original.PREFIX + old_size:])
    if decode(result) != body:
        raise ValueError("Joint carrier did not round trip")
    return result


def decode(data):
    magic, size, fraction, prior_bits = HEADER.unpack_from(data)
    if (magic not in (b"CGA1", b"CGA2") or (magic == b"CGA1" and prior_bits)
            or prior_bits > 12 or not 1 <= fraction <= 7):
        raise ValueError("Invalid joint carrier header")
    start = HEADER.size + original.PREFIX
    if size < 20 or start + size >= len(data):
        raise ValueError("Truncated joint carrier")
    prefix = data[HEADER.size:start]
    *minimum, mask = struct.unpack_from("<3bH", data, start)
    widths = [(mask >> (4 * c)) & 15 for c in range(3)]
    if mask >> 12 or max(widths) > 8:
        raise ValueError("Invalid weight ranges")
    nw = (12 * sum(widths) + 7) // 8
    position = start + 5
    weights = [n + minimum[i % 3] for i, n in enumerate(
        unpack_fields(data[position:position + nw], widths * 12))]
    position += nw
    descriptors = unpack_fields(data[position:position + 14], [3] * 36)
    position += 14
    reader = Reader(data[position:start + size])
    values = []
    history = [[0] * 32 for _ in range(3)]
    for a in range(12):
        atom = []
        for plane in range(3):
            selected = [] if plane == 0 else weights[a * 3:a * 3 + 1] if plane == 1 else weights[a * 3 + 1:a * 3 + 3]
            counts, row = prior_counts(history[plane], prior_bits), []
            for i in range(N):
                symbol = reader.get(original.cumulative(counts))
                counts[symbol] += INCREMENTS[descriptors[a * 3 + plane]]
                history[plane][symbol] += 1
                prediction = original.rounded(sum(w * atom[p][i] for p, w in enumerate(selected)), fraction) if selected else 0
                row.append(((original.unzigzag(symbol) + prediction + 16) & 31) - 16)
            atom.append(row)
        values.append(atom)
    basis = original.write_original(values, 5)
    if len(basis) != (int.from_bytes(prefix[:3], "little") + 7) // 8:
        raise ValueError("Restored carrier length mismatch")
    return prefix + basis + data[start + size:]
