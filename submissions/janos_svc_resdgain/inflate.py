#!/usr/bin/env python
import io
import os
import struct
import sys
from pathlib import Path

import brotli
import bz2
import einops
import lzma
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from submissions.quantizr.inflate import FP4Codebook, JointFrameGenerator, QConv2d, QEmbedding, get_decoded_state_dict, unpack_nibbles

def _read_exact(buf: memoryview, pos: int, size: int) -> tuple[bytes, int]:
    return bytes(buf[pos : pos + size]), pos + size

def get_decoded_state_dict_qf4a(payload_data, device: torch.device):
    buf = memoryview(payload_data)
    if bytes(buf[:4]) != b"QF4A":
        raise ValueError("bad QF4A payload")
    pos = 4
    block_size = int.from_bytes(buf[pos : pos + 2], "little")
    pos += 2
    module_count = int.from_bytes(buf[pos : pos + 2], "little")
    pos += 2
    model = JointFrameGenerator()
    q_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, (QConv2d, QEmbedding))]
    if module_count != len(q_modules):
        raise ValueError("QF4A module count mismatch")
    state_dict = {}
    covered = set()
    for name, module in q_modules:
        kind = int(buf[pos])
        pos += 1
        weight_numel = module.weight.numel()
        weight_shape = tuple(module.weight.shape)
        covered.add(f"{name}.weight")
        if kind == 1:
            scale_count = (weight_numel + block_size - 1) // block_size
            packed_count = (scale_count * block_size + 1) // 2
            packed_bytes, pos = _read_exact(buf, pos, packed_count)
            scale_bytes, pos = _read_exact(buf, pos, scale_count * 2)
            packed = torch.from_numpy(np.frombuffer(packed_bytes, dtype=np.uint8).copy()).to(device)
            scales = torch.from_numpy(np.frombuffer(scale_bytes, dtype=np.float16).copy()).to(device)
            nibbles = unpack_nibbles(packed, packed_count * 2)
            weight = FP4Codebook.dequantize_from_nibbles(nibbles, scales, weight_shape)
        else:
            weight_bytes, pos = _read_exact(buf, pos, weight_numel * 2)
            weight = torch.from_numpy(np.frombuffer(weight_bytes, dtype=np.float16).copy()).reshape(weight_shape).to(device).float()
        state_dict[f"{name}.weight"] = weight.float()
        if isinstance(module, QConv2d) and module.bias is not None:
            bias_numel = module.bias.numel()
            bias_bytes, pos = _read_exact(buf, pos, bias_numel * 2)
            bias = torch.from_numpy(np.frombuffer(bias_bytes, dtype=np.float16).copy()).reshape(tuple(module.bias.shape)).to(device).float()
            state_dict[f"{name}.bias"] = bias
            covered.add(f"{name}.bias")
    dense_count = int.from_bytes(buf[pos : pos + 2], "little")
    pos += 2
    dense_keys = [key for key in model.state_dict().keys() if key not in covered]
    if dense_count != len(dense_keys):
        raise ValueError("QF4A dense count mismatch")
    for key in dense_keys:
        ref = model.state_dict()[key]
        numel = ref.numel()
        raw, pos = _read_exact(buf, pos, numel * 2)
        value = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).reshape(tuple(ref.shape)).to(device).float()
        state_dict[key] = value
    if pos != len(buf):
        raise ValueError("trailing QF4A payload")
    return state_dict



H, W = 384, 512
OUT_H, OUT_W = 874, 1164
NUM_FRAMES = 600


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    out = 0
    while True:
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if b < 128:
            return out, pos
        shift += 7


def unzig(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


def decompress_section(codec: int, data: bytes) -> bytes:
    if codec == 0:
        return data
    if codec == 1:
        return brotli.decompress(data)
    if codec == 2:
        return bz2.decompress(data)
    if codec == 3:
        return lzma.decompress(data)
    raise ValueError(f"unknown mask section codec: {codec}")


def read_sections(path: Path) -> dict[bytes, bytes]:
    data = path.read_bytes()
    if data[:4] != b"SVC1":
        raise ValueError("bad semantic vector codec payload")
    count = struct.unpack_from("<H", data, 4)[0]
    pos = 6
    sections = {}
    for _ in range(count):
        name = data[pos : pos + 4]
        codec = data[pos + 4]
        size = struct.unpack_from("<I", data, pos + 5)[0]
        pos += 9
        sections[name] = decompress_section(codec, data[pos : pos + size])
        pos += size
    return sections


def decode_boundary(buf: bytes) -> np.ndarray:
    if buf.startswith((b"B2", b"B3")):
        compact_sizes = buf[:2] == b"B3"
        xd = np.zeros(NUM_FRAMES * W, dtype=np.int16)
        pos = 2
        stream_count = buf[pos]
        pos += 1
        for _ in range(stream_count):
            value = struct.unpack_from("<h", buf, pos)[0]
            if compact_sizes:
                size = struct.unpack_from("<H", buf, pos + 2)[0]
                pos += 4
            else:
                size = struct.unpack_from("<I", buf, pos + 2)[0]
                pos += 6
            stream = brotli.decompress(buf[pos : pos + size])
            pos += size
            spos = 0
            cursor = 0
            while spos < len(stream):
                delta, spos = read_varint(stream, spos)
                cursor += delta
                xd[cursor] = value
        if compact_sizes:
            size = struct.unpack_from("<H", buf, pos)[0]
            pos += 2
        else:
            size = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        stream = brotli.decompress(buf[pos : pos + size])
        spos = 0
        cursor = 0
        while spos < len(stream):
            delta, spos = read_varint(stream, spos)
            cursor += delta
            value, spos = read_varint(stream, spos)
            xd[cursor] = unzig(value)

        deltas = xd.reshape(NUM_FRAMES, W)
        out = np.empty((NUM_FRAMES, W), dtype=np.int16)
        for t in range(NUM_FRAMES):
            prev = H
            for x in range(W):
                prev += int(deltas[t, x])
                out[t, x] = prev
        return out

    out = np.empty((NUM_FRAMES, W), dtype=np.int16)
    pos = 0
    for t in range(NUM_FRAMES):
        prev = H
        for x in range(W):
            zz, pos = read_varint(buf, pos)
            prev += unzig(zz)
            out[t, x] = prev
    return out


def draw_lane_segments(mask: np.ndarray, t: int, buf: bytes, pos: int) -> int:
    seg_count, pos = read_varint(buf, pos)
    for _ in range(seg_count):
        point_count, pos = read_varint(buf, pos)
        points = []
        py = ps = pe = 0
        for _ in range(point_count):
            dy, pos = read_varint(buf, pos)
            ds, pos = read_varint(buf, pos)
            de, pos = read_varint(buf, pos)
            py += unzig(dy)
            ps += unzig(ds)
            pe += unzig(de)
            points.append((py, ps, pe))

        for (y0, s0, e0), (y1, s1, e1) in zip(points[:-1], points[1:]):
            if y1 < y0:
                continue
            for y in range(y0, y1 + 1):
                u = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
                s = max(0, int(round(s0 + u * (s1 - s0))))
                e = min(W, int(round(e0 + u * (e1 - e0))))
                if s < e:
                    mask[t, y, s:e] = 1
    return pos


def draw_car_spans(mask: np.ndarray, t: int, buf: bytes, pos: int) -> int:
    run_count, pos = read_varint(buf, pos)
    prev_x = 0
    for _ in range(run_count):
        dx, pos = read_varint(buf, pos)
        width, pos = read_varint(buf, pos)
        x0 = prev_x + dx
        prev_x = x0
        prev_top = 0
        prev_bot = 0
        for x in range(x0, x0 + width):
            dtop, pos = read_varint(buf, pos)
            dbot, pos = read_varint(buf, pos)
            prev_top += unzig(dtop)
            prev_bot += unzig(dbot)
            mask[t, prev_top:prev_bot, x] = 3
    return pos


def apply_residual(mask: np.ndarray, buf: bytes) -> None:
    flat = mask.reshape(-1)
    if buf.startswith((b"R2", b"R3")):
        compact_sizes = buf[:2] == b"R3"
        pos = 2
        stream_count = buf[pos]
        pos += 1
        for _ in range(stream_count):
            value = buf[pos]
            if compact_sizes:
                size = struct.unpack_from("<H", buf, pos + 1)[0]
                pos += 3
            else:
                size = struct.unpack_from("<I", buf, pos + 1)[0]
                pos += 5
            stream = brotli.decompress(buf[pos : pos + size])
            pos += size
            spos = 0
            start = 0
            while spos < len(stream):
                delta, spos = read_varint(stream, spos)
                start += delta
                length, spos = read_varint(stream, spos)
                flat[start : start + length] = value
        return

    pos = 0
    start = 0
    while pos < len(buf):
        delta, pos = read_varint(buf, pos)
        start += delta
        length, pos = read_varint(buf, pos)
        flat[start : start + length] = buf[pos]
        pos += 1


def load_semantic_vector_masks(path: Path, apply_resd: bool = True) -> torch.Tensor:
    sections = read_sections(path)
    road_top = decode_boundary(sections[b"RTOP"])
    hood_top = decode_boundary(sections[b"GTOP"])

    masks = np.full((NUM_FRAMES, H, W), 2, dtype=np.uint8)
    for t in range(NUM_FRAMES):
        for x in range(W):
            r = int(road_top[t, x])
            g = int(hood_top[t, x])
            if r < g:
                masks[t, r:g, x] = 0
            if g < H:
                masks[t, g:, x] = 4

    pos = 0
    lane_buf = sections[b"LANE"]
    for t in range(NUM_FRAMES):
        pos = draw_lane_segments(masks, t, lane_buf, pos)

    pos = 0
    car_buf = sections[b"CARS"]
    for t in range(NUM_FRAMES):
        pos = draw_car_spans(masks, t, car_buf, pos)

    if apply_resd:
        apply_residual(masks, sections[b"RESD"])
    return torch.from_numpy(masks).contiguous()



def _lane_row_intervals(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(xs) > 1) + 1
    parts = np.split(xs, breaks)
    return [(int(p[0]), int(p[-1] + 1)) for p in parts if p.size]


RESIDUAL_GAINS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0]
RESIDUAL_GAIN_BITS = 5


def _split_lane_gain_payload(raw: bytes) -> tuple[bytes, bytes | None]:
    if not raw.startswith(b"LG1"):
        return raw, None
    pos = 3
    lane_len, pos = read_varint(raw, pos)
    lane_raw = raw[pos : pos + lane_len]
    gain_raw = raw[pos + lane_len :]
    return lane_raw, gain_raw


def _unpack_gain_indices(raw: bytes, count: int = NUM_FRAMES) -> list[int]:
    out: list[int] = []
    acc = 0
    used = 0
    pos = 0
    mask = (1 << RESIDUAL_GAIN_BITS) - 1
    while len(out) < count:
        while used < RESIDUAL_GAIN_BITS:
            if pos >= len(raw):
                raise ValueError("truncated residual gain payload")
            acc |= int(raw[pos]) << used
            used += 8
            pos += 1
        out.append(acc & mask)
        acc >>= RESIDUAL_GAIN_BITS
        used -= RESIDUAL_GAIN_BITS
    if any(idx >= len(RESIDUAL_GAINS) for idx in out):
        raise ValueError("residual gain index out of range")
    return out


def _read_residual_gain_indices(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    raw = brotli.decompress(path.read_bytes())
    _lane_raw, gain_raw = _split_lane_gain_payload(raw)
    if gain_raw is None:
        return None
    return _unpack_gain_indices(gain_raw)


def _read_lane_control(path: Path) -> dict[int, tuple[int, int]]:
    raw = brotli.decompress(path.read_bytes())
    raw, _gain_raw = _split_lane_gain_payload(raw)
    pos = 0
    prev = 0
    out: dict[int, tuple[int, int]] = {}
    while pos < len(raw):
        delta, pos = read_varint(raw, pos)
        ds, pos = read_varint(raw, pos)
        de, pos = read_varint(raw, pos)
        idx = prev + delta
        out[idx] = (unzig(ds), unzig(de))
        prev = idx
    return out


def apply_lane_row_control(mask_frames_all: torch.Tensor, lane_control_path: Path, max_runs: int = 8) -> torch.Tensor:
    if not lane_control_path.exists():
        return mask_frames_all
    offsets = _read_lane_control(lane_control_path)
    base = mask_frames_all.cpu().numpy().astype(np.uint8)
    current_lane = base == 1
    adjusted_lane = np.zeros_like(current_lane, dtype=bool)
    record_idx = 0
    for t in range(NUM_FRAMES):
        for y in range(H):
            intervals = _lane_row_intervals(current_lane[t, y])
            for k, (s, e) in enumerate(intervals):
                if k >= max_runs:
                    adjusted_lane[t, y, s:e] = True
                    continue
                ds, de = offsets.get(record_idx, (0, 0))
                ns = max(0, min(W, s + int(ds)))
                ne = max(0, min(W, e + int(de)))
                if ne < ns:
                    ns, ne = ne, ns
                if ns < ne:
                    adjusted_lane[t, y, ns:ne] = True
                record_idx += 1
    out = base.copy()
    out[current_lane] = 0
    out[adjusted_lane] = 1
    return torch.from_numpy(out).contiguous()


def dilate_lane_torch(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    return F.max_pool2d(mask[:, None].float(), kernel_size=k, stride=1, padding=radius)[:, 0] > 0

def load_pose(path: Path) -> torch.Tensor:
    with open(path, "rb") as f:
        pose_bytes = brotli.decompress(f.read())
    return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()


def load_quantized_pose(path: Path) -> torch.Tensor:
    data = brotli.decompress(path.read_bytes())
    if data[:4] != b"PD16":
        raise ValueError("bad quantized pose payload")
    scale = struct.unpack_from("<i", data, 4)[0]
    delta = np.frombuffer(data, dtype="<i2", offset=8).reshape(NUM_FRAMES, 6).astype(np.int32)
    pose_q = np.cumsum(delta, axis=0)
    return torch.from_numpy((pose_q.astype(np.float32) / float(scale)).copy()).float()


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

    generator = JointFrameGenerator().to(device)
    weights = brotli.decompress((data_dir / "model.pt.br").read_bytes())
    if weights[:4] == b"QF4A":
        generator.load_state_dict(get_decoded_state_dict_qf4a(weights, device), strict=True)
    else:
        generator.load_state_dict(get_decoded_state_dict(weights, device), strict=True)
    generator.eval()

    lane_control_path = data_dir / "lane.rc.br"
    residual_gain_indices = _read_residual_gain_indices(lane_control_path)
    use_residual_gains = residual_gain_indices is not None
    mask_frames_all = load_semantic_vector_masks(data_dir / "mask.svc", apply_resd=True)
    source_mask_frames_all = apply_lane_row_control(mask_frames_all, lane_control_path)
    use_lane_control = not torch.equal(source_mask_frames_all, mask_frames_all)
    if use_residual_gains:
        noresd_mask_frames_all = load_semantic_vector_masks(data_dir / "mask.svc", apply_resd=False)
        noresd_source_mask_frames_all = apply_lane_row_control(noresd_mask_frames_all, lane_control_path)
    else:
        noresd_mask_frames_all = mask_frames_all
        noresd_source_mask_frames_all = source_mask_frames_all
    pose_path = data_dir / "pose.pq.br"
    if pose_path.exists():
        pose_frames_all = load_quantized_pose(pose_path)
    else:
        pose_frames_all = load_pose(data_dir / "pose.npy.br")

    cursor = 0
    batch_size = 4
    pairs_per_file = 600

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"
            file_masks = mask_frames_all[cursor : cursor + pairs_per_file]
            file_source_masks = source_mask_frames_all[cursor : cursor + pairs_per_file]
            file_noresd_masks = noresd_mask_frames_all[cursor : cursor + pairs_per_file]
            file_noresd_source_masks = noresd_source_mask_frames_all[cursor : cursor + pairs_per_file]
            file_gain_indices = residual_gain_indices[cursor : cursor + pairs_per_file] if residual_gain_indices is not None else None
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            cursor += pairs_per_file

            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")
                for i in pbar:
                    in_mask = file_masks[i : i + batch_size].to(device).long()
                    in_pose = file_poses[i : i + batch_size].to(device).float()

                    fake1, fake2 = generator(in_mask, in_pose)
                    if use_lane_control:
                        in_source_mask = file_source_masks[i : i + batch_size].to(device).long()
                        source_fake1, source_fake2 = generator(in_source_mask, in_pose)
                        paste = dilate_lane_torch(in_mask == 1, 8)[:, None]
                        fake2 = torch.where(paste, source_fake2, fake2)
                    fake1_up = F.interpolate(fake1, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1).clamp(0, 255).round()
                    if use_residual_gains:
                        in_noresd_mask = file_noresd_masks[i : i + batch_size].to(device).long()
                        noresd_fake1, noresd_fake2 = generator(in_noresd_mask, in_pose)
                        if use_lane_control:
                            in_noresd_source_mask = file_noresd_source_masks[i : i + batch_size].to(device).long()
                            noresd_source_fake1, noresd_source_fake2 = generator(in_noresd_source_mask, in_pose)
                            noresd_paste = dilate_lane_torch(in_noresd_mask == 1, 8)[:, None]
                            noresd_fake2 = torch.where(noresd_paste, noresd_source_fake2, noresd_fake2)
                        noresd_fake1_up = F.interpolate(noresd_fake1, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                        noresd_fake2_up = F.interpolate(noresd_fake2, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                        noresd_comp = torch.stack([noresd_fake1_up, noresd_fake2_up], dim=1).clamp(0, 255).round()
                        gain_idx = file_gain_indices[i : i + batch_size]
                        gains = torch.tensor([RESIDUAL_GAINS[int(idx)] for idx in gain_idx], device=device, dtype=batch_comp.dtype).view(-1, 1, 1, 1, 1)
                        batch_comp = noresd_comp + gains * (batch_comp - noresd_comp)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")
                    output = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output.cpu().numpy().tobytes())


if __name__ == "__main__":
    main()
