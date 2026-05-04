"""hnerv_repack_latent inflate: load our compact archive, run AaronLeslie138's HNeRV decoder.

Archive format (single 0.bin file):
  u32 dec_len   | dec_blob (brotli)   — concatenated INT8 codes (schema-driven)
  u32 sca_len   | sca_blob            — fp16 scales, one per tensor in schema order
  u32 lat_len   | lat_blob (brotli)   — per-dim asym uint8 + delta + lo/hi split
  u32 wrp_len   | wrp_blob (brotli)   — per-pair (u8 dim, i8 quant_delta), dim=255 means no-op

Credits: HNeRV decoder weights and architecture by AaronLeslie138 (PR #95 / hnerv_muon).
This submission re-packs his archive ~470 B smaller via schema-driven layer names + fp16 scales,
and adds a ~1.2 KB latent-correction sidecar (per-pair single-dim perturbation chosen to
minimize SegNet+PoseNet distortion).
"""
import io, os, shutil, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _ensure_brotli():
    try:
        import brotli  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'brotli'])
_ensure_brotli()
import brotli
import torch
import torch.nn.functional as F

from hnerv_model import HNeRVDecoder
from schema import SCHEMA, META
from sidecar import decode_corrections, apply_corrections

CAMERA_H, CAMERA_W = 874, 1164
NATIVE_H, NATIVE_W = META['eval_size']  # (384, 512)


def split_archive(b):
    o = 0
    parts = []
    for _ in range(4):
        L = struct.unpack_from('<I', b, o)[0]; o += 4
        parts.append(b[o:o+L]); o += L
    if o != len(b):
        raise RuntimeError(f'archive trailing: {o} vs {len(b)}')
    return parts  # [dec, sca, lat, wrp]


def decode_decoder(blob, sca_blob):
    raw = brotli.decompress(blob)
    codes = np.frombuffer(raw, dtype=np.int8)
    scales = np.frombuffer(sca_blob, dtype=np.float16)
    sd = {}
    o = 0
    for i, (name, shape) in enumerate(SCHEMA):
        n_el = int(np.prod(shape))
        chunk = codes[o:o+n_el].reshape(shape)
        sd[name] = torch.from_numpy(chunk.astype(np.float32) * float(scales[i]))
        o += n_el
    if o != codes.size:
        raise RuntimeError(f'decoder leftover: {o} vs {codes.size}')
    return sd


def decode_latents(blob):
    raw = brotli.decompress(blob)
    buf = io.BytesIO(raw)
    n, d = struct.unpack('<II', buf.read(8))
    mins = np.frombuffer(buf.read(d*2), dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(buf.read(d*2), dtype=np.float16).astype(np.float32)
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2,
                     -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i-1] + delta[i]
    return torch.from_numpy(q.astype(np.float32) * scales[None, :] + mins[None, :])


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, 'rb') as f:
        archive_bytes = f.read()
    print(f'[inflate] archive {len(archive_bytes)} bytes', flush=True)

    dec_b, sca_b, lat_b, wrp_b = split_archive(archive_bytes)
    print(f'[inflate] dec {len(dec_b)} sca {len(sca_b)} lat {len(lat_b)} wrp {len(wrp_b)}', flush=True)

    sd = decode_decoder(dec_b, sca_b)
    latents = decode_latents(lat_b)
    if wrp_b:
        dim_arr, delta_q_arr = decode_corrections(wrp_b)
        print(f'[inflate] sidecar: {(dim_arr != 255).sum()} pairs with corrections', flush=True)
        apply_corrections(latents, dim_arr, delta_q_arr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[inflate] device={device}', flush=True)
    decoder = HNeRVDecoder(latent_dim=META['latent_dim'], base_channels=META['base_channels'], eval_size=tuple(META['eval_size'])).to(device)
    decoder.load_state_dict(sd)
    decoder.eval()
    latents = latents.to(device)

    n_pairs = META['n_pairs']
    n = 0
    with torch.inference_mode(), open(dst_raw, 'wb') as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, 384, 512)
            flat = decoded.reshape(B*2, 3, NATIVE_H, NATIVE_W)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode='bicubic', align_corners=False)
            frames = (up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy())
            fout.write(frames.tobytes())
            n += B * 2
    print(f'[inflate] wrote {n} frames to {dst_raw}', flush=True)
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.hnerv_muon_lc.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
