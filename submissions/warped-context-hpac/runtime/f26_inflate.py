"""F26 archive inflation using the compact production decoder."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from .carrier_repack import (
    materialize_cpr1,
    split_frame0_selector_carrier,
)
from .entropy.renderer_weight_codec import WANS1_MAGIC, decode_wans1
from .frame0_selector import apply_pixel_mode, decode_selector
from .hpac_inference import configure_cuda_reproducibility
from .residual_archive import decode_production_tokens, read_residual_archive


class InflationError(RuntimeError):
    """Raised when the fixed F26 archive cannot be inflated safely."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_renderer(renderer_dir: Path) -> ModuleType:
    renderer_dir = renderer_dir.resolve()
    path = renderer_dir / "inflate.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_f26_renderer"
    module = sys.modules.get(name)
    if module is not None:
        if Path(module.__file__).resolve() != path:
            raise InflationError("a different F26 renderer is already loaded")
        return module
    sys.path.insert(0, str(renderer_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise InflationError(f"cannot load renderer from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path.pop(0)


def _apply_frame0_selector(
    destination: Path,
    renderer,
    selector_blob: bytes,
) -> dict[str, object]:
    modes, indices = decode_selector(selector_blob)
    if indices.size != renderer.N:
        raise InflationError("frame-0 selector count does not match frame count")
    output = np.memmap(
        destination,
        mode="r+",
        dtype=np.uint8,
        shape=(renderer.N * 2, renderer.CAMERA_H, renderer.CAMERA_W, 3),
    )
    for mode_index, mode in enumerate(modes):
        frame_ids = np.flatnonzero(indices == mode_index)
        if not frame_ids.size:
            continue
        output[2 * frame_ids] = apply_pixel_mode(
            np.asarray(output[2 * frame_ids]).copy(), mode
        )
    output.flush()
    return {
        "payload_bytes": len(selector_blob),
        "payload_sha256": hashlib.sha256(selector_blob).hexdigest(),
        "mode_count": len(modes),
        "frame_count": int(indices.size),
    }


def inflate_archive(
    archive_path: Path,
    destination: Path,
    *,
    renderer_dir: Path,
) -> dict[str, object]:
    """Inflate the F26 archive to the challenge raw-video format."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise InflationError("PyTorch is required to inflate F26") from error
    if not torch.cuda.is_available():
        raise InflationError("F26 inflation requires a CUDA-capable GPU")

    device = torch.device("cuda")
    configure_cuda_reproducibility()
    parts = read_residual_archive(archive_path)
    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":
        raise InflationError("archive does not use the fixed F26 residual schema")
    if parts.table is None:
        raise InflationError("F26 requires a residual correction table")
    if not parts.semantic_blob.startswith(WANS1_MAGIC):
        raise InflationError("F26 requires WANS1 semantic weights")

    renderer = _load_renderer(renderer_dir)
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = materialize_cpr1(carrier_blob, renderer)
    semantic_width_marker = bytes(40_252)
    semantic_pose = (
        struct.pack("<II", len(semantic_width_marker), len(canonical_carrier))
        + semantic_width_marker
        + canonical_carrier
    )
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    semantic = renderer.SemanticTokenRenderer(96)
    records = decode_wans1(parts.semantic_blob)
    state = {
        record.schema.name: torch.from_numpy(
            np.ascontiguousarray(record.values, dtype=np.float32)
        )
        for record in records
    }
    semantic.load_state_dict(state, strict=True)

    started = time.time()
    tokens, token_report = decode_production_tokens(
        parts, renderer, renderer_dir, device
    )
    renderer.render_video(semantic, basis, coefficients, tokens, destination, device)
    selector_report = (
        None
        if selector_blob is None
        else _apply_frame0_selector(
            destination,
            renderer,
            selector_blob,
        )
    )
    torch.cuda.synchronize(device)
    if not destination.is_file():
        raise InflationError("renderer did not create the requested raw output")
    return {
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "decode_and_render_seconds": time.time() - started,
        "raw_bytes": destination.stat().st_size,
        "residual_schema": parts.schema,
        "selector": selector_report,
        "token_decoder": token_report,
    }
