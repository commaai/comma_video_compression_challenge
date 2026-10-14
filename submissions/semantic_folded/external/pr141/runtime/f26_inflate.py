"""Device-parameterized F26 inflation with retained stage checkpoints.

This is a lifted copy of the promoted F26 decoder.  It preserves the archive
parse, token probability generation, renderer operations, operation order, and
portable dtypes while making the execution device explicit.  CPU execution is
an opt-in; the default remains CUDA-required.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import platform
import shutil
import struct
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .carrier_repack import materialize_cpr1, split_frame0_selector_carrier
from .compensation_overlay import apply_compensation_overlay
from .entropy.renderer_weight_codec import WANS1_MAGIC, decode_wans1
from .frame0_selector import apply_pixel_mode, decode_selector
from .hpac_inference import configure_cuda_reproducibility
from .residual_archive import decode_production_tokens, read_residual_archive


class InflationError(RuntimeError):
    """Raised when the fixed F26 archive cannot be inflated safely."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _source_tree_sha256(*roots: Path) -> str:
    records = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            records.append(
                {
                    "root": root.name,
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    *,
    pair_count: int | None = None,
) -> dict[str, object]:
    modes, indices = decode_selector(selector_blob)
    if indices.size != renderer.N:
        raise InflationError("frame-0 selector count does not match frame count")
    pair_count = renderer.N if pair_count is None else pair_count
    if pair_count < 1 or pair_count > renderer.N:
        raise InflationError("selector pair_count is outside the renderer field")
    indices = indices[:pair_count]
    output = np.memmap(
        destination,
        mode="r+",
        dtype=np.uint8,
        shape=(pair_count * 2, renderer.CAMERA_H, renderer.CAMERA_W, 3),
    )
    for mode_index, mode in enumerate(modes):
        frame_ids = np.flatnonzero(indices == mode_index)
        if not frame_ids.size:
            continue
        output[2 * frame_ids] = apply_pixel_mode(np.asarray(output[2 * frame_ids]).copy(), mode)
    output.flush()
    return {
        "payload_bytes": len(selector_blob),
        "payload_sha256": hashlib.sha256(selector_blob).hexdigest(),
        "mode_count": len(modes),
        "frame_count": int(indices.size),
    }


def _configure_device(torch, requested: str, num_threads: int | None) -> tuple[Any, dict[str, Any]]:
    if requested not in {"cpu", "cuda"}:
        raise InflationError("F26 device must be explicitly 'cpu' or 'cuda'")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InflationError("F26 CUDA inflation requires a CUDA-capable GPU")
    if device.type == "cpu":
        if num_threads != 4:
            raise InflationError("F26 contest-CPU inflation requires num_threads=4")
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before parallel work.
            # The staged entrypoint imports no torch work before this call.
            if torch.get_num_interop_threads() != 1:
                raise
        reproducibility = {
            "cpu_num_threads": torch.get_num_threads(),
            "cpu_num_interop_threads": torch.get_num_interop_threads(),
        }
    else:
        reproducibility = configure_cuda_reproducibility()
    return device, {
        "requested_device": requested,
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "reproducibility": reproducibility,
    }


def _checkpoint_binding(
    *,
    archive_path: Path,
    parts,
    renderer_dir: Path,
    device_report: dict[str, Any],
    pair_count: int,
    token_decoder: str,
    token_decoder_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "archive_sha256": _sha256_file(archive_path),
        "token_stream_sha256": hashlib.sha256(parts.token_stream).hexdigest(),
        "hpac_blob_sha256": hashlib.sha256(parts.hpac_blob).hexdigest(),
        "residual_payload_sha256": hashlib.sha256(parts.residual_payload).hexdigest(),
        "source_tree_sha256": _source_tree_sha256(
            Path(__file__).resolve().parent,
            renderer_dir.resolve(),
        ),
        "device": device_report,
        "pair_count": pair_count,
        "token_decoder": token_decoder,
        "token_decoder_fingerprint": token_decoder_fingerprint,
    }


def _load_token_checkpoint(
    checkpoint_dir: Path,
    *,
    binding: dict[str, Any],
    renderer,
    pair_count: int,
) -> tuple[Any, dict[str, Any]] | None:
    receipt_path = checkpoint_dir / "tokens_cpu_stage_complete.json"
    token_path = checkpoint_dir / "tokens_cpu_stage_complete.u8"
    if not receipt_path.exists() and not token_path.exists():
        return None
    if not receipt_path.is_file() or not token_path.is_file():
        raise InflationError("incomplete token checkpoint pair; refusing silent restart")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("binding") != binding:
        raise InflationError("token checkpoint binding differs from this runtime/archive")
    expected_bytes = pair_count * renderer.EVAL_H * renderer.EVAL_W
    if token_path.stat().st_size != expected_bytes:
        raise InflationError("token checkpoint byte count differs from F26 geometry")
    if _sha256_file(token_path) != receipt.get("tokens", {}).get("sha256"):
        raise InflationError("token checkpoint SHA-256 differs from its receipt")
    import torch

    array = np.fromfile(token_path, dtype=np.uint8).reshape(
        pair_count, renderer.EVAL_H, renderer.EVAL_W
    )
    return torch.from_numpy(array), dict(receipt["token_decoder"])


def _write_token_checkpoint(
    checkpoint_dir: Path,
    tokens,
    *,
    binding: dict[str, Any],
    token_report: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    token_path = checkpoint_dir / "tokens_cpu_stage_complete.u8"
    receipt_path = checkpoint_dir / "tokens_cpu_stage_complete.json"
    if token_path.exists() or receipt_path.exists():
        raise InflationError("refusing to overwrite an existing token checkpoint")
    array = tokens.detach().cpu().contiguous().numpy().astype(np.uint8, copy=False)
    temporary = token_path.with_name(f".{token_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        array.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, token_path)
    receipt = {
        "schema": "ddm_f26p_token_checkpoint.v1",
        "complete": True,
        "binding": binding,
        "tokens": _file_fact(token_path),
        "token_decoder": token_report,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _preserve_interrupted_render(partial: Path, checkpoint_dir: Path) -> None:
    if not partial.exists():
        return
    interrupted = checkpoint_dir / "interrupted_renders"
    interrupted.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    retained = interrupted / f"render_incomplete_{stamp}.raw"
    os.replace(partial, retained)
    _atomic_json(
        retained.with_suffix(".json"),
        {
            "schema": "ddm_f26p_interrupted_render.v1",
            "complete": False,
            "reason": "prior render did not reach the atomic destination rename",
            "retained_partial": _file_fact(retained),
        },
    )


def _parse_pair_count(renderer) -> int:
    text = os.environ.get("F26_ADVISORY_PAIR_LIMIT", "").strip()
    if not text:
        return int(renderer.N)
    try:
        pair_count = int(text)
    except ValueError as exc:
        raise InflationError("F26_ADVISORY_PAIR_LIMIT must be a positive integer") from exc
    if pair_count < 1 or pair_count > int(renderer.N):
        raise InflationError("F26_ADVISORY_PAIR_LIMIT is outside the real field")
    return pair_count


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_decoder_fingerprint(
    *,
    renderer,
    renderer_dir: Path,
    token_decoder: str,
    num_threads: int,
) -> dict[str, Any]:
    runtime_dir = Path(__file__).resolve().parent
    files = [
        runtime_dir / "residual_archive.py",
        runtime_dir / "hpac_inference.py",
        runtime_dir / "ihs2.py",
        renderer_dir / "hpac_integer.py",
        renderer_dir / "hpac_integer_sparse.py",
        renderer_dir / "integer_model_io.py",
    ]
    if token_decoder == "native-hpac":
        files.extend(
            [
                runtime_dir / "f26_hpac_native.py",
                runtime_dir / "f26_hpac_native.c",
            ]
        )
        library_text = os.environ.get("F26_HPAC_NATIVE_LIBRARY", "").strip()
        if not library_text or not Path(library_text).is_file():
            raise InflationError("native token mode requires F26_HPAC_NATIVE_LIBRARY")
        files.append(Path(library_text).resolve())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise InflationError(f"token decoder fingerprint inputs are absent: {missing}")
    records = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    renderer_contract = {
        "load_hpac_source_sha256": _sha256_bytes(inspect.getsource(renderer.load_hpac).encode()),
        "group_masks_source_sha256": _sha256_bytes(inspect.getsource(renderer.group_masks).encode()),
        "constants": {
            name: int(getattr(renderer, name))
            for name in (
                "N",
                "NUM_CLASSES",
                "EVAL_H",
                "EVAL_W",
                "HPAC_PATCH",
                "HPAC_DELTA",
                "HPAC_CHANNELS",
                "HPAC_FILM_DIM",
                "HPAC_LOGIT_PRECISION",
            )
        },
    }
    payload = {
        "token_decoder": token_decoder,
        "files": records,
        "renderer_contract": renderer_contract,
        "thread_config": {"torch_num_threads": num_threads, "torch_interop_threads": 1},
    }
    payload["sha256"] = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    return payload


def _token_cache_binding(
    *,
    parts,
    pair_count: int,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    table = np.ascontiguousarray(parts.table.values, dtype="<f4")
    return {
        "schema": "ddm_wc1_token_cache_binding.v1",
        "archive_sections": {
            "token_stream_sha256": _sha256_bytes(parts.token_stream),
            "hpac_blob_sha256": _sha256_bytes(parts.hpac_blob),
            "correction_table_sha256": hashlib.sha256(memoryview(table).cast("B")).hexdigest(),
        },
        "decoder_code_sha256": fingerprint["sha256"],
        "thread_config": fingerprint["thread_config"],
        "pair_count": pair_count,
    }


def _copy_retained_render(source: Path, destination: Path) -> float:
    started = time.perf_counter()
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    return time.perf_counter() - started


def inflate_archive(
    archive_path: Path,
    destination: Path,
    *,
    renderer_dir: Path,
    device_name: str = "cuda",
    num_threads: int | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, object]:
    """Inflate the real F26 archive on an explicit device.

    A durable token checkpoint is required for CPU runs.  This gives the long
    decode and render phases a crash-resume boundary without changing decoded
    values or the renderer's operation sequence.
    """
    archive_path = archive_path.resolve()
    destination = destination.resolve()
    renderer_dir = renderer_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise InflationError("PyTorch is required to inflate F26") from error
    device, device_report = _configure_device(torch, device_name, num_threads)
    if device.type == "cpu" and checkpoint_dir is None:
        raise InflationError("CPU inflation requires a durable checkpoint_dir")

    total_started = time.perf_counter()
    setup_started = time.perf_counter()
    parts = read_residual_archive(archive_path)
    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":
        raise InflationError("archive does not use the fixed F26 residual schema")
    if parts.table is None:
        raise InflationError("F26 requires a residual correction table")
    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):
        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")

    renderer = _load_renderer(renderer_dir)
    pair_count = _parse_pair_count(renderer)
    token_decoder = os.environ.get("F26_TOKEN_DECODER", "python").strip()
    if token_decoder not in {"python", "native-hpac"}:
        raise InflationError("F26_TOKEN_DECODER must be 'python' or 'native-hpac'")
    if token_decoder != "python":
        raise InflationError(
            "this generation wires the ddm_rr2 free probability corrector into the "
            "python token decoder only; the native-hpac path is unpatched and would "
            "decode a different field, so it is refused rather than trusted"
        )
    if pair_count != int(renderer.N) and token_decoder != "native-hpac":
        raise InflationError("advisory prefix inflation requires the resumable native token path")
    token_decoder_fingerprint = _token_decoder_fingerprint(
        renderer=renderer,
        renderer_dir=renderer_dir,
        token_decoder=token_decoder,
        num_threads=int(num_threads or 0),
    )
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = materialize_cpr1(carrier_blob, renderer)
    semantic_width_marker = bytes(40_252)
    semantic_pose = (
        struct.pack("<II", len(semantic_width_marker), len(canonical_carrier))
        + semantic_width_marker
        + canonical_carrier
    )
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    compensation_report = None
    if parts.compensation_blob is not None:
        basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
        _, _, coefficient_scales, encoded = renderer.decode_compact_carrier(
            canonical_carrier,
            basis_count=basis_count,
            frames=renderer.N,
            dimensions=renderer.CARRIER_DIM,
        )
        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
        base_codes = np.cumsum(delta, axis=0) & 0xFFF
        base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes).astype(np.int32)
        expected_base = torch.from_numpy(base_codes).float() * torch.from_numpy(coefficient_scales)[None]
        if not torch.equal(coefficients, expected_base):
            raise InflationError("compensation base-code reconstruction differs")
        candidate_codes = apply_compensation_overlay(base_codes, parts.compensation_blob)
        coefficients = torch.from_numpy(candidate_codes).float() * torch.from_numpy(coefficient_scales)[None]
        compensation_report = {
            "payload_bytes": len(parts.compensation_blob),
            "payload_sha256": hashlib.sha256(parts.compensation_blob).hexdigest(),
            "changed_coordinates": int(np.count_nonzero(candidate_codes != base_codes)),
        }
    semantic = renderer.SemanticTokenRenderer(96)
    tagged_state = renderer.unpack_variant_semantic_or_none(
        parts.semantic_blob,
        semantic.state_dict(),
    )
    if tagged_state is None:
        records = decode_wans1(parts.semantic_blob)
        tagged_state = {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32)
            )
            for record in records
        }
    semantic.load_state_dict(tagged_state, strict=True)
    setup_seconds = time.perf_counter() - setup_started

    binding = _checkpoint_binding(
        archive_path=archive_path,
        parts=parts,
        renderer_dir=renderer_dir,
        device_report=device_report,
        pair_count=pair_count,
        token_decoder=token_decoder,
        token_decoder_fingerprint=token_decoder_fingerprint,
    )
    checkpoint_resume = False
    token_cache_report: dict[str, Any] = {"status": "DISABLED"}
    token_path: Path | None = None
    token_started = time.perf_counter()
    loaded = (
        None
        if checkpoint_dir is None
        else _load_token_checkpoint(
            checkpoint_dir.resolve(),
            binding=binding,
            renderer=renderer,
            pair_count=pair_count,
        )
    )
    cache_root_text = os.environ.get("F26_ADVISORY_DECODE_CACHE_ROOT", "").strip()
    cache_binding = _token_cache_binding(
        parts=parts,
        pair_count=pair_count,
        fingerprint=token_decoder_fingerprint,
    )
    expected_token_bytes = pair_count * renderer.EVAL_H * renderer.EVAL_W
    if pair_count != int(renderer.N) and cache_root_text:
        raise InflationError("prefix runs cannot populate or consume the full-field token cache")
    if loaded is None and cache_root_text:
        from . import ddm_wc1_advisory_runtime as wc1

        cache_hit = wc1.load_token_cache(
            Path(cache_root_text),
            cache_binding,
            expected_bytes=expected_token_bytes,
        )
        if cache_hit is not None:
            import torch

            cached_path, token_cache_report = cache_hit
            cached = np.memmap(
                cached_path,
                mode="r",
                dtype=np.uint8,
                shape=(pair_count, renderer.EVAL_H, renderer.EVAL_W),
            )
            tokens = torch.from_numpy(np.asarray(cached).copy())
            created = token_cache_report.get("created")
            if not isinstance(created, dict) or not isinstance(created.get("token_decoder"), dict):
                raise InflationError("token-cache hit lacks its decoder report")
            token_report = dict(created["token_decoder"])
            token_report["cache_reuse"] = {
                "status": "HIT",
                "key": token_cache_report["key"],
                "payload_sha256": token_cache_report["payload"]["sha256"],
            }
            if checkpoint_dir is not None:
                _write_token_checkpoint(
                    checkpoint_dir.resolve(),
                    tokens,
                    binding=binding,
                    token_report=token_report,
                )
            loaded = (tokens, token_report)
    if loaded is None:
        if token_decoder == "native-hpac":
            if checkpoint_dir is None:
                raise InflationError("native token decode requires checkpoint_dir")
            from .f26_hpac_native import decode_native_tokens

            native_progress = checkpoint_dir.resolve() / "native_hpac_progress"
            tokens, token_report = decode_native_tokens(
                parts,
                renderer,
                renderer_dir,
                device,
                frame_limit=pair_count,
                output_path=native_progress / "tokens_partial.u8",
                checkpoint_dir=native_progress / "checkpoints",
            )
        else:
            tokens, token_report = decode_production_tokens(parts, renderer, renderer_dir, device)
        if checkpoint_dir is not None:
            _write_token_checkpoint(
                checkpoint_dir.resolve(),
                tokens,
                binding=binding,
                token_report=token_report,
            )
        if cache_root_text:
            from . import ddm_wc1_advisory_runtime as wc1

            token_path = checkpoint_dir.resolve() / "tokens_cpu_stage_complete.u8"
            _, token_cache_report = wc1.publish_token_cache(
                Path(cache_root_text),
                cache_binding,
                source=token_path,
                expected_bytes=expected_token_bytes,
                created={
                    "token_decoder": token_report,
                    "archive_sha256": _sha256_file(archive_path),
                    "runtime_source_tree_sha256": binding["source_tree_sha256"],
                    "host": platform.node(),
                },
            )
    else:
        tokens, token_report = loaded
        checkpoint_resume = True
        if token_cache_report.get("status") == "DISABLED":
            token_cache_report = {"status": "LOCAL_CHECKPOINT"}
    if checkpoint_dir is not None:
        token_path = checkpoint_dir.resolve() / "tokens_cpu_stage_complete.u8"
    if token_path is None or not token_path.is_file():
        raise InflationError("retained token payload is required before render")
    token_stage_seconds = time.perf_counter() - token_started

    render_started = time.perf_counter()
    render_workers = os.environ.get("F26_ADVISORY_RENDER_WORKERS", "").strip()
    parallel_report = None
    render_copy_seconds = 0.0
    if render_workers and render_workers != "1":
        if checkpoint_dir is None:
            raise InflationError("parallel advisory rendering requires checkpoint_dir")
        from . import ddm_wc1_advisory_runtime as wc1

        rss_text = os.environ.get("F26_ADVISORY_RENDER_RSS_BYTES", "").strip()
        measured_rss = None if not rss_text else int(rss_text)
        parallel_root = checkpoint_dir.resolve() / "parallel_render"
        render_stage = parallel_root / "render_stage.raw"
        parallel_report = wc1.render_video_parallel(
            semantic=semantic,
            basis=basis,
            coefficients=coefficients,
            token_path=token_path,
            token_sha256=_sha256_file(token_path),
            renderer_dir=renderer_dir,
            output_path=render_stage,
            progress_dir=parallel_root,
            pair_count=pair_count,
            camera_height=int(renderer.CAMERA_H),
            camera_width=int(renderer.CAMERA_W),
            requested_workers=render_workers,
            per_process_threads=int(num_threads or 0),
            measured_worker_rss_bytes=measured_rss,
        )
        partial = checkpoint_dir.resolve() / "selector_cpu_in_progress.raw"
        _preserve_interrupted_render(partial, checkpoint_dir.resolve())
        render_copy_seconds = _copy_retained_render(render_stage, partial)
    elif checkpoint_dir is None:
        partial = destination.with_name(f".{destination.name}.partial")
    else:
        checkpoint_dir = checkpoint_dir.resolve()
        partial = checkpoint_dir / "render_cpu_in_progress.raw"
        _preserve_interrupted_render(partial, checkpoint_dir)
    if parallel_report is None:
        if pair_count != int(renderer.N):
            raise InflationError("prefix rendering requires F26_ADVISORY_RENDER_WORKERS")
        renderer.render_video(semantic, basis, coefficients, tokens, partial, device)
    render_seconds = time.perf_counter() - render_started

    selector_started = time.perf_counter()
    selector_report = (
        None
        if selector_blob is None
        else _apply_frame0_selector(
            partial,
            renderer,
            selector_blob,
            pair_count=pair_count,
        )
    )
    selector_seconds = time.perf_counter() - selector_started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not partial.is_file():
        raise InflationError("renderer did not create the requested raw output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    raw_sha256 = _sha256_file(destination)
    total_seconds = time.perf_counter() - total_started
    return {
        "schema": "ddm_f26p_inflate_report.v1",
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "decode_and_render_seconds": total_seconds,
        "raw_bytes": destination.stat().st_size,
        "raw_sha256": raw_sha256,
        "pair_count": pair_count,
        "residual_schema": parts.schema,
        "compensation": compensation_report,
        "selector": selector_report,
        "token_decoder": token_report,
        "token_cache": token_cache_report,
        "parallel_render": parallel_report,
        "device": device_report,
        "checkpoint_resume": checkpoint_resume,
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "stage_seconds": {
            "archive_setup": setup_seconds,
            "token_decode_or_checkpoint_load": token_stage_seconds,
            "neural_render_and_resize": render_seconds,
            "render_checkpoint_copy": render_copy_seconds,
            "frame0_selector_and_io": selector_seconds,
            "total_including_raw_sha256": total_seconds,
        },
    }
