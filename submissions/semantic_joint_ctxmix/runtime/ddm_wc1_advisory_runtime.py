"""Advisory-only F26 token-cache and frame-parallel render primitives.

The shipping packet never imports this module.  A WC1 preparation step copies
it into a private advisory runtime tree, where all three accelerators remain
disabled unless the caller supplies explicit environment flags.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class WC1AdvisoryError(RuntimeError):
    """An advisory cache, resource, or byte-identity invariant failed."""


TOKEN_CACHE_SCHEMA = "ddm_wc1_token_cache.v1"
RENDER_PROGRESS_SCHEMA = "ddm_wc1_parallel_render_progress.v1"
RENDER_COMPLETE_SCHEMA = "ddm_wc1_parallel_render_complete.v1"
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise WC1AdvisoryError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WC1AdvisoryError(f"manifest root must be an object: {path}")
    if raw != canonical_json(value):
        raise WC1AdvisoryError(f"manifest is not canonically encoded: {path}")
    return value


def token_cache_key(binding: Mapping[str, Any]) -> str:
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_cache_entry(
    entry: Path,
    binding: Mapping[str, Any],
    *,
    expected_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = entry / "manifest.json"
    payload_path = entry / "tokens.u8"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise WC1AdvisoryError(f"incomplete token-cache entry: {entry}")
    manifest = _canonical_manifest(manifest_path)
    key = token_cache_key(binding)
    blockers: list[str] = []
    if manifest.get("schema") != TOKEN_CACHE_SCHEMA:
        blockers.append("schema")
    if manifest.get("key") != key or entry.name != key:
        blockers.append("key")
    if manifest.get("binding") != dict(binding):
        blockers.append("binding")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        blockers.append("payload_manifest")
    else:
        if payload.get("path") != str(payload_path.resolve()):
            blockers.append("payload_path")
        if payload.get("bytes") != expected_bytes or payload_path.stat().st_size != expected_bytes:
            blockers.append("payload_bytes")
        if payload.get("sha256") != sha256_file(payload_path):
            blockers.append("payload_sha256")
    if blockers:
        raise WC1AdvisoryError(f"token-cache entry failed validation {entry}: {blockers}")
    return payload_path, manifest


def load_token_cache(
    root: Path,
    binding: Mapping[str, Any],
    *,
    expected_bytes: int,
) -> tuple[Path, dict[str, Any]] | None:
    """Return a fully rehashed cache hit, or ``None`` for a clean miss."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = token_cache_key(binding)
    entry = root / key
    lock_path = root / f".{key}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not entry.exists():
            return None
        payload_path, manifest = _validate_cache_entry(
            entry, binding, expected_bytes=expected_bytes
        )
    return payload_path, {
        "status": "HIT",
        "key": key,
        "entry": str(entry),
        "manifest": file_fact(entry / "manifest.json"),
        "payload": file_fact(payload_path),
        "validation": "canonical-manifest plus full-payload SHA-256",
        "created": manifest.get("created"),
    }


def publish_token_cache(
    root: Path,
    binding: Mapping[str, Any],
    *,
    source: Path,
    expected_bytes: int,
    created: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Atomically publish a retained token payload under a single-flight lock."""

    source = source.resolve()
    if not source.is_file() or source.stat().st_size != expected_bytes:
        raise WC1AdvisoryError("token-cache source is absent or has the wrong byte count")
    source_fact = file_fact(source)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = token_cache_key(binding)
    entry = root / key
    lock_path = root / f".{key}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if entry.exists():
            payload_path, _ = _validate_cache_entry(
                entry, binding, expected_bytes=expected_bytes
            )
            if sha256_file(payload_path) != source_fact["sha256"]:
                raise WC1AdvisoryError("existing token-cache payload differs from the fresh decode")
            status = "RACE_HIT_IDENTICAL"
        else:
            temporary = root / f".{key}.{os.getpid()}.partial"
            if temporary.exists():
                raise WC1AdvisoryError(f"stale cache partial blocks population: {temporary}")
            temporary.mkdir()
            payload_path = temporary / "tokens.u8"
            with source.open("rb") as src, payload_path.open("wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            payload_fact = file_fact(payload_path)
            if payload_fact["sha256"] != source_fact["sha256"]:
                raise WC1AdvisoryError("token-cache copy changed payload bytes")
            manifest = {
                "schema": TOKEN_CACHE_SCHEMA,
                "key": key,
                "binding": dict(binding),
                "payload": {
                    **payload_fact,
                    "path": str((entry / "tokens.u8").resolve()),
                },
                "source": source_fact,
                "created": dict(created),
                "certify_or_block": {
                    "rebuildable": True,
                    "reason": "exact token decoder inputs and code identity are content-addressed",
                    "deletion_authorized": False,
                },
            }
            atomic_json(temporary / "manifest.json", manifest)
            os.replace(temporary, entry)
            payload_path, _ = _validate_cache_entry(
                entry, binding, expected_bytes=expected_bytes
            )
            status = "POPULATED"
    return payload_path, {
        "status": status,
        "key": key,
        "entry": str(entry),
        "manifest": file_fact(entry / "manifest.json"),
        "payload": file_fact(payload_path),
        "source": source_fact,
    }


def _available_memory_bytes() -> tuple[int, str]:
    if sys.platform == "linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024, "/proc/meminfo:MemAvailable"
        except (OSError, ValueError, IndexError):
            pass
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["/usr/bin/vm_stat"], check=False, capture_output=True, text=True
        )
        if completed.returncode == 0:
            lines = completed.stdout.splitlines()
            try:
                page_size = int(lines[0].split("page size of ", 1)[1].split(" bytes", 1)[0])
                values: dict[str, int] = {}
                for line in lines[1:]:
                    if ":" not in line:
                        continue
                    name, raw = line.split(":", 1)
                    values[name] = int(raw.strip().rstrip("."))
                pages = sum(
                    values.get(name, 0)
                    for name in (
                        "Pages free",
                        "Pages inactive",
                        "Pages speculative",
                        "Pages purgeable",
                    )
                )
                if pages > 0:
                    return pages * page_size, "vm_stat free+inactive+speculative+purgeable"
            except (ValueError, IndexError):
                pass
    raise WC1AdvisoryError("cannot derive available RAM for advisory worker admission")


def resolve_render_workers(
    requested: str,
    *,
    per_process_threads: int,
    measured_worker_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Resolve a non-latched worker count from live CPU and RAM capacity."""

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise WC1AdvisoryError("os.cpu_count() is unavailable")
    available_memory, memory_source = _available_memory_bytes()
    if per_process_threads < 1:
        raise WC1AdvisoryError("per-process thread count must be positive")
    cpu_reserve = min(2, max(0, cpu_count - per_process_threads))
    cpu_capacity = max(1, (cpu_count - cpu_reserve) // per_process_threads)
    if measured_worker_rss_bytes is None:
        worker_rss_budget = 4 * (1 << 30)
        rss_source = "conservative pre-measurement 4 GiB budget"
    else:
        if measured_worker_rss_bytes <= 0:
            raise WC1AdvisoryError("measured worker RSS must be positive")
        worker_rss_budget = math.ceil(measured_worker_rss_bytes * 1.25)
        rss_source = "measured peak RSS plus 25 percent safety margin"
    ram_reserve = max(2 * (1 << 30), worker_rss_budget)
    ram_capacity = max(1, (available_memory - ram_reserve) // worker_rss_budget)
    if requested == "auto":
        workers = max(1, min(cpu_capacity, ram_capacity))
    else:
        try:
            workers = int(requested)
        except ValueError as exc:
            raise WC1AdvisoryError("render workers must be 'auto' or a positive integer") from exc
        if workers < 1:
            raise WC1AdvisoryError("render worker count must be positive")
        if workers > cpu_capacity or workers > ram_capacity:
            raise WC1AdvisoryError(
                f"requested {workers} workers exceeds live capacity cpu={cpu_capacity} ram={ram_capacity}"
            )
    return {
        "requested": requested,
        "selected": int(workers),
        "cpu_count": cpu_count,
        "cpu_reserve": cpu_reserve,
        "per_process_threads": per_process_threads,
        "cpu_capacity": int(cpu_capacity),
        "available_memory_bytes": int(available_memory),
        "available_memory_source": memory_source,
        "worker_rss_budget_bytes": int(worker_rss_budget),
        "worker_rss_budget_source": rss_source,
        "ram_reserve_bytes": int(ram_reserve),
        "ram_capacity": int(ram_capacity),
    }


_WORKER_STATE: dict[str, Any] = {}


def _load_renderer(renderer_dir: Path) -> Any:
    name = f"_ddm_wc1_renderer_{os.getpid()}"
    sys.path.insert(0, str(renderer_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, renderer_dir / "inflate.py")
        if spec is None or spec.loader is None:
            raise WC1AdvisoryError(f"cannot load renderer from {renderer_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _worker_initialize(config: dict[str, Any]) -> None:
    for name in THREAD_ENV_NAMES:
        os.environ[name] = str(config["per_process_threads"])
    import torch

    torch.set_num_threads(int(config["per_process_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    renderer = _load_renderer(Path(config["renderer_dir"]))
    semantic = renderer.SemanticTokenRenderer(int(config["semantic_width"])).eval()
    semantic.load_state_dict(
        {
            name: torch.from_numpy(np.asarray(value).copy())
            for name, value in config["semantic_state"].items()
        },
        strict=True,
    )
    device = torch.device("cpu")
    semantic = semantic.to(device)
    raw_basis = torch.from_numpy(np.asarray(config["basis"]).copy()).to(device)
    basis = renderer.normalized_basis(raw_basis)
    coefficients = torch.from_numpy(np.asarray(config["coefficients"]).copy()).to(device)
    tokens = np.memmap(
        config["token_path"],
        mode="r",
        dtype=np.uint8,
        shape=(int(config["token_frames"]), int(renderer.EVAL_H), int(renderer.EVAL_W)),
    )
    output = np.memmap(
        config["output_path"],
        mode="r+",
        dtype=np.uint8,
        shape=(int(config["pair_count"]) * 2, int(renderer.CAMERA_H), int(renderer.CAMERA_W), 3),
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "torch": torch,
            "renderer": renderer,
            "semantic": semantic,
            "basis": basis,
            "coefficients": coefficients,
            "tokens": tokens,
            "output": output,
            "device": device,
        }
    )


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _render_chunk(bounds: tuple[int, int]) -> dict[str, Any]:
    start, end = bounds
    state = _WORKER_STATE
    torch = state["torch"]
    renderer = state["renderer"]
    semantic = state["semantic"]
    basis = state["basis"]
    coefficients = state["coefficients"]
    tokens = state["tokens"]
    output = state["output"]
    device = state["device"]
    master_seconds = 0.0
    carrier_seconds = 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for pair_index in range(start, end):
            tick = time.perf_counter()
            indices = torch.tensor([pair_index], device=device)
            token = torch.from_numpy(np.asarray(tokens[pair_index : pair_index + 1])).long().to(device)
            master = (
                torch.nn.functional.interpolate(
                    semantic(token, indices),
                    size=(renderer.CAMERA_H, renderer.CAMERA_W),
                    mode="bilinear",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
            )
            output[2 * pair_index + 1] = (
                master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            )
            master_seconds += time.perf_counter() - tick
        for pair_index in range(start, end):
            tick = time.perf_counter()
            carrier = torch.einsum(
                "bk,kchw->bchw", coefficients[pair_index : pair_index + 1], basis
            )
            carrier = carrier / math.sqrt(renderer.CARRIER_DIM)
            slave = (
                torch.nn.functional.interpolate(
                    (127.5 + renderer.CARRIER_AMPLITUDE * carrier)
                    .clamp(0.0, 255.0)
                    .round(),
                    size=(renderer.CAMERA_H, renderer.CAMERA_W),
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
            )
            output[2 * pair_index] = (
                slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            )
            carrier_seconds += time.perf_counter() - tick
    output.flush()
    return {
        "start_pair": start,
        "end_pair": end,
        "pairs": end - start,
        "worker_pid": os.getpid(),
        "worker_peak_rss_bytes": _max_rss_bytes(),
        "worker_torch_threads": torch.get_num_threads(),
        "worker_torch_interop_threads": torch.get_num_interop_threads(),
        "master_seconds": master_seconds,
        "carrier_seconds": carrier_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _sha256_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                raise WC1AdvisoryError(f"render payload ended inside byte range at {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _array_fact(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "bytes": int(array.nbytes),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def render_video_parallel(
    *,
    semantic: Any,
    basis: Any,
    coefficients: Any,
    token_path: Path,
    token_sha256: str,
    renderer_dir: Path,
    output_path: Path,
    progress_dir: Path,
    pair_count: int,
    camera_height: int,
    camera_width: int,
    requested_workers: str,
    per_process_threads: int,
    measured_worker_rss_bytes: int | None,
) -> dict[str, Any]:
    """Render disjoint canonical pair ranges with crash-resumable chunk receipts."""

    renderer_dir = renderer_dir.resolve()
    token_path = token_path.resolve()
    output_path = output_path.resolve()
    progress_dir = progress_dir.resolve()
    if pair_count < 1:
        raise WC1AdvisoryError("pair_count must be positive")
    resources = resolve_render_workers(
        requested_workers,
        per_process_threads=per_process_threads,
        measured_worker_rss_bytes=measured_worker_rss_bytes,
    )
    workers = int(resources["selected"])
    semantic_state = {
        name: value.detach().cpu().contiguous().numpy()
        for name, value in semantic.state_dict().items()
    }
    basis_array = basis.detach().cpu().contiguous().numpy()
    coefficients_array = coefficients.detach().cpu().contiguous().numpy()
    renderer_source = renderer_dir / "inflate.py"
    if camera_height < 1 or camera_width < 1:
        raise WC1AdvisoryError("camera geometry must be positive")
    frame_bytes = camera_height * camera_width * 3
    expected_bytes = pair_count * 2 * frame_bytes
    chunk_pairs = max(1, math.ceil(pair_count / max(1, workers * 4)))
    chunks = [
        (start, min(start + chunk_pairs, pair_count))
        for start in range(0, pair_count, chunk_pairs)
    ]
    binding = {
        "schema": RENDER_PROGRESS_SCHEMA,
        "pair_count": pair_count,
        "token": {
            "path": str(token_path),
            "bytes": token_path.stat().st_size,
            "sha256": token_sha256,
        },
        "semantic_state": {
            name: _array_fact(value) for name, value in sorted(semantic_state.items())
        },
        "basis": _array_fact(basis_array),
        "coefficients": _array_fact(coefficients_array),
        "renderer_source": file_fact(renderer_source),
        "advisory_source": file_fact(Path(__file__).resolve()),
        "instrument": {
            "workers": workers,
            "per_process_threads": per_process_threads,
            "interop_threads": 1,
            "chunk_pairs": chunk_pairs,
            "start_method": "spawn",
            "camera_height": camera_height,
            "camera_width": camera_width,
        },
        "output": {"path": str(output_path), "bytes": expected_bytes},
    }
    binding_path = progress_dir / "binding.json"
    chunks_dir = progress_dir / "chunks"
    if progress_dir.exists():
        if not binding_path.is_file() or not output_path.is_file():
            raise WC1AdvisoryError("parallel-render progress is incomplete")
        existing = _canonical_manifest(binding_path)
        if existing != binding:
            raise WC1AdvisoryError("parallel-render resume binding differs")
        if output_path.stat().st_size != expected_bytes:
            raise WC1AdvisoryError("parallel-render checkpoint has the wrong byte count")
    else:
        progress_dir.mkdir(parents=True)
        chunks_dir.mkdir()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.memmap(output_path, mode="w+", dtype=np.uint8, shape=(expected_bytes,))
        output.flush()
        del output
        atomic_json(binding_path, binding)

    pending: list[tuple[int, int]] = []
    completed_rows: list[dict[str, Any]] = []
    for start, end in chunks:
        receipt_path = chunks_dir / f"pairs_{start:04d}_{end:04d}.json"
        offset = 2 * start * frame_bytes
        length = 2 * (end - start) * frame_bytes
        if receipt_path.is_file():
            receipt = _canonical_manifest(receipt_path)
            if (
                receipt.get("start_pair") != start
                or receipt.get("end_pair") != end
                or receipt.get("output_range_sha256")
                != _sha256_range(output_path, offset, length)
            ):
                raise WC1AdvisoryError(f"parallel-render chunk receipt differs: {receipt_path}")
            completed_rows.append(receipt)
        else:
            pending.append((start, end))

    config = {
        "renderer_dir": str(renderer_dir),
        "semantic_width": int(semantic.token_embed.embedding_dim),
        "semantic_state": semantic_state,
        "basis": basis_array,
        "coefficients": coefficients_array,
        "token_path": str(token_path),
        "token_frames": pair_count,
        "output_path": str(output_path),
        "pair_count": pair_count,
        "per_process_threads": per_process_threads,
    }
    started = time.perf_counter()
    if pending:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_worker_initialize,
            initargs=(config,),
        ) as pool:
            for row in pool.imap_unordered(_render_chunk, pending):
                start = int(row["start_pair"])
                end = int(row["end_pair"])
                offset = 2 * start * frame_bytes
                length = 2 * (end - start) * frame_bytes
                receipt = {
                    **row,
                    "schema": "ddm_wc1_parallel_render_chunk.v1",
                    "output_range_offset": offset,
                    "output_range_bytes": length,
                    "output_range_sha256": _sha256_range(output_path, offset, length),
                }
                atomic_json(chunks_dir / f"pairs_{start:04d}_{end:04d}.json", receipt)
                completed_rows.append(receipt)
    completed_rows.sort(key=lambda row: int(row["start_pair"]))
    if [(row["start_pair"], row["end_pair"]) for row in completed_rows] != chunks:
        raise WC1AdvisoryError("parallel-render completion coverage is not canonical")
    report = {
        "schema": RENDER_COMPLETE_SCHEMA,
        "complete": True,
        "axis_label": "[M5-CPU scorer-free advisory render]",
        "resources": resources,
        "binding": binding,
        "pair_count": pair_count,
        "chunk_count": len(chunks),
        "resumed_chunk_count": len(chunks) - len(pending),
        "fresh_chunk_count": len(pending),
        "invocation_seconds": time.perf_counter() - started,
        "render_stage": file_fact(output_path),
        "worker_peak_rss_bytes": {
            str(row["worker_pid"]): max(
                int(item["worker_peak_rss_bytes"])
                for item in completed_rows
                if item["worker_pid"] == row["worker_pid"]
            )
            for row in completed_rows
        },
        "chunks": completed_rows,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    atomic_json(progress_dir / "complete.json", report)
    return report


__all__ = [
    "WC1AdvisoryError",
    "file_fact",
    "load_token_cache",
    "publish_token_cache",
    "render_video_parallel",
    "resolve_render_workers",
    "sha256_file",
    "token_cache_key",
]
