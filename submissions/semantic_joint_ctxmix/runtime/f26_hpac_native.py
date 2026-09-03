"""Python binding and resumable full-field driver for the native F26 decoder."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class F26NativeError(RuntimeError):
    """The native HPAC/RC64 path rejected its archive-derived inputs."""


class _Rc64State(ctypes.Structure):
    _fields_ = [
        ("low", ctypes.c_uint64),
        ("high", ctypes.c_uint64),
        ("code", ctypes.c_uint64),
        ("bit_position", ctypes.c_uint64),
        ("error", ctypes.c_int32),
    ]


_MODEL_SCALARS = (
    "height",
    "width",
    "patch",
    "patch_rows",
    "patch_cols",
    "patch_count",
    "channels",
    "groups",
    "a_taps",
    "b1_taps",
    "b2_taps",
    "logit_precision",
    "num_frames",
    "frame_dim",
    "has_frame_scale",
    "has_spm",
)
_MODEL_POINTERS = (
    "conv_a_weight",
    "conv_a_delta",
    "conv_a_initial",
    "conv_a_bias",
    "conv_a_exponent",
    "b1_weight",
    "b1_bias",
    "b1_exponent",
    "b2_weight",
    "b2_bias",
    "b2_exponent",
    "head_weight",
    "head_bias",
    "head_exponent",
    "residual_table",
    "frame_codes",
    "frame_shift_weight",
    "frame_shift_bias",
    "frame_shift_exponent",
    "frame_scale_weight",
    "frame_scale_bias",
    "frame_scale_exponent",
    "conv_past_weight",
    "conv_past_bias",
    "conv_past_exponent",
    "spm_dw_weight",
    "spm_dw_bias",
    "spm_dw_exponent",
    "spm_pw_weight",
    "spm_pw_bias",
    "spm_pw_exponent",
    "a_offsets",
    "group_h_offsets",
    "group_b1_offsets",
    "group_target_offsets",
    "group_output_offsets",
    "h_positions",
    "b1_gather",
    "b2_gather",
    "target_positions",
    "output_order",
    "flat_positions",
)


class _NativeModel(ctypes.Structure):
    _fields_ = [
        *((name, ctypes.c_int32) for name in _MODEL_SCALARS),
        *((name, ctypes.c_void_p) for name in _MODEL_POINTERS),
    ]


@dataclass(frozen=True)
class _ModelBuffers:
    native: _NativeModel
    arrays: dict[str, np.ndarray]
    manifest_sha256: str


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


def _buffer_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _integer_array(value: Any, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if not np.all(np.isfinite(array)) or not np.array_equal(array, np.rint(array)):
        raise F26NativeError(f"{name} is not an exact integer array")
    info = np.iinfo(dtype)
    if array.size and (array.min() < info.min or array.max() > info.max):
        raise F26NativeError(f"{name} exceeds {dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


def _module_codes(module: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight, bias, exponent = module.codes()
    if exponent is None:
        raise F26NativeError(f"{name} has no weight exponent")
    return (
        _integer_array(weight, np.dtype(np.int8), f"{name}.weight"),
        _integer_array(bias, np.dtype("<i2"), f"{name}.bias"),
        _integer_array(exponent, np.dtype(np.int8), f"{name}.exponent"),
    )


def _offsets(lengths: list[int]) -> np.ndarray:
    return np.asarray([0, *np.cumsum(lengths, dtype=np.int64)], dtype=np.int32)


def _build_model_buffers(parts: Any, runtime: Any, sparse: Any) -> _ModelBuffers:
    cache = sparse._cpr1_sparse_cache
    model = sparse.model
    channels = int(model.ch)
    if channels != 64 or int(model.num_classes) != 5:
        raise F26NativeError("native lowering is sealed to the 64x5 F26 model shape")

    conv_a_dense = _integer_array(
        cache.conv_a_weight, np.dtype(np.int8), "conv_a.active_weight"
    ).reshape(channels, model.num_classes + 2, len(sparse.a_offsets))
    conv_a_weight = np.ascontiguousarray(conv_a_dense.transpose(1, 2, 0))
    conv_a_delta = np.ascontiguousarray(
        conv_a_weight[: model.num_classes].astype(np.int16)
        - conv_a_weight[0:1].astype(np.int16),
        dtype=np.int16,
    )
    coordinate_rows = []
    for plan in cache.conv_a_plans:
        coordinate = _integer_array(
            plan.coordinates[0], np.dtype(np.int16), "conv_a.coordinate_plan"
        )
        coordinate_rows.append(
            np.einsum(
                "hpt,cpt->hc",
                coordinate.astype(np.int32),
                conv_a_dense[:, model.num_classes :, :].astype(np.int32),
                dtype=np.int32,
                optimize=True,
            )
        )
    coordinate_by_position = np.zeros((sparse.patch * sparse.patch, channels), dtype=np.int32)
    coordinate_seen = np.zeros(sparse.patch * sparse.patch, dtype=np.bool_)
    for plan, coordinate in zip(sparse.plans, coordinate_rows, strict=True):
        positions = plan.h_positions.cpu().numpy()
        for index, (row, column) in enumerate(positions):
            flat = int(row) * sparse.patch + int(column)
            if coordinate_seen[flat]:
                if not np.array_equal(coordinate_by_position[flat], coordinate[index]):
                    raise F26NativeError("coordinate contribution differs for a repeated position")
            else:
                coordinate_by_position[flat] = coordinate[index]
                coordinate_seen[flat] = True
    if not np.all(coordinate_seen):
        raise F26NativeError("sparse group plans do not cover every patch position")
    conv_a_initial = coordinate_by_position.copy()
    for row in range(sparse.patch):
        for column in range(sparse.patch):
            flat = row * sparse.patch + column
            for tap, (row_offset, column_offset) in enumerate(sparse.a_offsets):
                source_row = row + int(row_offset)
                source_column = column + int(column_offset)
                if 0 <= source_row < sparse.patch and 0 <= source_column < sparse.patch:
                    conv_a_initial[flat] += conv_a_dense[:, 0, tap].astype(np.int32)
    conv_a_initial = np.ascontiguousarray(conv_a_initial, dtype=np.int32)

    conv_a = _module_codes(model.conv_a, "conv_a")
    b1 = _module_codes(model.conv_b1, "conv_b1")
    b2 = _module_codes(model.conv_b2, "conv_b2")
    head = _module_codes(model.head, "head")
    frame_shift = _module_codes(model.frame_shift, "frame_shift")
    if not model.use_frame_scale or not model.use_spm:
        raise F26NativeError("the sealed F26 context path requires frame-scale and SPM")
    frame_scale = _module_codes(model.frame_scale, "frame_scale")
    conv_past = _module_codes(model.conv_past, "conv_past")
    spm_dw = _module_codes(model.spm_dw, "spm_dw")
    spm_pw = _module_codes(model.spm_pw, "spm_pw")
    b1_weight = _integer_array(
        cache.depthwise_weights[id(model.conv_b1)][0, 0],
        np.dtype(np.int8),
        "conv_b1.active_weight",
    )
    b2_weight = _integer_array(
        cache.depthwise_weights[id(model.conv_b2)][0, 0],
        np.dtype(np.int8),
        "conv_b2.active_weight",
    )
    head_weight = np.ascontiguousarray(head[0][:, :, 0, 0], dtype=np.int8)

    h_lengths = [int(plan.h_positions.shape[0]) for plan in sparse.plans]
    b1_lengths = [int(plan.b1_gather.shape[0]) for plan in sparse.plans]
    target_lengths = [int(plan.b2_gather.shape[0]) for plan in sparse.plans]
    output_lengths = [int(plan.output_order.shape[0]) for plan in sparse.plans]
    h_positions = np.ascontiguousarray(
        np.concatenate([plan.h_positions.cpu().numpy() for plan in sparse.plans]),
        dtype=np.int16,
    )
    b1_gather = np.ascontiguousarray(
        np.concatenate([plan.b1_gather.cpu().numpy() for plan in sparse.plans]),
        dtype=np.int16,
    )
    b2_gather = np.ascontiguousarray(
        np.concatenate([plan.b2_gather.cpu().numpy() for plan in sparse.plans]),
        dtype=np.int16,
    )
    target_positions = np.ascontiguousarray(
        np.concatenate([plan.targets.cpu().numpy() for plan in sparse.plans]),
        dtype=np.int16,
    )
    output_order = np.ascontiguousarray(
        np.concatenate([plan.output_order.cpu().numpy() for plan in sparse.plans]),
        dtype=np.int32,
    )

    flat_positions = []
    for mask in runtime.group_masks(next(model.parameters()).device):
        flat_positions.append(np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)))
    flat_positions_array = np.ascontiguousarray(np.concatenate(flat_positions), dtype=np.int32)
    if flat_positions_array.size != int(runtime.EVAL_H * runtime.EVAL_W):
        raise F26NativeError("group plans do not cover exactly one token frame")

    arrays = {
        "conv_a_weight": conv_a_weight,
        "conv_a_delta": conv_a_delta,
        "conv_a_initial": conv_a_initial,
        "conv_a_bias": conv_a[1],
        "conv_a_exponent": conv_a[2],
        "b1_weight": b1_weight,
        "b1_bias": b1[1],
        "b1_exponent": b1[2],
        "b2_weight": b2_weight,
        "b2_bias": b2[1],
        "b2_exponent": b2[2],
        "head_weight": head_weight,
        "head_bias": head[1],
        "head_exponent": head[2],
        "residual_table": np.ascontiguousarray(parts.table.values, dtype=np.float32),
        "frame_codes": _integer_array(
            model.frame_codes(), np.dtype(np.int8), "frame_codes"
        ),
        "frame_shift_weight": np.ascontiguousarray(frame_shift[0], dtype=np.int8),
        "frame_shift_bias": frame_shift[1],
        "frame_shift_exponent": frame_shift[2],
        "frame_scale_weight": np.ascontiguousarray(frame_scale[0], dtype=np.int8),
        "frame_scale_bias": frame_scale[1],
        "frame_scale_exponent": frame_scale[2],
        "conv_past_weight": np.ascontiguousarray(
            conv_past[0].transpose(2, 3, 1, 0), dtype=np.int8
        ),
        "conv_past_bias": conv_past[1],
        "conv_past_exponent": conv_past[2],
        "spm_dw_weight": np.ascontiguousarray(
            spm_dw[0][:, 0].transpose(1, 2, 0), dtype=np.int8
        ),
        "spm_dw_bias": spm_dw[1],
        "spm_dw_exponent": spm_dw[2],
        "spm_pw_weight": np.ascontiguousarray(
            spm_pw[0][:, :, 0, 0], dtype=np.int8
        ),
        "spm_pw_bias": spm_pw[1],
        "spm_pw_exponent": spm_pw[2],
        "a_offsets": np.ascontiguousarray(sparse.a_offsets, dtype=np.int16),
        "group_h_offsets": _offsets(h_lengths),
        "group_b1_offsets": _offsets(b1_lengths),
        "group_target_offsets": _offsets(target_lengths),
        "group_output_offsets": _offsets(output_lengths),
        "h_positions": h_positions,
        "b1_gather": b1_gather,
        "b2_gather": b2_gather,
        "target_positions": target_positions,
        "output_order": output_order,
        "flat_positions": flat_positions_array,
    }
    manifest = [
        {
            "name": name,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "bytes": int(array.nbytes),
            "sha256": _buffer_sha256(array),
        }
        for name, array in sorted(arrays.items())
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    scalars = {
        "height": int(runtime.EVAL_H),
        "width": int(runtime.EVAL_W),
        "patch": int(sparse.patch),
        "patch_rows": int(sparse.patch_rows),
        "patch_cols": int(sparse.patch_cols),
        "patch_count": int(sparse.patch_count),
        "channels": channels,
        "groups": len(sparse.plans),
        "a_taps": len(sparse.a_offsets),
        "b1_taps": int(b1_weight.shape[0]),
        "b2_taps": int(b2_weight.shape[0]),
        "logit_precision": int(runtime.HPAC_LOGIT_PRECISION),
        "num_frames": int(model.num_pairs),
        "frame_dim": int(model.frame_embed.weight.shape[1]),
        "has_frame_scale": int(model.use_frame_scale),
        "has_spm": int(model.use_spm),
    }
    native = _NativeModel(
        **scalars,
        **{name: array.ctypes.data for name, array in arrays.items()},
    )
    return _ModelBuffers(native=native, arrays=arrays, manifest_sha256=manifest_sha256)


def _load_library(path: Path) -> Any:
    library = ctypes.CDLL(str(path.resolve()))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    library.f26_rc64_create.argtypes = [u8p, ctypes.c_size_t]
    library.f26_rc64_create.restype = ctypes.c_void_p
    library.f26_rc64_destroy.argtypes = [ctypes.c_void_p]
    library.f26_rc64_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rc64State)]
    library.f26_rc64_get_state.restype = ctypes.c_int
    library.f26_rc64_set_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rc64State)]
    library.f26_rc64_set_state.restype = ctypes.c_int
    library.f26_rc64_bit_position.argtypes = [ctypes.c_void_p]
    library.f26_rc64_bit_position.restype = ctypes.c_size_t
    library.f26_hpac_decode_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeModel),
        u8p,
        u8p,
        ctypes.c_int32,
        u8p,
        f32p,
        f32p,
    ]
    library.f26_hpac_decode_frame.restype = ctypes.c_int
    library.f26_hpac_last_timing.argtypes = [ctypes.POINTER(ctypes.c_double)]
    library.f26_total_frequency.restype = ctypes.c_uint64
    if library.f26_total_frequency() != 1 << 31:
        raise F26NativeError("native library has an incompatible RC64 frequency total")
    return library


def _pointer(array: np.ndarray, ctype: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _state_dict(state: _Rc64State) -> dict[str, int]:
    return {name: int(getattr(state, name)) for name, _ in state._fields_}


def _state_from_dict(value: dict[str, Any]) -> _Rc64State:
    return _Rc64State(**{name: int(value[name]) for name, _ in _Rc64State._fields_})


def decode_native_tokens(
    parts: Any,
    runtime: Any,
    code_dir: Path,
    device: Any,
    *,
    frame_limit: int | None = None,
    output_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[Any, dict[str, object]]:
    """Decode a real F26 prefix or full field and retain resumable token bytes."""
    import torch
    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import _boundary_buckets, _sparse_class

    if device.type != "cpu":
        raise F26NativeError("the native F26 path is CPU-only")
    library_path_text = os.environ.get("F26_HPAC_NATIVE_LIBRARY")
    if not library_path_text:
        raise F26NativeError("F26_HPAC_NATIVE_LIBRARY is required")
    library_path = Path(library_path_text).resolve()
    if not library_path.is_file():
        raise F26NativeError(f"native library does not exist: {library_path}")
    total_frames = int(runtime.N if frame_limit is None else frame_limit)
    if total_frames <= 0 or total_frames > int(runtime.N):
        raise F26NativeError("frame_limit is outside the real n600 field")

    started = time.perf_counter()
    base_hpac = materialize_ihs1(parts.hpac_blob, runtime)
    model = runtime.load_hpac(base_hpac, device)
    sparse = _sparse_class(code_dir)(model, runtime.EVAL_H, runtime.EVAL_W)
    optimize_sparse_evaluator(sparse)
    buffers = _build_model_buffers(parts, runtime, sparse)
    setup_seconds = time.perf_counter() - started

    if output_path is None:
        output_path_text = os.environ.get("F26_NATIVE_TOKEN_OUTPUT")
        if not output_path_text:
            raise F26NativeError("native decode requires a durable token output path")
        output_path = Path(output_path_text)
    output_path = output_path.resolve()
    if checkpoint_dir is None:
        checkpoint_text = os.environ.get("F26_NATIVE_CHECKPOINT_DIR")
        checkpoint_dir = output_path.parent / "checkpoints" if not checkpoint_text else Path(checkpoint_text)
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    library = _load_library(library_path)
    expected_bytes = total_frames * int(runtime.EVAL_H) * int(runtime.EVAL_W)
    latest_path = checkpoint_dir / "LATEST.json"
    binding = {
        "schema": "ddm_f26q_native_binding.v1",
        "token_stream_sha256": hashlib.sha256(parts.token_stream).hexdigest(),
        "hpac_blob_sha256": hashlib.sha256(parts.hpac_blob).hexdigest(),
        "residual_payload_sha256": hashlib.sha256(parts.residual_payload).hexdigest(),
        "model_manifest_sha256": buffers.manifest_sha256,
        "native_binary_sha256": _sha256_file(library_path),
        "frames": total_frames,
        "height": int(runtime.EVAL_H),
        "width": int(runtime.EVAL_W),
    }
    start_frame = 0
    resume_state = None
    created_new_output = False
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if latest.get("binding") != binding:
            raise F26NativeError("native checkpoint binding differs from this decode")
        if not output_path.is_file() or output_path.stat().st_size != expected_bytes:
            raise F26NativeError("native checkpoint token field is missing or has wrong size")
        prefix_bytes = int(latest["next_frame"]) * int(runtime.EVAL_H) * int(runtime.EVAL_W)
        with output_path.open("rb") as stream:
            prefix_sha = hashlib.sha256(stream.read(prefix_bytes)).hexdigest()
        if prefix_sha != latest["token_prefix_sha256"]:
            raise F26NativeError("native checkpoint token prefix SHA-256 differs")
        start_frame = int(latest["next_frame"])
        resume_state = _state_from_dict(latest["rc64_state"])
        tokens = np.memmap(
            output_path,
            mode="r+",
            dtype=np.uint8,
            shape=(total_frames, runtime.EVAL_H, runtime.EVAL_W),
        )
    else:
        if output_path.exists():
            raise F26NativeError("native token output exists without a checkpoint receipt")
        tokens = np.memmap(
            output_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total_frames, runtime.EVAL_H, runtime.EVAL_W),
        )
        created_new_output = True

    payload = np.frombuffer(parts.token_stream, dtype=np.uint8).copy()
    decoder = library.f26_rc64_create(_pointer(payload, ctypes.c_uint8), payload.size)
    if not decoder:
        raise F26NativeError("native RC64 decoder allocation failed")
    if resume_state is not None and library.f26_rc64_set_state(decoder, ctypes.byref(resume_state)):
        library.f26_rc64_destroy(decoder)
        raise F26NativeError("native RC64 checkpoint restore failed")
    if created_new_output:
        initial_state = _Rc64State()
        if library.f26_rc64_get_state(decoder, ctypes.byref(initial_state)):
            library.f26_rc64_destroy(decoder)
            raise F26NativeError("native initial RC64 state capture failed")
        initial_checkpoint = {
            "schema": "ddm_f26q_native_checkpoint.v1",
            "complete": False,
            "binding": binding,
            "next_frame": 0,
            "token_prefix_bytes": 0,
            "token_prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "rc64_state": _state_dict(initial_state),
            "rolling_token_path": str(output_path),
        }
        _atomic_json(checkpoint_dir / "initial.json", initial_checkpoint)
        _atomic_json(latest_path, initial_checkpoint)

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    digest_scope = "full_field" if start_frame == 0 else "suffix_after_resume"
    context_seconds = 0.0
    boundary_seconds = 0.0
    native_seconds = 0.0
    digest_seconds = 0.0
    checkpoint_seconds = 0.0
    native_element_seconds = np.zeros(5, dtype=np.float64)
    trace_corrected = np.empty((runtime.EVAL_H * runtime.EVAL_W, 5), dtype=np.float32)
    trace_probability = np.empty_like(trace_corrected)
    zero_previous = np.zeros(runtime.EVAL_H * runtime.EVAL_W, dtype=np.uint8)
    try:
        for frame in range(start_frame, total_frames):
            context_started = time.perf_counter()
            if frame:
                previous = np.ascontiguousarray(
                    np.asarray(tokens[frame - 1]).reshape(-1), dtype=np.uint8
                )
            else:
                previous = zero_previous
            context_seconds += time.perf_counter() - context_started

            boundary_started = time.perf_counter()
            if frame:
                boundary = _boundary_buckets(np.asarray(tokens[frame - 1])).reshape(-1)
            else:
                boundary = np.full(runtime.EVAL_H * runtime.EVAL_W, 4, dtype=np.uint8)
            boundary = np.ascontiguousarray(boundary, dtype=np.uint8)
            boundary_seconds += time.perf_counter() - boundary_started

            current = np.asarray(tokens[frame]).reshape(-1)
            native_started = time.perf_counter()
            status = library.f26_hpac_decode_frame(
                decoder,
                ctypes.byref(buffers.native),
                _pointer(boundary, ctypes.c_uint8),
                _pointer(previous, ctypes.c_uint8),
                frame,
                _pointer(current, ctypes.c_uint8),
                _pointer(trace_corrected, ctypes.c_float),
                _pointer(trace_probability, ctypes.c_float),
            )
            native_seconds += time.perf_counter() - native_started
            if status:
                raise F26NativeError(f"native frame {frame} failed with status {status}")
            frame_element_seconds = np.empty(5, dtype=np.float64)
            library.f26_hpac_last_timing(_pointer(frame_element_seconds, ctypes.c_double))
            native_element_seconds += frame_element_seconds

            digest_started = time.perf_counter()
            corrected_digest.update(memoryview(trace_corrected).cast("B"))
            cdf_digest.update(memoryview(trace_probability).cast("B"))
            digest_seconds += time.perf_counter() - digest_started

            if (frame + 1) % 25 == 0 or frame + 1 == total_frames:
                checkpoint_started = time.perf_counter()
                tokens.flush()
                state = _Rc64State()
                if library.f26_rc64_get_state(decoder, ctypes.byref(state)):
                    raise F26NativeError("native RC64 state capture failed")
                prefix = np.asarray(tokens[: frame + 1])
                checkpoint = {
                    "schema": "ddm_f26q_native_checkpoint.v1",
                    "complete": frame + 1 == total_frames,
                    "binding": binding,
                    "next_frame": frame + 1,
                    "token_prefix_bytes": int(prefix.nbytes),
                    "token_prefix_sha256": _buffer_sha256(prefix),
                    "rc64_state": _state_dict(state),
                    "rolling_token_path": str(output_path),
                }
                stage_path = checkpoint_dir / f"through_frame_{frame:03d}.json"
                if stage_path.exists():
                    observed = json.loads(stage_path.read_text(encoding="utf-8"))
                    if observed != checkpoint:
                        raise F26NativeError(f"refusing to overwrite differing checkpoint {stage_path}")
                else:
                    _atomic_json(stage_path, checkpoint)
                _atomic_json(latest_path, checkpoint)
                checkpoint_seconds += time.perf_counter() - checkpoint_started
    finally:
        bit_position = int(library.f26_rc64_bit_position(decoder))
        library.f26_rc64_destroy(decoder)

    finalization_started = time.perf_counter()
    tokens.flush()
    token_sha256 = _sha256_file(output_path)
    finalization_seconds = time.perf_counter() - finalization_started
    elapsed = time.perf_counter() - started
    report = {
        "schema": "ddm_f26q_native_token_report.v1",
        "implementation": "native_c_direct_int16_context_delta_hpac_probability_rc64",
        "frames": total_frames,
        "resumed_from_frame": start_frame,
        "decoded_token_sha256": token_sha256,
        "decoded_token_bytes": output_path.stat().st_size,
        "decoded_token_path": str(output_path),
        "decoder_bit_position": bit_position,
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "digest_scope": digest_scope,
        "decode_runtime_seconds": elapsed,
        "stage_seconds": {
            "setup": setup_seconds,
            "frame_context_python_pointer_prep": context_seconds,
            "boundary_buckets": boundary_seconds,
            "native_fused_hpac_probability_rc64": native_seconds,
            "native_frame_context_int16": float(native_element_seconds[0]),
            "native_conv_state_initialization": float(native_element_seconds[1]),
            "native_sparse_hidden_and_logits": float(native_element_seconds[2]),
            "native_probability_and_rc64": float(native_element_seconds[3]),
            "native_incremental_conv_update": float(native_element_seconds[4]),
            "digest_updates": digest_seconds,
            "checkpoint_persistence": checkpoint_seconds,
            "final_token_flush_and_sha256": finalization_seconds,
        },
        "native_binary": {
            "path": str(library_path),
            "bytes": library_path.stat().st_size,
            "sha256": _sha256_file(library_path),
        },
        "model_manifest_sha256": buffers.manifest_sha256,
        "token_codec": "rc64",
        "checkpoint_dir": str(checkpoint_dir),
    }
    return torch.from_numpy(np.asarray(tokens)), report


def decode_production_tokens_native(
    parts: Any, runtime: Any, code_dir: Path, device: Any
) -> tuple[Any, dict[str, object]]:
    """Runtime entry point; full n600 only, with durable progress required."""
    tokens, report = decode_native_tokens(parts, runtime, code_dir, device)
    if int(report["frames"]) != int(runtime.N):
        raise F26NativeError("production native path did not decode the full n600 field")
    return tokens, report
