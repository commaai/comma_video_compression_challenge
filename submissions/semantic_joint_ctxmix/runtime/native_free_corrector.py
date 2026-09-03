# SPDX-License-Identifier: MIT
"""ddm_rr8 - ctypes binding for the C free corrector, with a config-drift refusal.

The C library reproduces ``free_corrector.FreeCorrector`` under the frozen
``SHIPPED_CONFIG`` ONLY.  It compiles that configuration in as constants -- 13 named
members, a 4,000-cell mixer context, ``count_buckets=1``, ``power_bits=6``, the SSE stage
OFF -- because a decoder takes no arguments and a corrector that silently accepted a
different config would desynchronise the arithmetic decoder rather than fail.

So the binding REFUSES unless the live Python config still matches the compiled one.  A
refusal here costs the speedup; a silent mismatch costs the submission.  ``ddm_rr2``'s
S = 27.83 is what a desynchronised decoder scores, and it looked like a model failure rather
than what it was.

The fallback is always the shipped Python corrector.  Nothing in this module is permitted to
change a decoded byte, which is what makes it a wall-clock change and not a re-pricing.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

__all__ = ["NativeCorrectorError", "NativeFreeCorrector", "load_native_corrector"]


class NativeCorrectorError(RuntimeError):
    """The native corrector is unavailable or its configuration has drifted."""


ABI_VERSION = 1
PLANE = 384 * 512
NUM_CLASSES = 5

#: The exact configuration compiled into ``f26_corrector_native.c``.  Every key is checked;
#: an ADDED key the C does not know about is also a refusal, because "the config grew a
#: field" is indistinguishable from "the C is now wrong" from in here.
EXPECTED_SHIPPED_CONFIG = {
    "families": (
        "shipped_joint",
        "temporal_spatial",
        "surprise_only",
        "spatial_surprise",
        "spatial_boundary",
        "run_surprise",
        "boundary_surprise",
        "temporal_surprise",
        "shipped_fast256",
        "shipped_fast4096",
        "surprise_fast256",
        "spatial4_surprise",
        "homog_surprise",
        # ddm_fx5: E1's six, compiled into f26_corrector_native.c alongside these.
        "homog_boundary_surprise",
        "spatial4_boundary",
        "homog_spatial4",
        "spatial4_temporal",
        "homog_surprise_fast256",
        "spatial4_surprise_fast256",
        # ddm_gb1: the decode-scan group-conditioning member.
        "groupbin8_surprise",
        "cls_groupbin8",
        "patch192_only",
        "tile48_groupbin8",
    ),
    "mixer_context": "cls_boundary_agree_homog_ubin8",
    "count_buckets": 1,
    "sse_context": "off",
    "sse_learn_weight": False,
    "normalize": True,
    "learn": True,
    "within_miss": True,
    "miss_cell": "nb3_prev1",
    "miss_min_count": 1,
    "miss_clamp": 16.0,
}

#: Module-level constants the C hard-codes.  These are NOT part of ``SHIPPED_CONFIG`` -- they
#: reach the mixer as argparse-style defaults -- so they need their own check or a change to
#: one of them would slip past the config comparison entirely.
EXPECTED_CONSTANTS = {
    "POWER_BITS": 6,
    "INT_POWER_BITS": 4,
    "WEIGHT_STORE_BITS": 20,
    "MIN_COUNT": 32,
    "U_BINS": 64,
    "RUN_LEVELS": 8,
    "BOUNDARY_LEVELS": 5,
    "NUM_CLASSES": 5,
    "SPATIAL_LEVELS": 5,
    "SPATIAL4_LEVELS": 6,
    "HOMOGENEITY_LEVELS": 5,
    "LR_SHIFT": 24,
    "GROUP_BINS": 8,
}


def _siblings():
    """Import the four corrector modules, in-tree or from a repo checkout.

    In the shipped tree this file sits inside the ``runtime`` package and the relative
    import is the right one.  The parity harness drives it from the repo, where there is no
    package parent -- and it must exercise THIS file rather than a patched twin, or the gate
    is testing something other than what ships.  So the absolute path is a fallback, not an
    alternative implementation.
    """
    try:
        from . import free_corrector, fx1_logistic_mixer_corrector, fx2_model_axis_corrector, rr4_free_corrector
    except ImportError:
        from runtime import (
            free_corrector,
            fx1_logistic_mixer_corrector,
            fx2_model_axis_corrector,
            rr4_free_corrector,
        )
    return (
        free_corrector,
        fx1_logistic_mixer_corrector,
        fx2_model_axis_corrector,
        rr4_free_corrector,
    )


def _live_constants() -> dict[str, int]:
    """Read the constants back out of the shipped modules, never from memory."""
    _, fx1, fx2, rr4 = _siblings()

    return {
        "POWER_BITS": int(fx1.POWER_BITS),
        "INT_POWER_BITS": int(fx1.INT_POWER_BITS),
        "WEIGHT_STORE_BITS": int(fx1.WEIGHT_STORE_BITS),
        "MIN_COUNT": int(rr4.MIN_COUNT),
        "U_BINS": int(rr4.U_BINS),
        "RUN_LEVELS": int(rr4.RUN_LEVELS),
        "BOUNDARY_LEVELS": int(rr4.BOUNDARY_LEVELS),
        "NUM_CLASSES": int(rr4.NUM_CLASSES),
        "SPATIAL_LEVELS": int(fx1.SPATIAL_LEVELS),
        "SPATIAL4_LEVELS": int(fx2.SPATIAL4_LEVELS),
        "HOMOGENEITY_LEVELS": int(fx2.HOMOGENEITY_LEVELS),
        "LR_SHIFT": int(fx1.LR_BASE_SHIFT) + 4,
        "GROUP_BINS": int(fx2.GROUP_BINS),
    }


def assert_config_matches() -> None:
    """Refuse if the live corrector configuration has drifted from the compiled one."""
    free_corrector = _siblings()[0]

    live = dict(free_corrector.SHIPPED_CONFIG)
    expected = dict(EXPECTED_SHIPPED_CONFIG)
    live["families"] = tuple(live.get("families", ()))
    if live != expected:
        differing = sorted(
            key
            for key in set(live) | set(expected)
            if live.get(key, "<missing>") != expected.get(key, "<missing>")
        )
        raise NativeCorrectorError(
            "SHIPPED_CONFIG has drifted from the configuration compiled into "
            f"f26_corrector_native.c; differing keys: {differing}"
        )

    live_constants = _live_constants()
    if live_constants != EXPECTED_CONSTANTS:
        differing = sorted(
            key
            for key in EXPECTED_CONSTANTS
            if live_constants.get(key) != EXPECTED_CONSTANTS[key]
        )
        raise NativeCorrectorError(
            f"corrector constants have drifted from the compiled ones: {differing}"
        )


def _bind(library: ctypes.CDLL) -> ctypes.CDLL:
    i64 = ctypes.POINTER(ctypes.c_int64)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    f32 = ctypes.POINTER(ctypes.c_float)

    library.f26_corrector_abi_version.argtypes = []
    library.f26_corrector_abi_version.restype = ctypes.c_int32
    library.f26_corrector_create.argtypes = [ctypes.c_int64]
    library.f26_corrector_create.restype = ctypes.c_void_p
    library.f26_corrector_destroy.argtypes = [ctypes.c_void_p]
    library.f26_corrector_destroy.restype = None
    library.f26_corrector_begin_frame.argtypes = [ctypes.c_void_p, i64, ctypes.c_int64]
    library.f26_corrector_begin_frame.restype = ctypes.c_int
    library.f26_corrector_group_state.argtypes = [
        ctypes.c_void_p,
        f32,
        i64,
        i64,
        ctypes.c_int64,
    ]
    library.f26_corrector_group_state.restype = ctypes.c_int
    library.f26_corrector_coding_row.argtypes = [ctypes.c_void_p, f32, ctypes.c_int64]
    library.f26_corrector_coding_row.restype = ctypes.c_int
    library.f26_corrector_observe.argtypes = [ctypes.c_void_p, i64, ctypes.c_int64]
    library.f26_corrector_observe.restype = ctypes.c_int
    library.f26_corrector_end_frame.argtypes = [ctypes.c_void_p, u8, ctypes.c_int64]
    library.f26_corrector_end_frame.restype = ctypes.c_int
    library.f26_corrector_table.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        i64,
        ctypes.c_int64,
        i64,
    ]
    library.f26_corrector_table.restype = ctypes.c_int

    version = int(library.f26_corrector_abi_version())
    if version != ABI_VERSION:
        raise NativeCorrectorError(
            f"native corrector ABI {version} does not match the expected {ABI_VERSION}"
        )
    return library


class _NativeGroupState:
    """Opaque per-group token.

    The real group state lives in C.  This object exists so the decode loop keeps its shipped
    shape (``state = group_state(...)``; ``coding_row(state)``; ``observe(state, symbols)``)
    and so a stale state cannot be replayed against a corrector that has moved on.
    """

    __slots__ = ("n", "owner", "serial")

    def __init__(self, owner: NativeFreeCorrector, serial: int, n: int) -> None:
        self.owner = owner
        self.serial = serial
        self.n = n


class NativeFreeCorrector:
    """Drop-in for ``free_corrector.FreeCorrector``, backed by the C library."""

    def __init__(self, plane: int, library_path: str | os.PathLike[str]) -> None:
        if plane != PLANE:
            raise NativeCorrectorError("the native corrector assumes the 384x512 plane")
        assert_config_matches()
        self.plane = int(plane)
        self.library = _bind(ctypes.CDLL(str(Path(library_path).resolve())))
        self.handle = self.library.f26_corrector_create(ctypes.c_int64(plane))
        if not self.handle:
            raise NativeCorrectorError(
                "native corrector allocation failed (plane mismatch, a non-conforming "
                "platform sqrt, or out of memory)"
            )
        self._serial = 0

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.library.f26_corrector_destroy(handle)
            self.handle = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        try:
            self.close()
        except Exception:
            pass

    # -- driving ------------------------------------------------------------

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        boundary = np.ascontiguousarray(
            np.asarray(boundary_flat, dtype=np.int64).reshape(-1)
        )
        if boundary.size != self.plane:
            raise ValueError("boundary bucket plane size mismatch")
        status = self.library.f26_corrector_begin_frame(
            self.handle,
            boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(boundary.size),
        )
        if status:
            raise NativeCorrectorError(f"begin_frame failed with status {status}")

    def group_state(
        self,
        probability: np.ndarray,
        predicted: np.ndarray,
        positions: np.ndarray,
    ) -> _NativeGroupState:
        # ``np.asarray(probability, dtype=np.float32)`` is what rr4 does before widening to
        # float64; ascontiguousarray only fixes the stride, never a value.
        rows = np.ascontiguousarray(probability, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != NUM_CLASSES:
            raise ValueError("probability rows must have shape [n, 5]")
        base_class = np.ascontiguousarray(
            np.asarray(predicted, dtype=np.int64).reshape(-1)
        )
        flat = np.ascontiguousarray(np.asarray(positions, dtype=np.int64).reshape(-1))
        n = rows.shape[0]
        if base_class.size != n or flat.size != n:
            raise ValueError("group predicted/positions length mismatch")

        status = self.library.f26_corrector_group_state(
            self.handle,
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            base_class.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(n),
        )
        if status:
            raise NativeCorrectorError(f"group_state failed with status {status}")
        self._serial += 1
        return _NativeGroupState(self, self._serial, n)

    def _check(self, state: _NativeGroupState) -> None:
        if state.owner is not self or state.serial != self._serial:
            raise NativeCorrectorError(
                "group state does not belong to the open group; the decode order changed"
            )

    def coding_row(self, state: _NativeGroupState) -> np.ndarray:
        self._check(state)
        output = np.empty((state.n, NUM_CLASSES), dtype=np.float32)
        status = self.library.f26_corrector_coding_row(
            self.handle,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(state.n),
        )
        if status:
            raise NativeCorrectorError(f"coding_row failed with status {status}")
        return output

    def observe(self, state: _NativeGroupState, symbols: np.ndarray) -> None:
        self._check(state)
        decoded = np.ascontiguousarray(np.asarray(symbols, dtype=np.int64).reshape(-1))
        if decoded.size != state.n:
            raise ValueError("decoded symbol count does not match the group")
        status = self.library.f26_corrector_observe(
            self.handle,
            decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(decoded.size),
        )
        if status:
            raise NativeCorrectorError(f"observe failed with status {status}")

    def end_frame(self, tokens_flat: np.ndarray) -> None:
        current = np.ascontiguousarray(
            np.asarray(tokens_flat, dtype=np.uint8).reshape(-1)
        )
        if current.size != self.plane:
            raise ValueError("token plane size mismatch")
        status = self.library.f26_corrector_end_frame(
            self.handle,
            current.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(current.size),
        )
        if status:
            raise NativeCorrectorError(f"end_frame failed with status {status}")

    # -- introspection, for the parity harness only -------------------------

    def table(self, which: int, position: int = 0) -> np.ndarray:
        """Copy out one live table.

        The differential harness compares STATE, not only output: a corrector whose tables
        have already diverged still emits an identical row wherever both sides are cold, so
        an output-only comparison can pass long after the run is lost.
        """
        size = ctypes.c_int64(0)
        status = self.library.f26_corrector_table(
            self.handle,
            ctypes.c_int32(which),
            ctypes.c_int32(position),
            None,
            ctypes.c_int64(0),
            ctypes.byref(size),
        )
        if status:
            raise NativeCorrectorError(f"table probe failed with status {status}")
        out = np.empty(size.value, dtype=np.int64)
        status = self.library.f26_corrector_table(
            self.handle,
            ctypes.c_int32(which),
            ctypes.c_int32(position),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(out.size),
            None,
        )
        if status:
            raise NativeCorrectorError(f"table read failed with status {status}")
        return out


def load_native_corrector(plane: int) -> NativeFreeCorrector | None:
    """Bind the native corrector, or return None so the caller falls back to Python.

    Fail-OPEN by design, mirroring ``ddm_rr6``'s build lesson: a hostile toolchain, a missing
    library or a drifted config must cost the SPEEDUP, never the submission.  The one
    exception is an explicitly-named-but-missing library, which is operator misconfiguration
    and a different class -- that raises.
    """
    named = os.environ.get("F26_CORRECTOR_NATIVE_LIBRARY")
    if not named:
        return None
    path = Path(named)
    if not path.is_file():
        raise NativeCorrectorError(f"named native corrector library is missing: {path}")
    try:
        return NativeFreeCorrector(plane, path)
    except NativeCorrectorError:
        raise
    except Exception as error:  # pragma: no cover - platform-specific loader failures
        raise NativeCorrectorError(f"native corrector failed to load: {error}") from error
