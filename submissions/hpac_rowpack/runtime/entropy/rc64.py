"""Python binding for the fixed F26 RC64 decoder."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

ALPHABET = 5
TOTAL = 1 << 31


class Rc64Error(ValueError):
    """The RC64 backend or probability stream is invalid."""


def _load_library(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    u8_pointer = ctypes.POINTER(ctypes.c_uint8)
    i32_pointer = ctypes.POINTER(ctypes.c_int32)
    f32_pointer = ctypes.POINTER(ctypes.c_float)
    library.rc64_decoder_create.argtypes = [u8_pointer, ctypes.c_size_t]
    library.rc64_decoder_create.restype = ctypes.c_void_p
    library.rc64_decoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_decode_probabilities.argtypes = [
        ctypes.c_void_p,
        f32_pointer,
        ctypes.c_size_t,
        i32_pointer,
    ]
    library.rc64_decoder_decode_probabilities.restype = ctypes.c_int
    library.rc64_decoder_bit_position.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_bit_position.restype = ctypes.c_size_t
    library.rc64_total_frequency.restype = ctypes.c_uint64
    if library.rc64_total_frequency() != TOTAL:
        raise Rc64Error("RC64 backend uses an incompatible frequency total")
    return library


class NativeDecoder:
    """Streaming wrapper around the compiled RC64 probability decoder."""

    def __init__(self, library_path: Path, payload: bytes) -> None:
        if not payload:
            raise Rc64Error("RC64 payload is empty")
        self.library = _load_library(library_path)
        self.payload = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        self.context = self.library.rc64_decoder_create(self.payload, len(payload))
        if not self.context:
            raise Rc64Error("RC64 decoder allocation failed")

    def close(self) -> None:
        if self.context:
            self.library.rc64_decoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        self.close()

    def decode(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ALPHABET or not values.size:
            raise Rc64Error("RC64 probabilities must have shape [N, 5]")
        output = np.empty(values.shape[0], dtype=np.int32)
        status = self.library.rc64_decoder_decode_probabilities(
            self.context,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.shape[0],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if status:
            raise Rc64Error(f"native RC64 decode failed with status {status}")
        return output

    @property
    def bit_position(self) -> int:
        return int(self.library.rc64_decoder_bit_position(self.context))
