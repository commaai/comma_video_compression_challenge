"""Restore CPR1 coefficients from the fixed-point CAP1 predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q8_MIN = -512
Q8_MAX = 512
BIAS_MIN = -16
BIAS_MAX = 16


class CoefficientPredictorError(ValueError):
    """An integer coefficient predictor input is invalid."""


def signed_mod(values: np.ndarray) -> np.ndarray:
    """Reduce signed values into CPR1's exact signed-int12 domain."""
    return ((np.asarray(values, dtype=np.int64) + 2048) & 0xFFF).astype(np.int32) - 2048


def round_q8(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Round signed Q8 products to nearest integer, ties away from zero."""
    products = np.asarray(values, dtype=np.int64) * np.asarray(factors, dtype=np.int64)
    return np.where(
        products >= 0, (products + 128) // 256, -((-products + 128) // 256)
    ).astype(np.int32)


@dataclass(frozen=True)
class Ar1BiasModel:
    """Schema-ready fixed-point AR(1) parameters, one per coefficient dimension."""

    factors_q8: np.ndarray
    biases: np.ndarray

    def __post_init__(self) -> None:
        factors = np.asarray(self.factors_q8)
        biases = np.asarray(self.biases)
        if (
            factors.ndim != 1
            or biases.ndim != 1
            or factors.shape != biases.shape
            or not factors.size
        ):
            raise CoefficientPredictorError(
                "AR1 parameters must be equal nonempty vectors"
            )
        if not np.issubdtype(factors.dtype, np.integer) or not np.issubdtype(
            biases.dtype, np.integer
        ):
            raise CoefficientPredictorError("AR1 parameters must be integers")
        if np.any(factors < Q8_MIN) or np.any(factors > Q8_MAX):
            raise CoefficientPredictorError("AR1 Q8 factor outside fixed schema range")
        if np.any(biases < BIAS_MIN) or np.any(biases > BIAS_MAX):
            raise CoefficientPredictorError("AR1 bias outside fixed schema range")

    @property
    def dimensions(self) -> int:
        return int(np.asarray(self.factors_q8).size)

    @property
    def metadata_bytes(self) -> int:
        # int16 Q8 factor plus int8 signed bias for every known schema dimension.
        return self.dimensions * 3


def unpack_ar1_bias_metadata(payload: bytes, dimensions: int) -> Ar1BiasModel:
    """Strictly decode the fixed-size AR parameter field with no trailing data."""
    if dimensions <= 0 or len(payload) != dimensions * 3:
        raise CoefficientPredictorError("invalid AR1 metadata length")
    factors = np.frombuffer(payload[: 2 * dimensions], dtype="<i2").copy()
    biases = np.frombuffer(payload[2 * dimensions :], dtype="i1").copy()
    return Ar1BiasModel(factors, biases)


def _require_coefficients(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise CoefficientPredictorError(
            "coefficients must be a two-dimensional integer matrix with two frames"
        )
    values = values.astype(np.int32, copy=False)
    if np.any(values < -2048) or np.any(values > 2047):
        raise CoefficientPredictorError("coefficients exceed CPR1 signed-int12 range")
    return values


def restore_ar1_bias(residuals: np.ndarray, model: Ar1BiasModel) -> np.ndarray:
    """Reconstruct coefficient values in the signed-int12 domain."""
    values = _require_coefficients(residuals)
    if values.shape[1] != model.dimensions:
        raise CoefficientPredictorError("residual dimensions do not match AR1 model")
    output = np.empty_like(values)
    output[0] = values[0]
    factors = np.asarray(model.factors_q8, dtype=np.int16)
    biases = np.asarray(model.biases, dtype=np.int16)
    for frame in range(1, values.shape[0]):
        prediction = signed_mod(round_q8(output[frame - 1], factors) + biases)
        output[frame] = signed_mod(prediction + values[frame])
    return output
