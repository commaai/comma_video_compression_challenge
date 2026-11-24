"""Inference optimizations for the integer HPAC token decoder."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch.nn import functional


def configure_cuda_reproducibility() -> dict[str, object]:
    """Select the strictest practical CUDA float32 reproducibility settings.

    The integer HPAC reductions are deliberately bounded below float32's
    exact-integer limit.  Disabling reduced-precision products and algorithm
    autotuning keeps those reductions, plus the renderer's float32 work, as
    comparable as CUDA permits across GPU generations.  We intentionally do
    not force an alternate deterministic cuDNN kernel: that changes the
    already scored rendered pixels even on the reference GPU.  Cross-device
    bitwise identity still requires the full rendered-output hash gate.
    """
    precision_api = "legacy"
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        precision_api = "fp32_precision"
        matmul_precision = torch.backends.cuda.matmul.fp32_precision
        cudnn_precision = torch.backends.cudnn.conv.fp32_precision
        precision_label = "highest"
    except (AttributeError, RuntimeError):  # pragma: no cover - version-specific.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        matmul_precision = "ieee"
        cudnn_precision = "ieee"
        precision_label = torch.get_float32_matmul_precision()
    torch.backends.cudnn.benchmark = False
    # HPAC's rounded integer reductions are exact with IEEE float32.  Forcing
    # deterministic cuDNN algorithms changes the promoted renderer rail, so
    # preserve its explicitly non-autotuned kernel selection.
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {
        "precision_api": precision_api,
        "float32_matmul_precision": precision_label,
        "cuda_matmul_fp32_precision": matmul_precision,
        "cudnn_conv_fp32_precision": cudnn_precision,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


@dataclass(frozen=True)
class _ConvAPlan:
    class_indices: torch.Tensor
    valid: torch.Tensor
    coordinates: torch.Tensor


@dataclass(frozen=True)
class _SparseCache:
    conv_a_weight: torch.Tensor
    conv_a_plans: tuple[_ConvAPlan, ...]
    depthwise_weights: dict[int, torch.Tensor]
    affine_values: dict[int, tuple[torch.Tensor, torch.Tensor | None]]
    depthwise_zero: torch.Tensor


def _constant_result(values: tuple[Any, ...]):
    def result() -> tuple[Any, ...]:
        return values

    return result


def _constant_tensor_result(value: torch.Tensor):
    def result() -> torch.Tensor:
        return value

    return result


def _inference_round(value: torch.Tensor) -> torch.Tensor:
    return value.round()


def _inference_requantize(
    value: torch.Tensor,
    shift: int,
    low: int = -127,
    high: int = 127,
) -> torch.Tensor:
    if shift:
        value = value / (1 << shift)
    return value.round().clamp(low, high)


def _freeze_model_codes(model: Any) -> None:
    for module in model.modules():
        codes = getattr(module, "codes", None)
        if not callable(codes):
            continue
        values = tuple(None if value is None else value.detach() for value in codes())
        module.codes = _constant_result(values)

    frame_codes = model.frame_codes().detach()
    model.frame_codes = _constant_tensor_result(frame_codes)


def _apply_codes(self, value: torch.Tensor, module: Any) -> torch.Tensor:
    bias, multiplier = self._cpr1_sparse_cache.affine_values[id(module)]
    if multiplier is not None:
        value = value * multiplier
    return value + bias


def _conv_a_features(
    self,
    current: torch.Tensor,
    group: int,
) -> torch.Tensor:
    cache = self._cpr1_sparse_cache
    plan = cache.conv_a_plans[group]
    classes = current.reshape(-1)[plan.class_indices]
    one_hot = functional.one_hot(
        classes,
        num_classes=self.model.num_classes,
    )
    one_hot = one_hot.permute(0, 1, 3, 2).to(cache.conv_a_weight.dtype)
    one_hot = one_hot * plan.valid
    coordinates = plan.coordinates.expand(self.patch_count, -1, -1, -1)
    features = torch.cat([one_hot, coordinates], dim=2)
    features = features.reshape(self.patch_count, features.shape[1], -1)
    value = functional.linear(features, cache.conv_a_weight)
    return self._apply_codes(value, self.model.conv_a)


def _depthwise(
    self,
    value: torch.Tensor,
    gather: torch.Tensor,
    module: Any,
) -> torch.Tensor:
    cache = self._cpr1_sparse_cache
    value = torch.cat([value, cache.depthwise_zero], dim=1)
    gathered = value[:, gather, :]
    weight = cache.depthwise_weights[id(module)]
    result = (gathered * weight).sum(2)
    return self._apply_codes(result, module)


def _selected_logits(
    self,
    current: torch.Tensor,
    context: tuple[torch.Tensor, ...],
    group: int,
) -> torch.Tensor:
    plan = self.plans[group]
    hidden = _inference_requantize(
        self._conv_a_features(current, group),
        1,
        -self.model.activation_bound,
        self.model.activation_bound,
    )
    shift, past, scale, spm = context
    if scale is not None:
        hidden = _inference_requantize(
            hidden * (16 + scale.squeeze(-1).transpose(1, 2)),
            4,
            -self.model.activation_bound,
            self.model.activation_bound,
        )
    position_index = plan.h_positions[:, 0] * self.patch + plan.h_positions[:, 1]
    past = past.flatten(2)[:, :, position_index].transpose(1, 2)
    if spm is not None:
        past = _inference_requantize(
            past + spm.flatten(2)[:, :, position_index].transpose(1, 2),
            0,
            -self.model.activation_bound,
            self.model.activation_bound,
        )
    hidden = _inference_requantize(
        hidden + shift.squeeze(-1).transpose(1, 2) + past,
        0,
        -self.model.activation_bound,
        self.model.activation_bound,
    )
    hidden = functional.relu(hidden)

    hidden = _inference_requantize(
        self._depthwise(
            hidden,
            plan.b1_gather,
            self.model.conv_b1,
        ),
        3,
        -self.model.activation_bound,
        self.model.activation_bound,
    )
    hidden = functional.relu(hidden)
    hidden = _inference_requantize(
        self._depthwise(
            hidden,
            plan.b2_gather,
            self.model.conv_b2,
        ),
        3,
        -self.model.activation_bound,
        self.model.activation_bound,
    )
    hidden = functional.relu(hidden)

    head = self.model.head
    weight, _, _ = head.codes()
    logits = functional.linear(hidden, weight[:, :, 0, 0])
    logits = _inference_requantize(
        self._apply_codes(logits, head),
        3,
        -32768,
        32767,
    )
    logits = logits / 8.0
    logits = logits.reshape(-1, self.model.num_classes)
    return logits[plan.output_order]


def _conv_a_plans(sparse: Any) -> tuple[_ConvAPlan, ...]:
    device = sparse.plans[0].targets.device
    offsets = torch.tensor(
        sparse.a_offsets,
        dtype=torch.long,
        device=device,
    )
    patch_indices = torch.arange(
        sparse.patch_count,
        dtype=torch.long,
        device=device,
    )
    patch_rows = patch_indices // sparse.patch_cols
    patch_columns = patch_indices % sparse.patch_cols
    plans = []
    for plan in sparse.plans:
        local_rows = plan.h_positions[:, 0, None] + offsets[None, :, 0]
        local_columns = plan.h_positions[:, 1, None] + offsets[None, :, 1]
        valid = (
            (local_rows >= 0)
            & (local_rows < sparse.patch)
            & (local_columns >= 0)
            & (local_columns < sparse.patch)
        )
        global_rows = patch_rows[:, None, None] * sparse.patch + local_rows[None]
        global_columns = (
            patch_columns[:, None, None] * sparse.patch + local_columns[None]
        )
        class_indices = (
            global_rows * (sparse.patch_cols * sparse.patch) + global_columns
        )
        class_indices = torch.where(
            valid[None],
            class_indices,
            torch.zeros_like(class_indices),
        )
        rows = (local_rows - sparse.patch // 2).to(torch.float32)
        columns = (local_columns - sparse.patch // 2).to(torch.float32)
        coordinates = torch.stack([rows, columns], dim=1)
        coordinates = coordinates * valid[:, None]
        plans.append(
            _ConvAPlan(
                class_indices=class_indices,
                valid=valid[None, :, None, :],
                coordinates=coordinates[None],
            )
        )
    return tuple(plans)


def optimize_sparse_evaluator(sparse: Any) -> None:
    """Freeze immutable model transforms used during token decoding."""
    model = sparse.model
    _freeze_model_codes(model)
    model_module = sys.modules[type(model).__module__]
    model_module.ste_round = _inference_round
    model_module.requantize = _inference_requantize

    conv_a = model.conv_a
    conv_a_weight, _, _ = conv_a.codes()
    active = conv_a.mask[0, 0].to(torch.bool).flatten()
    conv_a_weight = conv_a_weight.flatten(2)[:, :, active].reshape(
        conv_a_weight.shape[0], -1
    )
    depthwise_weights = {}
    affine_values = {}
    for module in (model.conv_a, model.conv_b1, model.conv_b2, model.head):
        weight, bias, exponent = module.codes()
        multiplier = (
            None if exponent is None else torch.pow(2.0, exponent).view(1, 1, -1)
        )
        affine_values[id(module)] = (bias.view(1, 1, -1), multiplier)
        if module in (model.conv_b1, model.conv_b2):
            active = module.mask[0, 0].to(torch.bool).flatten()
            depthwise_weights[id(module)] = (
                weight.flatten(2)[:, 0, active].t().view(1, 1, -1, weight.shape[0])
            )

    sparse._cpr1_sparse_cache = _SparseCache(
        conv_a_weight=conv_a_weight,
        conv_a_plans=_conv_a_plans(sparse),
        depthwise_weights=depthwise_weights,
        affine_values=affine_values,
        depthwise_zero=torch.zeros(
            sparse.patch_count,
            1,
            model.ch,
            dtype=conv_a_weight.dtype,
            device=conv_a_weight.device,
        ),
    )
    sparse._apply_codes = MethodType(_apply_codes, sparse)
    sparse._conv_a_features = MethodType(_conv_a_features, sparse)
    sparse._depthwise = MethodType(_depthwise, sparse)
    sparse.selected_logits = MethodType(_selected_logits, sparse)
