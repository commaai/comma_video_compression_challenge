"""Integer-lattice HPAC student with exact cross-device inference operations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def ste_round(value: torch.Tensor) -> torch.Tensor:
    return value + (value.round() - value).detach()


def requantize(value: torch.Tensor, shift: int, low=-127, high=127) -> torch.Tensor:
    value = value / (1 << shift)
    return ste_round(value).clamp(low, high)


def integer_channel_norm(
    value: torch.Tensor, mode: str, activation_bound: int
) -> torch.Tensor:
    if mode == "none":
        return value
    mean = ste_round(value.mean(dim=1, keepdim=True))
    centered = value - mean
    if mode == "center":
        return centered.clamp(-activation_bound, activation_bound)
    if mode != "power":
        raise ValueError(f"unsupported integer norm mode: {mode}")
    energy = ste_round(centered.square().mean(dim=1, keepdim=True))
    normalized = torch.where(
        energy < 32,
        centered * 4,
        torch.where(
            energy < 128,
            centered * 2,
            torch.where(
                energy < 512,
                centered,
                torch.where(
                    energy < 2048,
                    ste_round(centered / 2),
                    ste_round(centered / 4),
                ),
            ),
        ),
    )
    return normalized.clamp(-activation_bound, activation_bound)


def integer_activation(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "relu":
        return F.relu(value)
    if mode == "leaky":
        return torch.where(value >= 0, value, ste_round(value / 4))
    raise ValueError(f"unsupported integer activation: {mode}")


class IntegerNormGate(nn.Module):
    def __init__(self, channels: int, activation_bound: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(channels))
        self.activation_bound = activation_bound
        self.gate_bound = 16

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = integer_channel_norm(
            value, "power", self.activation_bound
        )
        gate = ste_round(self.gate.clamp(0, self.gate_bound)).view(1, -1, 1, 1)
        mixed = value * (16 - gate) + normalized * gate
        return requantize(
            mixed, 4, -self.activation_bound, self.activation_bound
        )


def patch_group_mask(kernel: int, delta: int, type_: str) -> torch.Tensor:
    mask = torch.zeros(kernel, kernel)
    center = (kernel - 1) // 2
    for row in range(kernel):
        for col in range(kernel):
            offset = col - center + delta * (row - center)
            if offset < 0 or (type_ == "B" and offset == 0):
                mask[row, col] = 1
    return mask


class IntegerConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel, *, padding=0, dilation=1,
                 groups=1, mask: Optional[torch.Tensor] = None,
                 weight_bound=127, use_weight_scales=False,
                 exponent_min=-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in // groups, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight_bound = weight_bound
        self.exponent_min = exponent_min
        nn.init.normal_(self.weight, mean=0.0, std=1.5)
        if use_weight_scales:
            self.weight.data.mul_(8)
            self.exponent = nn.Parameter(torch.full((c_out,), -3.0))
        if mask is None:
            mask = torch.ones(kernel, kernel)
        self.register_buffer("mask", mask.view(1, 1, kernel, kernel), persistent=False)

    def codes(self):
        weight = ste_round(
            self.weight.clamp(-self.weight_bound, self.weight_bound)
        ) * self.mask
        bias = ste_round(self.bias.clamp(-32768, 32767))
        exponent = None
        if hasattr(self, "exponent"):
            exponent = ste_round(self.exponent.clamp(self.exponent_min, 0))
        return weight, bias, exponent

    def forward(self, value):
        weight, bias, exponent = self.codes()
        result = F.conv2d(
            value, weight, None if exponent is not None else bias,
            padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )
        if exponent is not None:
            result = result * torch.pow(2.0, exponent).view(1, -1, 1, 1)
            result = result + bias.view(1, -1, 1, 1)
        return result


class IntegerLinear(nn.Module):
    def __init__(self, c_in: int, c_out: int, weight_bound=127,
                 use_weight_scales=False, exponent_min=-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.weight_bound = weight_bound
        self.exponent_min = exponent_min
        nn.init.normal_(self.weight, mean=0.0, std=1.5)
        if use_weight_scales:
            self.weight.data.mul_(8)
            self.exponent = nn.Parameter(torch.full((c_out,), -3.0))

    def codes(self):
        weight = ste_round(
            self.weight.clamp(-self.weight_bound, self.weight_bound)
        )
        bias = ste_round(self.bias.clamp(-32768, 32767))
        exponent = None
        if hasattr(self, "exponent"):
            exponent = ste_round(self.exponent.clamp(self.exponent_min, 0))
        return weight, bias, exponent

    def forward(self, value):
        weight, bias, exponent = self.codes()
        result = F.linear(value, weight, None if exponent is not None else bias)
        if exponent is not None:
            result = result * torch.pow(2.0, exponent).view(1, -1)
            result = result + bias.view(1, -1)
        return result


class IntegerHPAC(nn.Module):
    def __init__(self, num_pairs=600, num_classes=5, patch=32,
                 delta=2, channels=64, frame_dim=8, norm_mode="none",
                 activation="relu", use_frame_scale=False, weight_bound=127,
                 activation_bound=127, use_weight_scales=False,
                 weight_exponent_min=-6, use_spm=False,
                 use_norm_gates=False):
        super().__init__()
        self.num_pairs = num_pairs
        self.num_classes = num_classes
        self.P = patch
        self.delta = delta
        self.ch = channels
        self.norm_mode = norm_mode
        self.activation = activation
        self.use_frame_scale = use_frame_scale
        self.weight_bound = weight_bound
        self.activation_bound = activation_bound
        self.use_weight_scales = use_weight_scales
        self.use_spm = use_spm
        self.use_norm_gates = use_norm_gates
        if channels * weight_bound * activation_bound + 32768 >= 2 ** 24:
            raise ValueError("head convolution can exceed exact float32 integer range")
        if norm_mode != "none" and activation_bound > 511:
            raise ValueError("integer channel energy requires activation_bound <= 511")
        self.frame_embed = nn.Embedding(num_pairs, frame_dim)
        nn.init.normal_(self.frame_embed.weight, mean=0.0, std=2.0)
        linear_kwargs = {
            "weight_bound": weight_bound,
            "use_weight_scales": use_weight_scales,
            "exponent_min": weight_exponent_min,
        }
        conv_kwargs = {
            "weight_bound": weight_bound,
            "use_weight_scales": use_weight_scales,
            "exponent_min": weight_exponent_min,
        }
        self.frame_shift = IntegerLinear(frame_dim, channels, **linear_kwargs)
        if use_frame_scale:
            self.frame_scale = IntegerLinear(frame_dim, channels, **linear_kwargs)
            nn.init.zeros_(self.frame_scale.weight)
            nn.init.zeros_(self.frame_scale.bias)
        self.conv_a = IntegerConv2d(
            num_classes + 2, channels, 7, padding=3,
            mask=patch_group_mask(7, delta, "A"), **conv_kwargs,
        )
        self.conv_b1 = IntegerConv2d(
            channels, channels, 5, padding=4, dilation=2, groups=channels,
            mask=patch_group_mask(5, delta, "B"), **conv_kwargs,
        )
        self.conv_b2 = IntegerConv2d(
            channels, channels, 3, padding=4, dilation=4, groups=channels,
            mask=patch_group_mask(3, delta, "B"), **conv_kwargs,
        )
        self.conv_past = IntegerConv2d(
            num_classes, channels, 3, padding=1, **conv_kwargs
        )
        if use_spm:
            self.spm_dw = IntegerConv2d(
                channels, channels, 3, padding=1, groups=channels,
                **conv_kwargs,
            )
            self.spm_pw = IntegerConv2d(
                channels, channels, 1, **conv_kwargs
            )
            nn.init.zeros_(self.spm_pw.weight)
            nn.init.zeros_(self.spm_pw.bias)
        self.head = IntegerConv2d(
            channels, num_classes, 1, **conv_kwargs
        )
        if use_norm_gates:
            self.norm_a = IntegerNormGate(channels, activation_bound)
            self.norm_b1 = IntegerNormGate(channels, activation_bound)
            self.norm_b2 = IntegerNormGate(channels, activation_bound)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)

    def frame_codes(self):
        return ste_round(self.frame_embed.weight.clamp(-127, 127))

    def _patch_coord_grid(self, batch: int, device: torch.device):
        if self._coord_cache.numel() == 0 or self._coord_cache.device != device:
            axis = torch.arange(self.P, device=device, dtype=torch.float32)
            axis = axis - self.P // 2
            yy = axis.view(1, 1, self.P, 1).expand(1, 1, self.P, self.P)
            xx = axis.view(1, 1, 1, self.P).expand(1, 1, self.P, self.P)
            self._coord_cache = torch.cat([yy, xx], dim=1)
        return self._coord_cache.expand(batch, -1, -1, -1)

    def _to_patches(self, value):
        batch, channels, height, width = value.shape
        patch_rows, patch_cols = height // self.P, width // self.P
        value = value.view(
            batch, channels, patch_rows, self.P, patch_cols, self.P
        ).permute(0, 2, 4, 1, 3, 5).contiguous()
        return value.view(batch * patch_rows * patch_cols, channels, self.P, self.P)

    def _from_patches(self, value, batch: int, patch_rows: int, patch_cols: int):
        channels = value.shape[1]
        value = value.view(
            batch, patch_rows, patch_cols, channels, self.P, self.P
        ).permute(0, 3, 1, 4, 2, 5).contiguous()
        return value.view(batch, channels, patch_rows * self.P, patch_cols * self.P)

    def prepare_frame_context(self, idx, previous_raw):
        batch, height, width = previous_raw.shape
        patch_count = (height // self.P) * (width // self.P)
        embedding = self.frame_codes()[idx]
        shift = requantize(
            self.frame_shift(embedding), 1,
            -self.activation_bound, self.activation_bound,
        )
        shift = shift.view(batch, 1, self.ch, 1, 1).expand(
            batch, patch_count, self.ch, 1, 1
        ).reshape(batch * patch_count, self.ch, 1, 1)
        previous_one_hot = F.one_hot(
            previous_raw, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        past = requantize(
            self.conv_past(previous_one_hot), 0,
            -self.activation_bound, self.activation_bound,
        )
        spm = None
        if self.use_spm:
            patch_rows, patch_cols = height // self.P, width // self.P
            pooled = past.view(
                batch, self.ch, patch_rows, self.P, patch_cols, self.P
            ).mean(dim=(3, 5))
            pooled = ste_round(pooled)
            pooled = integer_activation(requantize(
                self.spm_dw(pooled), 3,
                -self.activation_bound, self.activation_bound,
            ), self.activation)
            pooled = requantize(
                self.spm_pw(pooled), 4,
                -self.activation_bound, self.activation_bound,
            )
            spm = pooled.unsqueeze(3).unsqueeze(5).expand(
                batch, self.ch, patch_rows, self.P, patch_cols, self.P
            ).contiguous().view(batch, self.ch, height, width)
        scale = None
        if self.use_frame_scale:
            scale = requantize(self.frame_scale(embedding), 4, -8, 8)
            scale = scale.view(batch, 1, self.ch, 1, 1).expand(
                batch, patch_count, self.ch, 1, 1
            ).reshape(batch * patch_count, self.ch, 1, 1)
        return (
            shift, self._to_patches(past), scale,
            None if spm is None else self._to_patches(spm),
        )

    def cached_context_logits(self, current, context):
        batch, height, width = current.shape
        patch_rows, patch_cols = height // self.P, width // self.P
        patch_count = patch_rows * patch_cols
        one_hot = F.one_hot(
            current, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        patches = self._to_patches(one_hot)
        coords = self._patch_coord_grid(batch * patch_count, current.device)
        hidden = requantize(
            self.conv_a(torch.cat([patches, coords], dim=1)), 1,
            -self.activation_bound, self.activation_bound,
        )
        shift, past, scale, spm = context
        if scale is not None:
            hidden = requantize(
                hidden * (16 + scale), 4,
                -self.activation_bound, self.activation_bound,
            )
        if self.norm_mode == "none":
            if self.use_norm_gates:
                hidden = self.norm_a(hidden)
            if spm is not None:
                past = requantize(
                    past + spm, 0,
                    -self.activation_bound, self.activation_bound,
                )
            hidden = integer_activation(requantize(
                hidden + shift + past, 0,
                -self.activation_bound, self.activation_bound,
            ), self.activation)
            hidden = requantize(
                self.conv_b1(hidden), 3,
                -self.activation_bound, self.activation_bound,
            )
            if self.use_norm_gates:
                hidden = self.norm_b1(hidden)
            hidden = integer_activation(hidden, self.activation)
            hidden = requantize(
                self.conv_b2(hidden), 3,
                -self.activation_bound, self.activation_bound,
            )
            if self.use_norm_gates:
                hidden = self.norm_b2(hidden)
            hidden = integer_activation(hidden, self.activation)
            logits = requantize(self.head(hidden), 3, -32768, 32767)
            return self._from_patches(
                logits, batch, patch_rows, patch_cols
            ) / 8.0

        hidden = integer_channel_norm(
            hidden, self.norm_mode, self.activation_bound
        )
        hidden = integer_activation(requantize(
            hidden + shift, 0, -self.activation_bound, self.activation_bound
        ), self.activation)
        hidden = requantize(
            hidden + past, 0, -self.activation_bound, self.activation_bound
        )
        if spm is not None:
            hidden = requantize(
                hidden + spm, 0,
                -self.activation_bound, self.activation_bound,
            )
        hidden = integer_channel_norm(
            requantize(
                self.conv_b1(hidden), 3,
                -self.activation_bound, self.activation_bound,
            ), self.norm_mode, self.activation_bound,
        )
        hidden = integer_activation(hidden, self.activation)
        hidden = integer_channel_norm(
            requantize(
                self.conv_b2(hidden), 3,
                -self.activation_bound, self.activation_bound,
            ), self.norm_mode, self.activation_bound,
        )
        hidden = integer_activation(hidden, self.activation)
        logits = requantize(self.head(hidden), 3, -32768, 32767)
        return self._from_patches(logits, batch, patch_rows, patch_cols) / 8.0

    def forward(self, current, idx, previous_raw):
        context = self.prepare_frame_context(idx, previous_raw)
        return self.cached_context_logits(current, context)
