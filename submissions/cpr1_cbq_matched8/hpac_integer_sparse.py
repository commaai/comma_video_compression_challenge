"""Sparse selected-logit evaluator for IntegerHPAC arithmetic decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hpac_integer import IntegerHPAC, integer_activation, requantize


@dataclass
class GroupPlan:
    targets: torch.Tensor
    h_positions: torch.Tensor
    b1_gather: torch.Tensor
    b2_gather: torch.Tensor
    output_order: torch.Tensor


def _active_offsets(module) -> list[tuple[int, int]]:
    kernel = module.mask.shape[-1]
    center = (kernel - 1) // 2
    offsets = []
    for row, col in module.mask[0, 0].nonzero(as_tuple=False).tolist():
        offsets.append((
            (row - center) * module.dilation,
            (col - center) * module.dilation,
        ))
    return offsets


def _positions_for_group(patch: int, delta: int, group: int):
    return [
        (row, col)
        for row in range(patch)
        for col in range(patch)
        if col + delta * row == group
    ]


def _expanded_positions(positions, offsets, patch):
    return sorted({
        (row + dy, col + dx)
        for row, col in positions
        for dy, dx in offsets
        if 0 <= row + dy < patch and 0 <= col + dx < patch
    })


def _gather_map(outputs, inputs, offsets, patch):
    lookup = {position: index for index, position in enumerate(inputs)}
    sentinel = len(inputs)
    return [
        [
            lookup.get((row + dy, col + dx), sentinel)
            if 0 <= row + dy < patch and 0 <= col + dx < patch
            else sentinel
            for dy, dx in offsets
        ]
        for row, col in outputs
    ]


class SparseIntegerHPAC:
    def __init__(self, model: IntegerHPAC, height=384, width=512):
        if model.norm_mode != "none" or model.use_norm_gates:
            raise ValueError("sparse evaluator does not support channel normalization")
        self.model = model
        self.patch = model.P
        self.patch_rows = height // model.P
        self.patch_cols = width // model.P
        self.patch_count = self.patch_rows * self.patch_cols
        self.a_offsets = _active_offsets(model.conv_a)
        self.b1_offsets = _active_offsets(model.conv_b1)
        self.b2_offsets = _active_offsets(model.conv_b2)
        device = next(model.parameters()).device
        self.plans = [
            self._build_plan(group, device)
            for group in range((1 + model.delta) * model.P - model.delta)
        ]

    def _build_plan(self, group: int, device) -> GroupPlan:
        targets = _positions_for_group(self.patch, self.model.delta, group)
        b1_positions = _expanded_positions(
            targets, self.b2_offsets, self.patch
        )
        h_positions = _expanded_positions(
            b1_positions, self.b1_offsets, self.patch
        )
        b1_gather = _gather_map(
            b1_positions, h_positions, self.b1_offsets, self.patch
        )
        b2_gather = _gather_map(
            targets, b1_positions, self.b2_offsets, self.patch
        )

        patch_major = []
        for patch_index in range(self.patch_count):
            patch_row, patch_col = divmod(patch_index, self.patch_cols)
            for row, col in targets:
                global_row = patch_row * self.patch + row
                global_col = patch_col * self.patch + col
                patch_major.append(global_row * self.patch_cols * self.patch + global_col)
        output_order = sorted(range(len(patch_major)), key=patch_major.__getitem__)
        return GroupPlan(
            targets=torch.tensor(targets, dtype=torch.long, device=device),
            h_positions=torch.tensor(
                h_positions, dtype=torch.long, device=device
            ),
            b1_gather=torch.tensor(
                b1_gather, dtype=torch.long, device=device
            ),
            b2_gather=torch.tensor(
                b2_gather, dtype=torch.long, device=device
            ),
            output_order=torch.tensor(
                output_order, dtype=torch.long, device=device
            ),
        )

    @staticmethod
    def _apply_codes(value, module):
        _, bias, exponent = module.codes()
        if exponent is not None:
            value = value * torch.pow(2.0, exponent).view(1, 1, -1)
        return value + bias.view(1, 1, -1)

    def _conv_a(self, inputs, positions):
        module = self.model.conv_a
        weight, _, _ = module.codes()
        active = module.mask[0, 0].to(torch.bool).flatten()
        weight = weight.flatten(2)[:, :, active].reshape(weight.shape[0], -1)

        padded = F.pad(inputs, (3, 3, 3, 3))
        row = positions[:, 0, None] + 3 + torch.tensor(
            [dy for dy, _ in self.a_offsets], device=inputs.device
        )
        col = positions[:, 1, None] + 3 + torch.tensor(
            [dx for _, dx in self.a_offsets], device=inputs.device
        )
        flat_index = row * (self.patch + 6) + col
        gathered = padded.flatten(2)[:, :, flat_index]
        features = gathered.permute(0, 2, 1, 3).reshape(
            inputs.shape[0], positions.shape[0], -1
        )
        value = F.linear(features, weight)
        return self._apply_codes(value, module)

    def _depthwise(self, value, gather, module):
        weight, _, _ = module.codes()
        active = module.mask[0, 0].to(torch.bool).flatten()
        weight = weight.flatten(2)[:, 0, active]
        zero = torch.zeros(
            value.shape[0], 1, value.shape[2],
            dtype=value.dtype, device=value.device,
        )
        value = torch.cat([value, zero], dim=1)
        gathered = value[:, gather, :]
        result = (gathered * weight.t().view(1, 1, -1, value.shape[2])).sum(2)
        return self._apply_codes(result, module)

    @torch.no_grad()
    def selected_logits(self, current, context, group: int):
        plan = self.plans[group]
        one_hot = F.one_hot(
            current, num_classes=self.model.num_classes
        ).permute(0, 3, 1, 2).float()
        patches = self.model._to_patches(one_hot)
        coords = self.model._patch_coord_grid(
            self.patch_count, current.device
        )
        inputs = torch.cat([patches, coords], dim=1)

        hidden = requantize(
            self._conv_a(inputs, plan.h_positions), 1,
            -self.model.activation_bound, self.model.activation_bound,
        )
        shift, past, scale, spm = context
        if scale is not None:
            hidden = requantize(
                hidden * (16 + scale.squeeze(-1).transpose(1, 2)), 4,
                -self.model.activation_bound, self.model.activation_bound,
            )
        position_index = (
            plan.h_positions[:, 0] * self.patch + plan.h_positions[:, 1]
        )
        past = past.flatten(2)[:, :, position_index].transpose(1, 2)
        if spm is not None:
            past = requantize(
                past + spm.flatten(2)[:, :, position_index].transpose(1, 2),
                0, -self.model.activation_bound, self.model.activation_bound,
            )
        hidden = integer_activation(requantize(
            hidden + shift.squeeze(-1).transpose(1, 2) + past, 0,
            -self.model.activation_bound, self.model.activation_bound,
        ), self.model.activation)

        hidden = integer_activation(requantize(
            self._depthwise(hidden, plan.b1_gather, self.model.conv_b1), 3,
            -self.model.activation_bound, self.model.activation_bound,
        ), self.model.activation)
        hidden = integer_activation(requantize(
            self._depthwise(hidden, plan.b2_gather, self.model.conv_b2), 3,
            -self.model.activation_bound, self.model.activation_bound,
        ), self.model.activation)

        head = self.model.head
        weight, _, _ = head.codes()
        logits = F.linear(hidden, weight[:, :, 0, 0])
        logits = requantize(
            self._apply_codes(logits, head), 3, -32768, 32767
        ) / 8.0
        logits = logits.reshape(-1, self.model.num_classes)[plan.output_order]
        return logits
