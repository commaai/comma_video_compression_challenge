"""Fixed tensor schema used by the CPR1 semantic renderer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorSchema:
    name: str
    shape: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def is_fp16(self) -> bool:
        return len(self.shape) < 2

    @property
    def scale_count(self) -> int:
        if self.is_fp16:
            return 0
        return self.shape[-1] if self.name.endswith("embed.weight") else self.shape[0]


def _schemas() -> tuple[TensorSchema, ...]:
    items: list[tuple[str, tuple[int, ...]]] = [
        ("token_embed.weight", (5, 96)),
        ("frame_embed.weight", (600, 8)),
        ("coord_mix.weight", (96, 100, 1, 1)),
        ("coord_mix.bias", (96,)),
    ]
    for block in range(4):
        prefix = f"blocks.{block}"
        items.extend(
            [
                (f"{prefix}.dw.weight", (96, 1, 3, 3)),
                (f"{prefix}.dw.bias", (96,)),
                (f"{prefix}.pw.weight", (96, 96, 1, 1)),
                (f"{prefix}.pw.bias", (96,)),
                (f"{prefix}.norm.weight", (96,)),
                (f"{prefix}.norm.bias", (96,)),
                (f"{prefix}.film.weight", (192, 8)),
                (f"{prefix}.film.bias", (192,)),
            ]
        )
    items.extend([("head.weight", (3, 96, 3, 3)), ("head.bias", (3,))])
    return tuple(TensorSchema(name, shape) for name, shape in items)


SEMANTIC_SCHEMA = _schemas()


@dataclass
class TensorStorage:
    schema: TensorSchema
    format: str
    values: np.ndarray
    scales: np.ndarray | None
    codes: np.ndarray | None
    raw_fp16: bytes | None = None
    raw_scales: bytes | None = None

    @property
    def raw_bytes(self) -> int:
        if self.format == "fp16":
            return len(self.raw_fp16 or b"")
        assert self.codes is not None and self.scales is not None
        return len(self.raw_scales or b"") + (self.codes.size + 1) // 2
