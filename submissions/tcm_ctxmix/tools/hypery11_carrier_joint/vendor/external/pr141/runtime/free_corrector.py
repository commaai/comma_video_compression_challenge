# SPDX-License-Identifier: MIT
"""ddm_ma1 - the WITHIN-MISS relative law, the last un-adapted sector of the tail.

WHAT WAS LEFT.  ``ddm_rr4``'s transport is rank-one and every arm since has
worked the same half of it.  ``coding_row`` learns the HIT probability ``q`` and
then scales EVERY non-argmax column by the single scalar ``(1 - q) / one_minus``
-- so the relative law inside the miss sector is exactly the neural prior's,

    r_k = row64[k] / (1 - p_max)          k != argmax

and no arm has ever touched it.  ``ddm_fx1`` §5 priced it at **1,247.19 B** (the
perfect-model ceiling) and named it unbuilt; ``ddm_fx2`` §6 re-confirmed the
decomposition to the cent and named it "still untouched, still the largest priced
target on the token stream".  This module is that build.

THE ESTIMATOR IS THE ONE ALREADY IN THE ARCHIVE, lifted one axis.  ``ddm_rr4``
corrects the hit event by a Krichevsky-Trofimov count ratio -- observed hits over
prior-expected hits.  The same ratio, keyed by class instead of by hit, corrects
the miss sector::

    M[c, k] = (n[c, k] + KT) / (e[c, k] + KT)
    w_k     = row64[k] * M[c, k]                       (k != argmax)
    coded_k = w_k * (S / W) * (1 - q) / one_minus       S = sum row64,  W = sum w

``n`` counts observed miss classes, ``e`` accumulates the prior's own relative
mass over the same population in FIXED POINT so the sum is order-independent
(``ddm_rr4``'s ``phat_q`` discipline).  Both are folded in ``observe``, which the
receiver reaches only after decoding, so the model is causal.  Nothing here is a
new mechanism class -- it is the archive's own estimator pointed at the one
sector nobody had pointed it at.

THE ``S / W`` FACTOR IS WHAT MAKES THE CONTROL EXACT.  It preserves the sector's
total mass, so with ``M == 1`` everywhere ``w_k == row64[k]`` bitwise, ``W == S``
bitwise, ``S / W == 1.0`` exactly, and the coded row is the shipped row.  The
class therefore NESTS the law it extends: ``within_miss=False`` reproduces
``ddm_fx2``'s D1 build bit-for-bit, and a cold cell emits the prior's own
relative law, exactly as ``ddm_rr4``'s cold contexts emit exactly HPAC.

EXACTNESS.  No ``log``, ``exp``, ``pow``, ``**`` or table anywhere on the decision
path -- only ``+ - * /`` and comparisons, all IEEE correctly rounded, so the
receiver is bit-identical on every conforming platform.  This is the
``ddm_rr2`` refusal class (S = 27.83, a libm ULP desynchronising the decoder) and
the sister test walks this module's AST to assert it, exactly as ``ddm_rr4`` and
``ddm_fx1`` do.

THE FEATURES ARE ALREADY COMPUTED AND ALREADY FREE.  The cell is built from the
causal neighbours ``ddm_fx2`` already derives for its spatial members and from
``ddm_rr4``'s ``prev1`` temporal plane.  No new decoded state, no new pass, and
nothing video-derived enters the runtime: the tables start empty at both ends and
are filled only from symbols the receiver has already decoded.  Rule 118 clean.

CONSTANTS ARE MEASURED, NOT CARRIED.  ``ddm_rr4``'s ``MIN_COUNT = 32`` was derived
for a 51,200-cell context over 117.9M positions.  This sector is 1,296 cells over
223,694 records -- a different regime.  Swept: ``min_count`` is monotone and 1
wins (32 costs 12.9 B, 13% of the whole win), and the odds clamp is a broad
plateau whose peak happens to sit at ``ddm_rr4``'s own 2^4.  Carrying 32 would
also have produced a FALSE saturation verdict -- at 32 the richer contexts lose,
at 1 they win.
"""

from __future__ import annotations

import numpy as np

from .fx2_model_axis_corrector import (
    CAUSAL_OFFSETS,
    HEIGHT,
    WIDTH,
    Fx2ModelAxisMixer,
)
from .rr4_free_corrector import GroupState, NUM_CLASSES

__all__ = [
    "MISS_CLAMP_HIGH",
    "MISS_CLAMP_LOW",
    "MISS_KT_ALPHA",
    "MISS_MIN_COUNT",
    "SHIPPED_CONFIG",
    "FreeCorrector",
    "Ma1WithinMissCorrector",
    "miss_cells",
]

UNKNOWN = NUM_CLASSES
"""Sentinel for a neighbour that is not yet decoded, or the temporal plane before
the first frame completes.  A distinct level rather than a fold into a real class,
so "I do not know" never masquerades as evidence."""

MISS_KT_ALPHA = 0.5
"""Krichevsky-Trofimov, the same smoother ``ddm_rr4`` uses. Not swept."""

MISS_MIN_COUNT = 1
"""Observations before a cell may correct. MEASURED over {1,2,4,8,16,32,64}:
monotone, 1 wins. KT already handles a cold cell safely, so a hard gate on top of
it only discards evidence."""

MISS_CLAMP_HIGH = 16.0
"""Symmetric odds clamp, 2^4 exactly. MEASURED over {2,4,8,16,32,64,256,1024}:
a broad plateau, peak at 16 (2 costs 22.5 B; 8 through 1024 are within 0.7 B)."""

MISS_CLAMP_LOW = 1.0 / MISS_CLAMP_HIGH

_PHAT_SCALE = 1 << 30
"""Fixed-point resolution for the expected-mass accumulator, so ``np.add.at``
sums integers and the result cannot depend on summation order."""

# Slot order of ``ddm_fx2.CAUSAL_OFFSETS`` = ((-1,0), (0,-1), (1,-1), (-1,-1)).
_SLOT_LEFT, _SLOT_UP, _SLOT_UPRIGHT, _SLOT_UPLEFT = 0, 1, 2, 3


def miss_cells() -> dict[str, tuple[int, object]]:
    """Cell rules over ``(neighbour classes, prev1)``, all causal and free.

    Every rule takes ``(nb, prev1)`` where ``nb`` is ``[4, n]`` of decoded
    neighbour classes with ``UNKNOWN`` for not-yet-decoded, in the
    ``CAUSAL_OFFSETS`` slot order.
    """
    b = NUM_CLASSES + 1

    def nb3_prev1(nb: np.ndarray, prev1: np.ndarray) -> np.ndarray:
        head = (nb[_SLOT_UP] * b + nb[_SLOT_UPRIGHT]) * b + nb[_SLOT_LEFT]
        return head * b + prev1

    def nb3(nb: np.ndarray, prev1: np.ndarray) -> np.ndarray:
        return (nb[_SLOT_UP] * b + nb[_SLOT_UPRIGHT]) * b + nb[_SLOT_LEFT]

    def nb4(nb: np.ndarray, prev1: np.ndarray) -> np.ndarray:
        head = (nb[_SLOT_UP] * b + nb[_SLOT_UPRIGHT]) * b + nb[_SLOT_LEFT]
        return head * b + nb[_SLOT_UPLEFT]

    def nbup_nbleft(nb: np.ndarray, prev1: np.ndarray) -> np.ndarray:
        return nb[_SLOT_UP] * b + nb[_SLOT_LEFT]

    # Sizes are written as products, not powers.  ``**`` on integer constants is
    # not a libm call and could not desynchronise anything, but the AST gate that
    # enforces the ddm_rr2 refusal bans the operator outright -- and a gate with
    # an exception carved into it is a gate that can be gamed later.  Cheaper to
    # not need the exception.
    b2 = b * b
    b3 = b2 * b
    b4 = b3 * b
    return {
        "nb3_prev1": (b4, nb3_prev1),
        "nb3": (b3, nb3),
        "nb4": (b4, nb4),
        "nbup_nbleft": (b2, nbup_nbleft),
    }


class Ma1WithinMissCorrector(Fx2ModelAxisMixer):
    """``ddm_fx2``'s live law plus an adaptive miss-sector relative law.

    With ``within_miss=False`` this class IS ``ddm_fx2``, bit-for-bit -- the
    control that makes every delta attributable to the new sector and never to
    the plumbing.
    """

    def __init__(self, plane: int, **kwargs) -> None:
        self._within_miss = bool(kwargs.pop("within_miss", True))
        cell_name = str(kwargs.pop("miss_cell", "nb3_prev1"))
        self._miss_min_count = int(kwargs.pop("miss_min_count", MISS_MIN_COUNT))
        clamp_high = float(kwargs.pop("miss_clamp", MISS_CLAMP_HIGH))
        super().__init__(plane, **kwargs)

        cells = miss_cells()
        if cell_name not in cells:
            raise ValueError(f"unknown miss cell {cell_name!r}; have {sorted(cells)}")
        self._miss_cell_name = cell_name
        self._n_miss_cells, self._miss_rule = cells[cell_name]
        self._miss_clamp_high = clamp_high
        self._miss_clamp_low = 1.0 / clamp_high

        self._miss_counts = np.zeros((self._n_miss_cells, NUM_CLASSES), dtype=np.int64)
        self._miss_expect = np.zeros((self._n_miss_cells, NUM_CLASSES), dtype=np.int64)
        self._miss_seen = np.zeros(self._n_miss_cells, dtype=np.int64)
        self._miss_pending: np.ndarray | None = None
        self._nb_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._lanes = np.arange(NUM_CLASSES)[None, :]

    # -- the cell -----------------------------------------------------------

    def _causal_neighbours(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cache the parent's neighbour gather so the cell can reuse it.

        DECODE COST, not tidiness.  ``ddm_fx2``'s ``group_state`` already gathers
        all four causal neighbours for its spatial members; recomputing them here
        would run that loop twice per group over the whole plane, on the one
        constraint that actually picks this candidate (``ddm_fx1`` measured the
        real parse-back at 1,639.78 s against a 1,800 s budget).  Caching makes
        the cell's marginal cost the index arithmetic alone.
        """
        result = super()._causal_neighbours(flat)
        self._nb_cache = result
        return result

    def _miss_cell(self, flat: np.ndarray) -> np.ndarray:
        """Cell index per position, from already-decoded state only."""
        # CONSUME the cache rather than merely reading it.  A size check alone
        # would false-match a stale entry whenever two consecutive groups happen
        # to hold the same number of positions, and 190 groups over one plane
        # make that collision ordinary rather than exotic.  In the driven path
        # the parent refills it every group, so consuming costs nothing and the
        # stale read becomes unrepresentable.
        cached = self._nb_cache
        self._nb_cache = None
        if cached is not None and cached[0].shape[1] == flat.size:
            classes, available = cached
            nb = np.where(available, classes, UNKNOWN)
        else:
            # Only reached when the parent did not gather (direct unit-test use).
            x = flat % WIDTH
            y = flat // WIDTH
            nb = np.empty((len(CAUSAL_OFFSETS), flat.size), dtype=np.int64)
            for slot, (dx, dy) in enumerate(CAUSAL_OFFSETS):
                nx = x + dx
                ny = y + dy
                inside = (nx >= 0) & (nx < WIDTH) & (ny >= 0) & (ny < HEIGHT)
                src = np.clip(ny, 0, HEIGHT - 1) * WIDTH + np.clip(nx, 0, WIDTH - 1)
                usable = inside & self.known[src]
                nb[slot] = np.where(usable, self.current[src].astype(np.int64), UNKNOWN)
        if self.have_prev:
            prev1 = self.prev1[flat].astype(np.int64)
        else:
            prev1 = np.full(flat.size, UNKNOWN, dtype=np.int64)
        return self._miss_rule(nb, prev1)

    def _miss_multiplier(self, cell: np.ndarray) -> np.ndarray:
        """``M[c, k]``, exactly 1.0 wherever the cell has nothing to say."""
        m = np.ones((cell.size, NUM_CLASSES), dtype=np.float64)
        warm = self._miss_seen[cell] >= self._miss_min_count
        if not warm.any():
            return m
        ratio = (self._miss_counts[cell] + MISS_KT_ALPHA) / (
            self._miss_expect[cell] / _PHAT_SCALE + MISS_KT_ALPHA
        )
        np.clip(ratio, self._miss_clamp_low, self._miss_clamp_high, out=ratio)
        return np.where(warm[:, None], ratio, m)

    # -- driving ------------------------------------------------------------

    def group_state(
        self, probability: np.ndarray, predicted: np.ndarray, positions: np.ndarray
    ) -> GroupState:
        state = super().group_state(probability, predicted, positions)
        if self._within_miss:
            self._miss_pending = self._miss_cell(
                np.asarray(positions, dtype=np.int64).reshape(-1)
            )
        return state

    def coding_row(self, state: GroupState) -> np.ndarray:
        row = super().coding_row(state)
        if not self._within_miss or self._miss_pending is None:
            return row

        cell = self._miss_pending
        m = self._miss_multiplier(cell)
        arg = state.arg
        index = np.arange(arg.size)
        m[index, arg] = 1.0

        active = np.any(m != 1.0, axis=1)
        if not np.any(active):
            return row

        # Mass-preserving reweight of the non-argmax columns.  Sums are taken
        # lane by lane in a fixed order so the reduction is unambiguous rather
        # than dependent on a library's pairwise blocking.
        row64 = row.astype(np.float64)
        weighted = row64 * m
        weighted[index, arg] = 0.0
        base = row64.copy()
        base[index, arg] = 0.0
        big_w = np.zeros(arg.size, dtype=np.float64)
        big_s = np.zeros(arg.size, dtype=np.float64)
        for lane in range(NUM_CLASSES):
            big_w += weighted[:, lane]
            big_s += base[:, lane]

        usable = active & (big_w > 0.0) & (big_s > 0.0)
        if not np.any(usable):
            return row
        scale = np.ones(arg.size, dtype=np.float64)
        scale[usable] = big_s[usable] / big_w[usable]
        updated = weighted * scale[:, None]
        updated[index, arg] = row64[index, arg]
        out = np.where(usable[:, None], updated, row64)
        return out.astype(np.float32)

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        if self._within_miss and self._miss_pending is not None:
            decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
            miss = decoded != state.arg
            if np.any(miss):
                cell = self._miss_pending[miss]
                row64 = state.row64[miss]
                one_minus = state.one_minus[miss]
                relative = row64 / one_minus[:, None]
                relative[np.arange(cell.size), state.arg[miss]] = 0.0
                np.add.at(self._miss_counts, (cell, decoded[miss]), 1)
                np.add.at(
                    self._miss_expect,
                    (cell[:, None], self._lanes),
                    np.rint(relative * _PHAT_SCALE).astype(np.int64),
                )
                np.add.at(self._miss_seen, cell, 1)
            self._miss_pending = None
        super().observe(state, symbols)


from .fx2_model_axis_corrector import SHIPPED_CONFIG as _FX2_SHIPPED_CONFIG

SHIPPED_CONFIG: dict = dict(_FX2_SHIPPED_CONFIG)
SHIPPED_CONFIG.update(
    {
        "within_miss": True,
        "miss_cell": "nb3_prev1",
        "miss_min_count": MISS_MIN_COUNT,
        "miss_clamp": MISS_CLAMP_HIGH,
    }
)
"""``ddm_fx2``'s frozen 13-member D1 build, plus the miss-sector law.

The inherited half is UNCHANGED and deliberately so: this arm adds a sector, it
does not re-select the hit-event model. Keeping D1 frozen means the measured
delta is the new sector alone, and it means the candidate inherits D1's already
parse-backed decode profile rather than needing a fresh one for both halves.
"""


class FreeCorrector(Ma1WithinMissCorrector):
    """Drop-in for the receiver: a decoder takes no arguments."""

    def __init__(self, plane: int) -> None:
        super().__init__(plane, **SHIPPED_CONFIG)
