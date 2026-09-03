"""ddm_rr4 - the free decode-time probability corrector, platform-exact (v2).

WHY v2 EXISTS.  ``ddm_rr2`` shipped a corrector that was proven byte-exact on
macOS/arm64 and then desynchronised on the contest T4 (Linux/x86_64), which
refused the row at ``S = 27.83``.  ``ddm_rr4`` measured the cause and it is not
what the refusal memo assumed.  The neural AR-prior probabilities are NOT
device-dependent: the frontier archive reports the SAME
``corrected_quantized_logit_sha256 = 562ac652...`` and
``corrected_cdf_input_sha256 = dd48843b...`` on macOS-arm64-CPU and on
T4-CUDA-x86_64, because the HPAC student is an integer lattice with, in its own
words, "exact cross-device inference operations".

The divergence was introduced by the corrector itself.  v1 computed

    logit_p = log2(p_max / one_minus)
    q       = 1 / (1 + exp2(-(logit_p + delta)))

so every emitted probability passed through ``np.log2`` and ``np.exp2`` - libm
routines that are NOT correctly rounded and differ by an ULP between platforms.
Measured on the real retained ``ddm_hm1`` logits, this round trip perturbs
**50.09%** of positions by about one float32 ULP even where ``delta`` is exactly
0.0 and v1's own docstring promises "exactly HPAC".  The RC64 backend turns a
row into integer frequencies with ``frequency = (uint64_t)(value * 2^31)``,
which is exact, so one float32 ULP at p ~ 0.5 moves a frequency by 128 counts.
Any single differing position desynchronises the arithmetic decoder for the
rest of the stream.

WHAT v2 CHANGES.  Nothing about the estimator's mathematics.  v2 computes the
same quantities with exact operations only.  The key observation is that the
transcendentals cancel: v1 only ever needed ``m = 2^delta``, and

    delta = log2(e/(1-e)) - log2(x/(1-x))    =>    2^delta = (e/(1-e)) / (x/(1-x))

so the log and the exp annihilate and ``m`` is a ratio of counts.  The corrected
row then follows from an algebraic identity that never forms a logarithm:

    q     = p_max * m / (p_max * m + one_minus)
    P'(c) = (1 - q) * P(c) / (1 - P(argmax))        for c != argmax

Every operation in that path is IEEE-754 correctly rounded (+, -, *, /, compare,
float32<->float64) and therefore bit-identical on every conforming platform.
There is no ``log``, ``exp``, ``log2``, ``exp2``, ``pow`` or ``**`` anywhere in
the decision path, and ``test_ddm_rr4_free_corrector_v2.py`` asserts that by
walking this module's AST.

Two further exactness repairs, both of which also remove a real risk:

* The surprise bin was ``int(-log2(one_minus) / 0.5)``.  It is now a comparison
  against a frozen descending threshold table ``2^(-k/2)``.  The partition is
  the same one v1 intended; it is now computed exactly instead of through libm.
  Measured exposure in v1: 6.16% of real positions sit within 1e-12 of a bin
  edge, where an ULP of libm disagreement flips the context index outright.
* The statistics accumulated ``p_max`` in float64 with ``np.add.at``, whose
  summation order is an implementation detail.  They are now int64 counters over
  a fixed-point ``p_max``, so accumulation is exactly order independent.

WHAT v2 DOES NOT CHANGE.  It is still free under contest rule 118: the shift is
estimated online from already-decoded symbols by a fixed generic rule, nothing
is transmitted, and the constants are the same first-principles values ddm_rr2
froze - a Krichevsky-Trofimov smoother, a power-of-two bin width, a cold-context
floor, a symmetric odds clamp.  None was swept against the scored clip.  The
threshold table is ``2^(-k/2)``, a mathematical constant sequence, not data.

DISTORTION.  None, by construction.  Only the probability model handed to the
arithmetic coder changes; the decoded token field is bit-identical.

This file must stay dependency-free apart from numpy: it is copied verbatim into
the receiver runtime, exactly as ddm_rr2's was.
"""

from __future__ import annotations

import math

import numpy as np

# --- generic constants (first principles; never swept against the clip) ------
# Identical to ddm_rr2's set; DELTA_CLIP is expressed as its odds bound because
# v2 never forms the log-odds shift itself.

NUM_CLASSES = 5
U_STEP = 0.5  # width of a surprise bin, in bits of -log2(1 - p_max)
U_BINS = 64  # saturates the surprise axis at 32 bits
RUN_LEVELS = 8  # unchanged-run length, saturated
RUN_CAP = 255  # run counter saturation, so the state stays uint8-sized
BOUNDARY_LEVELS = 5  # the RCF1 boundary bucket alphabet
KT_ALPHA = 0.5  # Krichevsky-Trofimov smoothing
MIN_COUNT = 32  # below this a context emits m = 1, i.e. exactly HPAC
DELTA_CLIP = 4.0  # symmetric log2-odds clamp; as odds, [2^-4, 2^4]
ODDS_LOW = 0.0625  # 2^-4 exactly
ODDS_HIGH = 16.0  # 2^+4 exactly
PROB_EPS = 1e-9
PHAT_SCALE = 1 << 30  # fixed-point resolution for the p_max accumulator

CONTEXT_SIZE = NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS

# 1/sqrt(2) is needed for the odd half-bit thresholds.  IEEE-754 requires sqrt to
# be correctly rounded, unlike log/exp, so this is platform-exact - but the value
# is pinned to its exact bit pattern so a non-conforming platform fails closed
# rather than silently emitting a different stream.
_INV_SQRT2 = math.sqrt(0.5)
_INV_SQRT2_BITS = 0x3FE6A09E667F3BCD
if int.from_bytes(np.float64(_INV_SQRT2).tobytes(), "little") != _INV_SQRT2_BITS:
    raise RuntimeError("platform sqrt(0.5) is not correctly rounded; refusing to encode")


def _surprise_thresholds() -> np.ndarray:
    """Descending table of ``2^(-k/2)`` for k = 1 .. U_BINS-1, exactly.

    ``ubin >= k`` holds exactly when ``one_minus <= 2^(-k/2)``, so comparing
    against this table reproduces ``floor(-log2(one_minus) / U_STEP)`` without
    ever evaluating a logarithm.  Even k gives an exact power of two; odd k is
    that scaled by the correctly rounded 1/sqrt(2) above.
    """
    table = np.empty(U_BINS - 1, dtype=np.float64)
    for k in range(1, U_BINS):
        value = math.ldexp(1.0, -(k // 2))
        if k % 2:
            value *= _INV_SQRT2
        table[k - 1] = value
    return table


# Ascending order, so np.searchsorted can count how many thresholds a value is
# below in one pass.  Built once at import; identical on every platform.
_SURPRISE_ASC = _surprise_thresholds()[::-1].copy()


class GroupState:
    """Per-group quantities the corrector derives once and reuses."""

    __slots__ = ("arg", "context", "one_minus", "p_max", "p_max_q", "row64")

    def __init__(
        self,
        row64: np.ndarray,
        arg: np.ndarray,
        p_max: np.ndarray,
        one_minus: np.ndarray,
        p_max_q: np.ndarray,
        context: np.ndarray,
    ) -> None:
        self.row64 = row64
        self.arg = arg
        self.p_max = p_max
        self.one_minus = one_minus
        self.p_max_q = p_max_q
        self.context = context


class FreeCorrector:
    """Adaptive hit-event corrector shared by the encoder and the receiver.

    The caller drives it in the shipped decode order::

        corrector.begin_frame(boundary_flat)
        for each causal group:
            state = corrector.group_state(probability, predicted, positions)
            coding_row = corrector.coding_row(state)      # hand to the coder
            corrector.observe(state, symbols)             # after coding
        corrector.end_frame(tokens_flat)

    The API is identical to ``ddm_rr2``'s so the encoder and the receiver swap
    by changing one import.
    """

    __slots__ = ("boundary", "counts", "have_prev", "hits", "phat_q", "plane", "prev1", "prev2", "run")

    def __init__(self, plane: int) -> None:
        self.plane = int(plane)
        self.counts = np.zeros(CONTEXT_SIZE, dtype=np.int64)
        self.hits = np.zeros(CONTEXT_SIZE, dtype=np.int64)
        self.phat_q = np.zeros(CONTEXT_SIZE, dtype=np.int64)
        self.prev1 = np.zeros(self.plane, dtype=np.uint8)
        self.prev2 = np.zeros(self.plane, dtype=np.uint8)
        self.run = np.zeros(self.plane, dtype=np.int64)
        self.have_prev = False
        self.boundary = np.full(self.plane, BOUNDARY_LEVELS - 1, dtype=np.int64)

    # -- driving ------------------------------------------------------------

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        """Pin this frame's boundary buckets (the receiver already has them)."""
        boundary = np.asarray(boundary_flat, dtype=np.int64).reshape(-1)
        if boundary.size != self.plane:
            raise ValueError("boundary bucket plane size mismatch")
        self.boundary = boundary

    def group_state(
        self,
        probability: np.ndarray,
        predicted: np.ndarray,
        positions: np.ndarray,
    ) -> GroupState:
        """Derive the context and the base odds for one causal group.

        ``probability`` is the receiver's own float32 row, unchanged;
        ``predicted`` is the argmax of the pre-table HPAC logits, which is also
        what selects the RCF1 correction cell; ``positions`` are the group's
        flat plane indices.
        """
        row64 = np.asarray(probability, dtype=np.float32).astype(np.float64)
        if row64.ndim != 2 or row64.shape[1] != NUM_CLASSES:
            raise ValueError("probability rows must have shape [n, 5]")
        index = np.arange(row64.shape[0])
        arg = row64.argmax(axis=1)
        p_max = row64[index, arg]
        one_minus = np.maximum(1.0 - p_max, PROB_EPS)

        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        if base_class.size != row64.shape[0] or flat.size != row64.shape[0]:
            raise ValueError("group predicted/positions length mismatch")

        # ubin = floor(-log2(one_minus) / U_STEP), by exact comparison against
        # the frozen 2^(-k/2) table.  ubin >= k holds exactly when
        # one_minus <= 2^(-k/2), so ubin is the number of thresholds that are
        # >= one_minus; searchsorted on the ascending table counts those below,
        # and the complement is that count.
        below = np.searchsorted(_SURPRISE_ASC, one_minus, side="left")
        ubin = np.clip((U_BINS - 1) - below, 0, U_BINS - 1).astype(np.int64)

        if self.have_prev:
            agree1 = (self.prev1[flat].astype(np.int64) == base_class).astype(np.int64)
            agree2 = (self.prev2[flat].astype(np.int64) == base_class).astype(np.int64)
        else:
            agree1 = np.zeros(flat.size, dtype=np.int64)
            agree2 = np.zeros(flat.size, dtype=np.int64)
        run = np.minimum(self.run[flat], RUN_LEVELS - 1)

        head = ((base_class * U_BINS + ubin) * 2 + agree1) * 2 + agree2
        context = (head * RUN_LEVELS + run) * BOUNDARY_LEVELS + self.boundary[flat]

        # Fixed-point p_max for the order-independent accumulator.  Multiplying
        # a float32 by 2^30 is exact, so the rounding here is the only step and
        # it is a single correctly rounded rint.
        p_max_q = np.rint(p_max * PHAT_SCALE).astype(np.int64)
        return GroupState(row64, arg, p_max, one_minus, p_max_q, context)

    def odds_multiplier(self, state: GroupState) -> np.ndarray:
        """``2^delta`` from strictly already-decoded symbols, without any log.

        v1 formed ``delta = log2(e/(1-e)) - log2(x/(1-x))`` and then raised 2 to
        it.  The pair cancels exactly, so the multiplier is a ratio of smoothed
        counts and needs no transcendental at all.
        """
        context = state.context
        count = self.counts[context].astype(np.float64)
        denominator = count + 2.0 * KT_ALPHA

        hit_numerator = self.hits[context].astype(np.float64) + KT_ALPHA
        hit_denominator = denominator - hit_numerator
        expected = self.phat_q[context].astype(np.float64) / PHAT_SCALE
        exp_numerator = expected + KT_ALPHA
        exp_denominator = denominator - exp_numerator

        multiplier = np.ones(context.shape, dtype=np.float64)
        usable = (
            (self.counts[context] >= MIN_COUNT)
            & (hit_numerator > 0.0)
            & (hit_denominator > 0.0)
            & (exp_numerator > 0.0)
            & (exp_denominator > 0.0)
        )
        multiplier[usable] = (
            hit_numerator[usable] * exp_denominator[usable]
        ) / (hit_denominator[usable] * exp_numerator[usable])
        np.clip(multiplier, ODDS_LOW, ODDS_HIGH, out=multiplier)
        return multiplier

    def coding_row(self, state: GroupState) -> np.ndarray:
        """The corrected float32 probability row to hand to the RC64 coder.

        Exactly idempotent where the multiplier is 1.0: those rows are returned
        as the receiver's own bytes, so a cold context emits exactly HPAC, which
        is what the estimator has always claimed to do.
        """
        multiplier = self.odds_multiplier(state)
        row = state.row64.copy()

        active = multiplier != 1.0
        if not np.any(active):
            return row.astype(np.float32)

        p_max = state.p_max[active]
        one_minus = state.one_minus[active]
        shifted = p_max * multiplier[active]
        q = np.clip(shifted / (shifted + one_minus), PROB_EPS, 1.0 - PROB_EPS)
        scale = (1.0 - q) / one_minus
        rows = row[active] * scale[:, None]
        rows[np.arange(rows.shape[0]), state.arg[active]] = q
        row[active] = rows
        return row.astype(np.float32)

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        """Fold this group's decoded symbols into the statistics."""
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        if decoded.size != state.context.size:
            raise ValueError("decoded symbol count does not match the group")
        hit = (decoded == state.arg).astype(np.int64)
        np.add.at(self.counts, state.context, 1)
        np.add.at(self.hits, state.context, hit)
        np.add.at(self.phat_q, state.context, state.p_max_q)

    def end_frame(self, tokens_flat: np.ndarray) -> None:
        """Advance the per-pixel temporal memory once the frame is complete."""
        current = np.asarray(tokens_flat, dtype=np.uint8).reshape(-1)
        if current.size != self.plane:
            raise ValueError("token plane size mismatch")
        if self.have_prev:
            self.run = np.where(current == self.prev1, np.minimum(self.run + 1, RUN_CAP), 0)
            self.prev2 = self.prev1
        self.prev1 = current.copy()
        self.have_prev = True

    # -- crash resumability -------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray | bool]:
        return {
            "counts": self.counts,
            "hits": self.hits,
            "phat_q": self.phat_q,
            "prev1": self.prev1,
            "prev2": self.prev2,
            "run": self.run,
            "have_prev": np.array([self.have_prev], dtype=np.bool_),
        }

    def load_state_dict(self, state: dict) -> None:
        self.counts = np.asarray(state["counts"], dtype=np.int64).copy()
        self.hits = np.asarray(state["hits"], dtype=np.int64).copy()
        self.phat_q = np.asarray(state["phat_q"], dtype=np.int64).copy()
        self.prev1 = np.asarray(state["prev1"], dtype=np.uint8).copy()
        self.prev2 = np.asarray(state["prev2"], dtype=np.uint8).copy()
        self.run = np.asarray(state["run"], dtype=np.int64).copy()
        self.have_prev = bool(np.asarray(state["have_prev"]).reshape(-1)[0])
