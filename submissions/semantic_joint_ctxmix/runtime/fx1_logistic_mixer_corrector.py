"""ddm_fx1 - the FIXED-POINT LOG-ODDS (logistic) mixer, platform-exact by radicals.

WHY THIS EXISTS.  ``ddm_me1`` raced context-mixture architectures over the shipped
``ddm_rr4`` transport law and every one LOST (+67 B to +1,398 B on the full n600
field).  The reason is a theorem, not a mystery: every blend those architectures
could express was a weighted ARITHMETIC MEAN of the odds multipliers, and

    min_k m_k  <=  (sum_k w_k m_k) / (sum_k w_k)  <=  max_k m_k

so a mean can never leave the convex hull of its members.  When one member is
already the best available model -- and ours is, the 51,200-bin joint context of
``ddm_rr4_free_corrector_v2`` -- blending it with weaker members can only drag
the estimate away from the right answer.  Averaging cannot beat its best member.

Real context mixing does not average, it SHARPENS: it adds in the log-odds
domain, so members that agree reinforce PAST the confidence of either one.  That
is a weighted GEOMETRIC mean of the odds multipliers,

    m_mix = prod_k m_k ** w_k                        (weights need NOT sum to 1)

and the classical realisation (PAQ-style logistic mixing) reaches it through
``stretch`` = log-odds and ``squash`` = its inverse, with the weights learned
online by gradient descent on the coding loss.

WHY THAT LOOKED FORBIDDEN, AND WHY IT IS NOT.  ``log``/``exp`` are exactly what
``ddm_rr4_free_corrector_v2`` had to remove: they are libm routines, NOT
correctly rounded, and they differ by an ULP between macOS-arm64 and
Linux-x86_64.  One ULP at p ~ 0.5 moves an RC64 integer frequency by 128 counts,
which desynchronises the arithmetic decoder for the rest of the stream -- the
``ddm_rr2`` refusal at S = 27.83.

The cure here does not reintroduce them.  Restrict the mixing weights to a
FIXED-POINT DYADIC GRID, ``w = W / 2**b`` with ``W`` an integer, and the weighted
geometric mean becomes a RADICAL:

    m ** (W / 2**b)  =  m**k * prod_{i in bits(j)} m ** (1 / 2**(i+1))
                        where  W = k * 2**b + j,  0 <= j < 2**b

Every factor on the right is reachable from ``m`` by repeated SQUARE ROOTS and
MULTIPLICATION.  IEEE-754 REQUIRES ``sqrt`` to be correctly rounded -- it sits in
the same exactness class as ``+``, ``-``, ``*``, ``/``, and NOT in the same class
as ``log``.  So this mixer is bit-identical on every conforming platform while
computing a genuine weighted geometric mean.  There is no ``log``, ``exp``,
``log2``, ``exp2``, ``pow`` or ``**`` on the decision path, and the sister test
asserts that by walking this module's AST, exactly as ``ddm_rr4`` does.

Two consequences worth stating, because they are the controls:

* An INTEGER weight costs no radical at all: ``W = 2**b`` gives ``j = 0`` and
  ``k = 1``, so the result is ``m`` returned BIT-IDENTICALLY.  A single family at
  weight exactly 1.0 therefore collapses onto the shipped law at delta 0.0000
  bits, and any measured difference afterwards is the mixture, never the plumbing.
* All weights at 0 give ``m_mix = 1.0`` exactly, i.e. exactly HPAC.  The
  architecture NESTS both the uncorrected law and the shipped law, so the learner
  starts from a known-good point and a failed learner degrades toward the
  incumbent rather than toward something arbitrary.

THE STRETCH FOR THE GRADIENT.  Descent on the log-loss of the hit event needs
each member's log-odds contribution ``ln m_k``.  The radicals already computed
for the mixing supply it without a logarithm:

    ln(m)  ~  2**b * (m ** (1 / 2**b) - 1)

because ``m**(1/N) = exp(ln m / N) ~ 1 + ln(m)/N``.  At the odds clamp ``m = 16``
with ``b = 6`` this surrogate reads 2.832 against ln 16 = 2.7726, a 2.1%
overshoot at the extreme and far less in the bulk.  It is monotone, exactly
signed, exact arithmetic, and free -- the radical is already in hand.

TWO WEIGHT PRECISIONS, ON PURPOSE.  The learner ACCUMULATES in a fine int64 grid
(``WEIGHT_STORE_BITS``) and the mixer POWERS on a coarse dyadic grid (``b``).
They are separated because they are optimising different things: a single
gradient step is far smaller than 2**-b, so a learner quantised to the power grid
would round every step to zero and never move at all, while a power grid as fine
as the learner would cost one square root per bit.  Fine accumulation, coarse
radicals; both exact integers.

THE UPDATE IS INTEGER, END TO END.  Float summation order is an implementation
detail (pairwise summation depends on array length and SIMD width) -- the same
class of hazard ``ddm_rr4`` removed from its accumulators.  So the per-position
gradient terms are quantised to int64 BEFORE they are summed, and the weights
live in int64.  Integer addition is exactly order independent, so the learner
reproduces on the receiver regardless of how numpy traverses a group.

RULE 118.  Generic decoder code, ZERO transmitted bytes.  The weights are neither
stored nor shipped: they start at a fixed integer, move only under symbols the
receiver has ALREADY decoded, and regenerate exactly on the receiver side.  The
constants are first-principles (a dyadic weight grid, a power-of-two learning
rate, the inherited KT smoother, the inherited symmetric odds clamp).

DISTORTION.  None, by construction.  Only the probability model handed to the
arithmetic coder changes; the decoded token field is bit-identical.

This file stays dependency-free apart from numpy and the shipped constants: it is
copied verbatim into the receiver runtime, exactly as ``ddm_rr4``'s was.
"""

from __future__ import annotations

import numpy as np

from .rr4_free_corrector import (
    BOUNDARY_LEVELS,
    KT_ALPHA,
    MIN_COUNT,
    NUM_CLASSES,
    ODDS_HIGH,
    ODDS_LOW,
    PHAT_SCALE,
    PROB_EPS,
    RUN_LEVELS,
    U_BINS,
    GroupState,
)
from .rr4_free_corrector import (
    FreeCorrector as ShippedFreeCorrector,
)

__all__ = [
    "ERR_SCALE",
    "MIXER_CONTEXTS",
    "SHIPPED_CONFIG",
    "STRETCH_SCALE",
    "WEIGHT_STORE_BITS",
    "FixedPointLogisticMixer",
    "FreeCorrector",
    "MixerFamily",
    "dyadic_power",
    "family_specs",
    "round_shift",
    "stretch_from_radical",
]

HEIGHT = 384
WIDTH = 512
SPATIAL_LEVELS = 5

# --- fixed-point grids (first principles; powers of two, never swept) --------

WEIGHT_STORE_BITS = 20
"""Fractional bits of the accumulating weight.  Fine enough that one gradient
step is representable rather than rounding to zero."""

WEIGHT_STORE_ONE = 1 << WEIGHT_STORE_BITS
"""The stored integer meaning weight exactly 1.0, i.e. the shipped law."""

POWER_BITS = 6
"""Fractional bits of the dyadic grid the radicals realise; also the number of
square roots per member.  1/64 resolution on an already-small correction."""

INT_POWER_BITS = 4
"""Binary-exponentiation depth for the integer part; covers |w| < 16."""

ERR_SCALE = 1 << 20
"""Fixed-point resolution of the (y - q) residual in the weight update."""

STRETCH_SCALE = 1 << 20
"""Fixed-point resolution of the log-odds stretch in the weight update."""

STRETCH_CLAMP = 32 * STRETCH_SCALE
"""Symmetric bound on a member's stretch, so one clipped member cannot dominate.
A count-ratio member realises at most +-2.83 at the inherited odds clamp, so this
never binds for them; it is sized to also admit the BASE member, whose log-odds
reaches about +-21 where HPAC is most confident.  A fail-closed bound, not a
tuned one."""

COUNT_BUCKET_EDGES = np.array(
    [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152],
    dtype=np.int64,
)
"""Dyadic-then-sparser edges for the EVIDENCE-COUNT axis of the weight index.

Optimal shrinkage of an estimate is a function of how much evidence stands behind
it -- that is the whole content of a Krichevsky-Trofimov smoother, and the shipped
law applies its correction at weight exactly 1.0 no matter whether the bin was
observed 32 times or 3 million.  Indexing the mixer weight by the member's OWN
count bucket lets the learner discover the shrinkage curve instead of assuming
one.  Membership is decided by integer comparison against frozen powers of two,
so the bucket is exact and needs no logarithm.  The counts are the ones standing
BEFORE this group is observed, which is exactly what the receiver holds."""


COUNT_HALVING_PASSES = 40
"""Bound on the repeated-halving loop; 40 passes covers any int64 count."""


BASE_MEMBER = "base_odds"
"""The neural prior's own log-odds, offered to the mixer as a member.

Every count-table member can only RECALIBRATE the correction; none of them can
question the HPAC prior itself.  Giving the mixer ``ln(p_max / one_minus)`` as a
member makes ``w`` a learned exponent on the prior's odds, which is exactly
online temperature scaling of the neural model -- estimated from already-decoded
symbols, so still zero transmitted bytes.  Its weight starts at 0, which is
weight 1.0 on the prior, i.e. the shipped law untouched."""

LR_BASE_SHIFT = ERR_SCALE.bit_length() - 1 + STRETCH_SCALE.bit_length() - 1 - WEIGHT_STORE_BITS
"""Shift that turns a quantised ``residual * stretch`` product back into stored
weight units, i.e. learning rate exactly 1.0.  ``lr_shift = LR_BASE_SHIFT + e``
is a learning rate of ``2**-e``."""

WEIGHT_LOW = -4 * WEIGHT_STORE_ONE
WEIGHT_HIGH = 8 * WEIGHT_STORE_ONE
"""Weight clamp.  Wide enough for genuine sharpening, bounded so a transient
cannot run the mixer away; symmetric in spirit with the inherited odds clamp."""


def _assert_sqrt_is_correctly_rounded() -> None:
    """Fail closed on a platform whose float64 sqrt is not IEEE correct.

    The entire exactness argument of this module rests on IEEE-754 REQUIRING
    sqrt to be correctly rounded -- that is the whole reason a radical is
    permitted where a logarithm is not -- so it is checked, not assumed.

    The probes are MATHEMATICAL FACTS, not readings taken from this machine:
    sqrt of an exactly representable square is that square's root exactly, and
    the correctly rounded sqrt(1/2) is pinned by ``ddm_rr4`` at import.  A probe
    table transcribed from whatever the local libm happened to print would
    certify the platform against itself and prove nothing.
    """
    for root in (1.0, 1.5, 2.0, 3.0, 4.0, 1.0625, 65536.0, 1e8):
        squared = np.float64(root) * np.float64(root)
        if np.sqrt(squared) != np.float64(root):
            raise RuntimeError(
                f"platform sqrt is not correctly rounded at {root}; refusing to encode"
            )
    value = np.float64(1.0)
    for _ in range(8):
        value = value / np.float64(4.0)
    for _ in range(16):
        if np.sqrt(value * value) != value:
            raise RuntimeError("platform sqrt is not correctly rounded on a power of four")
        value = value * np.float64(4.0)


def round_shift(value: np.ndarray, bits: int) -> np.ndarray:
    """Arithmetic right shift with round-half-up, as exact integer arithmetic.

    A bare ``>>`` on int64 FLOORS, so every negative update would lose a count.
    Across the 114,000 group updates of one n600 pass that systematic -1 drift
    reaches -0.1 in weight units per member -- a bias masquerading as learning.
    Adding the half before shifting removes it.
    """
    if bits <= 0:
        return value
    return (value + (np.int64(1) << np.int64(bits - 1))) >> np.int64(bits)


def stretch_from_radical(radical: np.ndarray, power_bits: int) -> np.ndarray:
    """``~ln(m)`` recovered from ``m ** (1 / 2**power_bits)``, with no logarithm."""
    return (radical - 1.0) * float(1 << power_bits)


def dyadic_power(
    value: np.ndarray,
    radicals: list[np.ndarray] | None,
    weight: np.ndarray,
    power_bits: int,
) -> np.ndarray:
    """``value ** (weight / 2**power_bits)`` using only ``*``, ``/`` and radicals.

    ``radicals[i]`` must be ``value ** (1 / 2**(i+1))``, i.e. the successive
    square roots; it may be ``None`` only when every weight is an exact multiple
    of ``2**power_bits``.

    Exactness: every operation is IEEE correctly rounded and the traversal order
    is fixed, so the result is bit-identical on every conforming platform.  A
    weight that is an exact multiple of ``2**power_bits`` takes the integer path
    alone, so weight 1.0 returns ``value`` itself, unrounded.
    """
    negative = weight < 0
    magnitude = np.abs(weight)
    integer_part = magnitude >> np.int64(power_bits)
    fraction = magnitude & np.int64((1 << power_bits) - 1)

    shape = np.broadcast_shapes(np.shape(value), np.shape(weight))
    accumulator = np.ones(shape, dtype=np.float64)
    base = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).astype(
        np.float64, copy=True
    )
    remaining = integer_part
    for _ in range(INT_POWER_BITS):
        accumulator = np.where((remaining & 1) == 1, accumulator * base, accumulator)
        base = base * base
        remaining = remaining >> np.int64(1)

    if np.any(fraction != 0):
        if radicals is None:
            raise ValueError("fractional weights require the radicals")
        for index in range(power_bits):
            bit = (fraction >> np.int64(power_bits - 1 - index)) & 1
            accumulator = np.where(bit == 1, accumulator * radicals[index], accumulator)

    return np.where(negative, 1.0 / accumulator, accumulator)


class MixerFamily:
    """One context model: an index rule plus its own online count tables.

    The odds-multiplier arithmetic is IDENTICAL to the shipped
    ``FreeCorrector.odds_multiplier``; only the context rule differs.  Holding the
    estimator fixed is what makes the race a clean test of MIXING, rather than a
    test of two changes at once.
    """

    __slots__ = ("count_limit", "counts", "hits", "name", "phat_q", "rule", "size")

    def __init__(self, name: str, size: int, rule, count_limit: int = 0) -> None:
        self.name = name
        self.size = int(size)
        self.rule = rule
        self.count_limit = int(count_limit)
        self.counts = np.zeros(self.size, dtype=np.int64)
        self.hits = np.zeros(self.size, dtype=np.int64)
        self.phat_q = np.zeros(self.size, dtype=np.int64)

    def multiplier(self, index: np.ndarray) -> np.ndarray:
        count = self.counts[index].astype(np.float64)
        denominator = count + 2.0 * KT_ALPHA
        hit_numerator = self.hits[index].astype(np.float64) + KT_ALPHA
        hit_denominator = denominator - hit_numerator
        expected = self.phat_q[index].astype(np.float64) / PHAT_SCALE
        exp_numerator = expected + KT_ALPHA
        exp_denominator = denominator - exp_numerator

        multiplier = np.ones(index.shape, dtype=np.float64)
        usable = (
            (self.counts[index] >= MIN_COUNT)
            & (hit_numerator > 0.0)
            & (hit_denominator > 0.0)
            & (exp_numerator > 0.0)
            & (exp_denominator > 0.0)
        )
        multiplier[usable] = (hit_numerator[usable] * exp_denominator[usable]) / (
            hit_denominator[usable] * exp_numerator[usable]
        )
        np.clip(multiplier, ODDS_LOW, ODDS_HIGH, out=multiplier)
        return multiplier

    def observe(self, index: np.ndarray, hit: np.ndarray, p_max_q: np.ndarray) -> None:
        np.add.at(self.counts, index, 1)
        np.add.at(self.hits, index, hit)
        np.add.at(self.phat_q, index, p_max_q)
        if self.count_limit:
            # RECENCY.  A cumulative counter over 600 frames never forgets, but a
            # dashcam clip is nonstationary: the scene, the exposure and the road
            # geometry all change under it.  Halving a bin once it saturates makes
            # the estimate an exponential window of about ``count_limit``
            # observations, so a member can track the present instead of averaging
            # the whole clip.  Halving is an exact integer shift, so the receiver
            # reproduces it, and it reads only already-decoded symbols.
            #
            # Halve REPEATEDLY, not once.  One halving per group only bounds the
            # count when a group contributes fewer observations than the limit;
            # a large group would leave the bin permanently above it and the
            # member would silently degrade into a second cumulative copy --
            # racing as a distinct member while being a duplicate.  The loop is
            # bounded because each pass strictly halves.
            touched = np.unique(index)
            for _ in range(COUNT_HALVING_PASSES):
                hot = touched[self.counts[touched] > self.count_limit]
                if not hot.size:
                    break
                self.counts[hot] >>= 1
                self.hits[hot] >>= 1
                self.phat_q[hot] >>= 1


def family_specs() -> dict[str, tuple[int, object]]:
    """The candidate member pool, keyed by name.

    ``shipped_joint`` is the live 51,200-bin law and is always member 0, so every
    architecture in the race NESTS the shipped law and is judged from a position
    it can always fall back to.  The rest are small orthogonal views that
    ``ddm_me1`` measured as unusable UNDER AVERAGING; this arm exists because
    averaging was the wrong verdict surface for them.

    Sizes are deliberately in the thousands, not the millions.  The public
    rank-one competitor spreads 6.17M context slots over 117,964,800 symbols --
    about 19 observations per slot, which is starved.  Ours run 320 to 51,200
    slots, and the shipped joint context was MEASURED at only 0.0516% of
    observations below ``MIN_COUNT``.  Density is the resource being spent here.
    """

    def shipped(f):
        head = ((f["cls"] * U_BINS + f["ubin"]) * 2 + f["agree1"]) * 2 + f["agree2"]
        return (head * RUN_LEVELS + f["run"]) * BOUNDARY_LEVELS + f["boundary"]

    def spatial_surprise(f):
        return (f["cls"] * SPATIAL_LEVELS + f["spatial"]) * U_BINS + f["ubin"]

    def spatial_boundary(f):
        return (f["cls"] * SPATIAL_LEVELS + f["spatial"]) * BOUNDARY_LEVELS + f["boundary"]

    def surprise_only(f):
        return f["cls"] * U_BINS + f["ubin"]

    def temporal_spatial(f):
        head = (f["cls"] * 2 + f["agree1"]) * 2 + f["agree2"]
        return head * SPATIAL_LEVELS + f["spatial"]

    def run_surprise(f):
        return (f["cls"] * RUN_LEVELS + f["run"]) * U_BINS + f["ubin"]

    def boundary_surprise(f):
        return (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * U_BINS + f["ubin"]

    def temporal_surprise(f):
        head = (f["cls"] * 2 + f["agree1"]) * 2 + f["agree2"]
        return head * U_BINS + f["ubin"]

    def unused(f):
        return np.zeros(f["cls"].shape, dtype=np.int64)

    joint_size = NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS
    return {
        "shipped_joint": (joint_size, shipped),
        BASE_MEMBER: (1, unused),
        "shipped_fast256": (joint_size, shipped, 256),
        "shipped_fast4096": (joint_size, shipped, 4096),
        "surprise_fast256": (NUM_CLASSES * U_BINS, surprise_only, 256),
        "spatial_surprise": (NUM_CLASSES * SPATIAL_LEVELS * U_BINS, spatial_surprise),
        "spatial_boundary": (NUM_CLASSES * SPATIAL_LEVELS * BOUNDARY_LEVELS, spatial_boundary),
        "surprise_only": (NUM_CLASSES * U_BINS, surprise_only),
        "temporal_spatial": (NUM_CLASSES * 2 * 2 * SPATIAL_LEVELS, temporal_spatial),
        "run_surprise": (NUM_CLASSES * RUN_LEVELS * U_BINS, run_surprise),
        "boundary_surprise": (NUM_CLASSES * BOUNDARY_LEVELS * U_BINS, boundary_surprise),
        "temporal_surprise": (NUM_CLASSES * 2 * 2 * U_BINS, temporal_surprise),
    }


MIXER_CONTEXTS: dict[str, tuple[int, object]] = {
    "none": (1, lambda f: np.zeros(f["cls"].shape, dtype=np.int64)),
    "cls": (NUM_CLASSES, lambda f: f["cls"]),
    "boundary": (BOUNDARY_LEVELS, lambda f: f["boundary"]),
    "cls_boundary": (
        NUM_CLASSES * BOUNDARY_LEVELS,
        lambda f: f["cls"] * BOUNDARY_LEVELS + f["boundary"],
    ),
    "ubin8": (8, lambda f: np.minimum(f["ubin"] >> 3, 7)),
    "cls_ubin8": (NUM_CLASSES * 8, lambda f: f["cls"] * 8 + np.minimum(f["ubin"] >> 3, 7)),
    "cls_boundary_ubin8": (
        NUM_CLASSES * BOUNDARY_LEVELS * 8,
        lambda f: (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * 8
        + np.minimum(f["ubin"] >> 3, 7),
    ),
    "cls_agree_ubin8": (
        NUM_CLASSES * 4 * 8,
        lambda f: (f["cls"] * 4 + f["agree1"] * 2 + f["agree2"]) * 8
        + np.minimum(f["ubin"] >> 3, 7),
    ),
    "cls_run_ubin8": (
        NUM_CLASSES * RUN_LEVELS * 8,
        lambda f: (f["cls"] * RUN_LEVELS + f["run"]) * 8 + np.minimum(f["ubin"] >> 3, 7),
    ),
    "cls_boundary_agree_ubin8": (
        NUM_CLASSES * BOUNDARY_LEVELS * 4 * 8,
        lambda f: ((f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * 4 + f["agree1"] * 2 + f["agree2"])
        * 8
        + np.minimum(f["ubin"] >> 3, 7),
    ),
}
"""Weight-set selectors.  A mixer context lets the learner apply a DIFFERENT
shrinkage where the evidence has a different character -- the standard PAQ device
-- while keeping each weight set densely updated."""


class FixedPointLogisticMixer(ShippedFreeCorrector):
    """The shipped transport driven by a fixed-point weighted geometric blend."""

    def __init__(
        self,
        plane: int,
        *,
        families: tuple[str, ...] = ("shipped_joint",),
        mixer_context: str = "none",
        count_buckets: int = 1,
        power_bits: int = POWER_BITS,
        learn: bool = True,
        lr_shift: int = LR_BASE_SHIFT + 4,
        normalize: bool = True,
        initial_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(plane)
        if plane != HEIGHT * WIDTH:
            raise ValueError("the mixer assumes the shipped 384x512 plane")
        if not 1 <= power_bits <= 16:
            raise ValueError("power_bits must be in [1, 16]")
        _assert_sqrt_is_correctly_rounded()

        specs = family_specs()
        if not families:
            raise ValueError("at least one family is required")
        missing = [name for name in families if name not in specs]
        if missing:
            raise ValueError(f"unknown families: {missing}")
        self.families = [MixerFamily(name, *specs[name]) for name in families]

        if mixer_context not in MIXER_CONTEXTS:
            raise ValueError(f"unknown mixer context {mixer_context!r}")
        self.mixer_context_name = mixer_context
        self.n_mixer_contexts, self._mixer_rule = MIXER_CONTEXTS[mixer_context]
        if not 1 <= count_buckets <= len(COUNT_BUCKET_EDGES) + 1:
            raise ValueError("count_buckets out of range")
        self.count_buckets = int(count_buckets)
        self._count_edges = COUNT_BUCKET_EDGES[: self.count_buckets - 1]
        self.n_weight_sets = self.n_mixer_contexts * self.count_buckets

        self.power_bits = int(power_bits)
        self.store_shift = WEIGHT_STORE_BITS - self.power_bits
        self.learn = bool(learn)
        self.lr_shift = int(lr_shift)
        self.normalize = bool(normalize)

        # Member 0 starts at weight 1.0 so the mixer BEGINS at exactly the
        # shipped law; every other member starts at 0, contributing nothing.  The
        # learner has to earn every byte away from a known-good point, and a
        # failed learner degrades toward the incumbent rather than to noise.
        if initial_weights is None:
            # 1.0 on the shipped law, 0.0 on everything else: the mixer BEGINS
            # at the incumbent regardless of which members are enlisted.
            initial = [1.0 if f.name == "shipped_joint" else 0.0 for f in self.families]
        else:
            if len(initial_weights) != len(self.families):
                raise ValueError("initial_weights must match the family count")
            initial = list(initial_weights)
        self.weights = np.zeros((self.n_weight_sets, len(self.families)), dtype=np.int64)
        for position, value in enumerate(initial):
            self.weights[:, position] = np.int64(round(value * WEIGHT_STORE_ONE))

        self.current = np.zeros(plane, dtype=np.uint8)
        self.known = np.zeros(plane, dtype=bool)
        self._pending: dict | None = None

    # -- driving ------------------------------------------------------------

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        super().begin_frame(boundary_flat)
        self.known[:] = False
        self.current[:] = 0

    def _spatial_level(self, flat: np.ndarray, base_class: np.ndarray) -> np.ndarray:
        """Causal-spatial agreement level from already-decoded left/up pixels."""
        x = flat % WIDTH
        y = flat // WIDTH
        left = np.maximum(flat - 1, 0)
        up = np.maximum(flat - WIDTH, 0)
        has_left = (x > 0) & self.known[left]
        has_up = (y > 0) & self.known[up]
        agree_left = has_left & (self.current[left].astype(np.int64) == base_class)
        agree_up = has_up & (self.current[up].astype(np.int64) == base_class)
        available = has_left.astype(np.int64) + has_up.astype(np.int64)
        agreeing = agree_left.astype(np.int64) + agree_up.astype(np.int64)
        return np.where(available == 0, 0, agreeing + 1).astype(np.int64)

    def group_state(
        self, probability: np.ndarray, predicted: np.ndarray, positions: np.ndarray
    ) -> GroupState:
        state = super().group_state(probability, predicted, positions)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)

        # Unpack the sub-features the shipped context already packed, so the
        # families recombine them without recomputing the surprise bin.
        packed = state.context
        boundary = packed % BOUNDARY_LEVELS
        rest = packed // BOUNDARY_LEVELS
        run = rest % RUN_LEVELS
        rest //= RUN_LEVELS
        agree2 = rest % 2
        rest //= 2
        agree1 = rest % 2
        rest //= 2
        ubin = rest % U_BINS
        cls = rest // U_BINS

        features = {
            "cls": cls,
            "ubin": ubin,
            "agree1": agree1,
            "agree2": agree2,
            "run": run,
            "boundary": boundary,
            "spatial": self._spatial_level(flat, base_class),
        }
        self._pending = {
            "flat": flat,
            "indices": [family.rule(features) for family in self.families],
            "mixer": np.asarray(self._mixer_rule(features), dtype=np.int64).reshape(-1),
        }
        return state

    # -- the mixture --------------------------------------------------------

    def odds_multiplier(self, state: GroupState) -> np.ndarray:
        """Weighted GEOMETRIC mean of the members' odds multipliers.

        The weighted sum in log-odds that ``ddm_me1`` proved an arithmetic mean
        can never express, realised through radicals so it stays platform-exact.
        """
        if self._pending is None:
            raise RuntimeError("odds_multiplier() called without a group_state()")
        mixer = self._pending["mixer"]
        blended = np.ones(state.context.shape, dtype=np.float64)
        stretches: list[np.ndarray] = []
        weight_indices: list[np.ndarray] = []

        for position, (family, index) in enumerate(
            zip(self.families, self._pending["indices"], strict=True)
        ):
            multiplier = (
                state.p_max / state.one_minus
                if family.name == BASE_MEMBER
                else family.multiplier(index)
            )
            if self.count_buckets > 1 and family.name != BASE_MEMBER:
                bucket = np.searchsorted(self._count_edges, family.counts[index], side="right")
                weight_index = mixer * self.count_buckets + bucket
            else:
                weight_index = mixer * self.count_buckets
            weight_indices.append(weight_index)
            grid_weight = round_shift(self.weights[weight_index, position], self.store_shift)

            # The radicals cost one sqrt per bit, so they are taken only when
            # something actually needs them: a fractional weight, or the learner.
            need = self.learn or bool(
                np.any((grid_weight & np.int64((1 << self.power_bits) - 1)) != 0)
            )
            radicals: list[np.ndarray] | None = None
            if need:
                radicals = []
                root = multiplier
                for _ in range(self.power_bits):
                    root = np.sqrt(root)
                    radicals.append(root)
            # Append ONE entry per member unconditionally, even when it is None.
            # A conditional append would let ``stretches`` and ``weight_indices``
            # fall out of step, and the learner would then apply member j's
            # gradient to member j+1's weight -- silently, with every exactness
            # and determinism test still passing.  Positional alignment across
            # these lists is load-bearing, so it is made structural.
            stretches.append(
                stretch_from_radical(radicals[-1], self.power_bits) if radicals else None
            )

            blended = blended * dyadic_power(multiplier, radicals, grid_weight, self.power_bits)

        np.clip(blended, ODDS_LOW, ODDS_HIGH, out=blended)
        self._pending["stretches"] = stretches
        self._pending["weight_indices"] = weight_indices
        return blended

    def coding_row(self, state: GroupState) -> np.ndarray:
        """The corrected float32 row, plus the hit probability the learner needs."""
        multiplier = self.odds_multiplier(state)
        row = state.row64.copy()

        # q is needed by the gradient at EVERY position, including those the
        # transport leaves untouched, so it is formed for the whole group.
        shifted = state.p_max * multiplier
        q = np.clip(shifted / (shifted + state.one_minus), PROB_EPS, 1.0 - PROB_EPS)
        if self._pending is not None:
            self._pending["q"] = q

        active = multiplier != 1.0
        if np.any(active):
            scale = (1.0 - q[active]) / state.one_minus[active]
            rows = row[active] * scale[:, None]
            rows[np.arange(rows.shape[0]), state.arg[active]] = q[active]
            row[active] = rows
        return row.astype(np.float32)

    # -- the learner --------------------------------------------------------

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called without a group_state()")
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        hit = (decoded == state.arg).astype(np.int64)

        if self.learn and "q" in self._pending:
            self._update_weights(hit, self._pending["q"], self._pending["mixer"])

        for family, index in zip(self.families, self._pending["indices"], strict=True):
            if family.name != BASE_MEMBER:
                family.observe(index, hit, state.p_max_q)

        flat = self._pending["flat"]
        self.current[flat] = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        self.known[flat] = True
        self._pending = None

    def _update_weights(self, hit: np.ndarray, q: np.ndarray, mixer: np.ndarray) -> None:
        """One integer gradient step on the log-loss of the hit event.

        For the logistic model ``d/dw_k -[y ln q + (1-y) ln(1-q)] = -(y-q)*s_k``,
        so descent adds ``lr * (y - q) * s_k``.

        Everything is quantised to int64 BEFORE the group sum: float summation
        order is an implementation detail, and this learner must reproduce
        exactly on the receiver whatever order numpy chooses within a group.
        """
        assert self._pending is not None
        stretches = self._pending.get("stretches")
        if not stretches:
            return
        residual = np.rint((hit.astype(np.float64) - q) * ERR_SCALE).astype(np.int64)
        weight_indices = self._pending["weight_indices"]
        if len(stretches) != len(weight_indices) or len(stretches) != len(self.families):
            raise RuntimeError(
                "mixer bookkeeping fell out of step: "
                f"{len(stretches)} stretches, {len(weight_indices)} indices, "
                f"{len(self.families)} members"
            )

        for position, stretch in enumerate(stretches):
            if stretch is None:
                continue
            index = weight_indices[position]
            counts = np.bincount(index, minlength=self.n_weight_sets).astype(np.int64)
            quantised = np.clip(
                np.rint(stretch * STRETCH_SCALE), -STRETCH_CLAMP, STRETCH_CLAMP
            ).astype(np.int64)
            gradient = np.zeros(self.n_weight_sets, dtype=np.int64)
            np.add.at(gradient, index, residual * quantised)
            if self.normalize:
                gradient = np.where(counts > 0, gradient // np.maximum(counts, 1), 0)
            step = round_shift(gradient, self.lr_shift)
            self.weights[:, position] = np.clip(
                self.weights[:, position] + step, WEIGHT_LOW, WEIGHT_HIGH
            )

    # -- observability ------------------------------------------------------

    def weight_report(self) -> dict[str, list[float]]:
        """Learned weights as floats, for the memo.  Never read by the decoder."""
        scale = float(WEIGHT_STORE_ONE)
        return {
            family.name: [float(value) / scale for value in self.weights[:, position]]
            for position, family in enumerate(self.families)
        }


# --- the frozen shipped configuration ---------------------------------------

SHIPPED_CONFIG: dict = {
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
    ),
    "mixer_context": "cls_boundary_agree_ubin8",
    "count_buckets": 1,
    "power_bits": POWER_BITS,
    "lr_shift": LR_BASE_SHIFT + 4,
    "normalize": True,
    "learn": True,
}
"""The configuration this arm measured at -560.07 B on the full n600 field.

It is FROZEN here rather than passed in because a decoder takes no arguments: the
receiver must reconstruct the identical model from the archive and its own code
alone.  Anything configurable at encode time but not at decode time is a
desynchronisation waiting to happen, so the encoder reads the same frozen dict.
"""


class FreeCorrector(FixedPointLogisticMixer):
    """Drop-in for ``ddm_rr4_free_corrector_v2.FreeCorrector``, frozen config.

    The encoder selects a corrector by module name and constructs it as
    ``FreeCorrector(plane)`` (see ``ddm_rr2_encoder_byteclose``), so exposing this
    name with no extra arguments is what makes the mixer byte-closeable without
    touching the build chain.
    """

    def __init__(self, plane: int) -> None:
        super().__init__(plane, **SHIPPED_CONFIG)
