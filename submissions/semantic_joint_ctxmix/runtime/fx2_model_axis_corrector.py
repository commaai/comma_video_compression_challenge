"""ddm_fx2 - the probability-MODEL axis: richer causal geometry + the SSE/APM stage.

WHAT THIS ADDS TO ``ddm_fx1``, AND WHY EACH ADDITION IS FREE.

``ddm_fx1`` proved the probability-MODEL axis is live: a fixed-point weighted
GEOMETRIC mean of odds multipliers took -560.07 B off the n600 token field at a
bit-identical decoded token field.  Its own ``NEXT_IF_RESUMED`` named two
unstarted items, and this module is both of them plus the context-discovery axis
the operator asked for.  Every mechanism here is generic receiver code reading
only ALREADY-DECODED symbols, so all of it is rule-118 free: zero transmitted
bytes, no learned table shipped, no video-derived constant.

1. THE CAUSAL TEMPLATE WAS UNDER-USED (measured, not assumed).

``ddm_fx1._spatial_level`` consults the LEFT and UP pixels only.  That was a
conservative choice made without measuring the decode order.  The shipped
wavefront is 190 causal groups driven by ``group_index.u8``, and a direct probe
of that file (``ddm_fx2`` R1) recovers the rule exactly:

    group(x, y) = (x & 63) + 2 * (y & 63)

from which the causality of any offset follows by arithmetic rather than by
hope.  A neighbour is already decoded exactly when its group index is strictly
smaller.  For the UP-RIGHT neighbour ``(x+1, y-1)``:

    x & 63 != 63 :  g = (x&63) + 1 + 2*(y&63) - 2  =  g(p) - 1   < g(p)
    x & 63 == 63 :  g =        0 + 2*(y&63) - 2    =  g(p) - 65  < g(p)

so it is causal whenever ``y & 63 != 0`` -- which is EXACTLY the condition for
the UP neighbour, and strictly weaker than the condition for LEFT.  Measured on
the real plane: up 98.6945%, up-right 98.6945%, left 98.6301% of in-bounds
positions.  **The up-right neighbour was always as available as up.**  The whole
upper-right quadrant follows the same way (up-left 97.34%, (+2,-2) 97.38%).

So the template widens to four neighbours.  Correctness does not rest on that
arithmetic: every read is still gated by the runtime ``self.known`` mask, exactly
as ``ddm_fx1`` gated its two, so a neighbour that is NOT yet decoded contributes
nothing rather than contributing garbage.  The arithmetic explains why the wider
template is worth its cost; the mask is what makes it safe.

2. LOCAL HOMOGENEITY IS A BOUNDARY DETECTOR THE MODEL DID NOT HAVE.

Every existing spatial feature asks "do my neighbours agree with the PREDICTION".
None asks "do my neighbours agree with EACH OTHER".  Those are different
questions, and the second one is the geometric one: the argmax field is
piecewise constant, so a position whose decoded neighbours are unanimous sits in
a cell interior and a position whose decoded neighbours disagree sits on a
codim-1 cell boundary.  That is where the misses concentrate.  ``_neighbour_
homogeneity`` counts the DISTINCT classes among the available causal neighbours
and is computed entirely from decoded state -- orthogonal to the prediction, and
free.

3. THE SSE / APM SECOND STAGE (the classic stage the fx1 build did not have).

Mixing combines members.  It cannot fix a bias that survives the mix: if the
blended estimate says 0.97 in some regime and the truth is 0.95 there, no
reweighting of the members repairs it, because every member is already telling
the same story.  The PAQ answer is a second stage indexed by the mixer's OWN
OUTPUT -- Secondary Symbol Estimation -- an adaptive map from "what the model
predicted" to "what actually happened".

The realisation here reuses the shipped machinery so it inherits its exactness:

* the bin is ``ddm_rr4``'s frozen ``2^(-k/2)`` threshold table applied to
  ``1 - q_mix``.  That is a pure comparison ladder, so it is exact, and it
  spaces the bins logarithmically in the tail, which is what an APM wants and
  is normally obtained from ``stretch`` = log-odds.  We get the spacing without
  the logarithm.
* the correction is the SAME Krichevsky-Trofimov count ratio the members use,
  but with the accumulated expectation being ``q_mix`` rather than ``p_max``.
  It therefore reads as odds(observed hit rate) / odds(mean predicted q), which
  multiplied into the odds of ``q_mix`` is precisely the calibrated estimate.

Two nesting properties, and they are the controls: a cold SSE bin is below
``MIN_COUNT`` so its multiplier is EXACTLY 1.0, and multiplying by 1.0 is
bit-identical; and a perfectly calibrated bin has hit rate equal to mean ``q``,
which drives the ratio to 1.0 as well.  **A useless SSE stage costs exactly zero
bits, bit-for-bit** -- it cannot degrade the incumbent, only decline to help.

4. STATISTICAL CONTEXT DISCOVERY.

The mixer contexts added here were not hand-guessed.  They were RANKED first, by
a hindsight-optimal instrument that measures, for a candidate partition, the code
length achievable if each cell were given its best dyadic weight -- an upper
bound on what any online learner can extract from that partition, computable in
one pass without running a race.  The partition that wins is then written here as
generic integer code.  Only the STRUCTURE crosses from the analysis to the
receiver (a handful of integers in this source file); no table, no centroid, no
video-derived value is transmitted, so the rate cost is zero.  That this is
design-time information transfer measured on the scored clip is stated plainly in
the memo, exactly as ``ddm_fx1`` section 7 stated it for its own selection.

EXACTNESS.  Unchanged from ``ddm_fx1``: no ``log``, ``exp``, ``log2``, ``exp2``,
``pow`` or ``**`` on the decision path; the sister test walks this module's AST
and refuses one.  Every added operation is integer comparison, integer counting,
or a correctly rounded ``+ - * / rint sqrt``.

DISTORTION.  None, by construction.  Only the probability law changes; the
decoded token field is bit-identical, so d_seg and d_pose cannot move.
"""

from __future__ import annotations

import numpy as np

from .fx1_logistic_mixer_corrector import (
    MIXER_CONTEXTS,
    WEIGHT_STORE_ONE,
    FixedPointLogisticMixer,
    MixerFamily,
    dyadic_power,
    round_shift,
    stretch_from_radical,
)
from .rr4_free_corrector import (
    _SURPRISE_ASC,
    BOUNDARY_LEVELS,
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
    "HOMOGENEITY_LEVELS",
    "SHIPPED_CONFIG",
    "SPATIAL4_LEVELS",
    "SSE_CONTEXTS",
    "FreeCorrector",
    "Fx2ModelAxisMixer",
    "fx2_family_specs",
    "fx2_mixer_contexts",
]

HEIGHT = 384
WIDTH = 512

# --- the widened causal template (R1) ----------------------------------------

CAUSAL_OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (0, -1), (1, -1), (-1, -1))
"""(dx, dy) neighbours consulted by the widened spatial features.

Ordered by measured causality on the real ``group_index.u8``: up 98.6945%,
up-right 98.6945%, left 98.6301%, up-left 97.3425% of in-bounds positions.  All
four are read through the ``known`` mask, so the ordering is an efficiency
statement, not a correctness one.
"""

GROUP_BINS = 8
"""ddm_gb1: bins of the 190-step within-tile decode index.  Eight, because
``ddm_mi1`` measured the exact 190-level index OVERFITTING (51.25 B held-out
against 104.37 in-sample) while eight bins held 64.20 / 65.77."""

SPATIAL4_LEVELS = 6
"""``0`` = no causal neighbour yet; otherwise ``1 + (how many of the available
causal neighbours carry the predicted class)``, saturating at 5."""

HOMOGENEITY_LEVELS = 5
"""``0`` = no causal neighbour yet; otherwise the number of DISTINCT decoded
classes among the available causal neighbours, capped at 4.  ``1`` is a cell
interior, ``>= 2`` is a decoded-field boundary."""


# --- the SSE / APM stage (R2) ------------------------------------------------

SSE_BINS = U_BINS
"""Bins of the mixer's own output probability, on ``ddm_rr4``'s frozen
``2^(-k/2)`` ladder applied to ``1 - q``.  Log-spaced in the confident tail
without a logarithm, which is where 70% of this stream's bits live."""

SSE_CONTEXTS: dict[str, tuple[int, object]] = {
    "off": (0, None),
    "qbin": (SSE_BINS, lambda f: f["qbin"]),
    "cls_qbin": (NUM_CLASSES * SSE_BINS, lambda f: f["cls"] * SSE_BINS + f["qbin"]),
    "bnd_qbin": (
        BOUNDARY_LEVELS * SSE_BINS,
        lambda f: f["boundary"] * SSE_BINS + f["qbin"],
    ),
    "cls_bnd_qbin": (
        NUM_CLASSES * BOUNDARY_LEVELS * SSE_BINS,
        lambda f: (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * SSE_BINS + f["qbin"],
    ),
    "cls_homog_qbin": (
        NUM_CLASSES * HOMOGENEITY_LEVELS * SSE_BINS,
        lambda f: (f["cls"] * HOMOGENEITY_LEVELS + f["homog"]) * SSE_BINS + f["qbin"],
    ),
}
"""Weight-set selectors for the second stage.  ``off`` disables it entirely and
is the control: with ``off`` this class must reproduce ``ddm_fx1`` bit-for-bit."""


def fx2_family_specs() -> dict[str, tuple]:
    """``ddm_fx1``'s member pool plus the members the widened template unlocks.

    The estimator inside every member is unchanged -- the same KT count ratio --
    so a race between these and the inherited members is a clean test of the
    CONTEXT, never of two changes at once.  That discipline is what made the fx1
    table readable and it is preserved here.
    """
    from .fx1_logistic_mixer_corrector import family_specs

    specs = dict(family_specs())

    def spatial4_surprise(f):
        return (f["cls"] * SPATIAL4_LEVELS + f["spatial4"]) * U_BINS + f["ubin"]

    def spatial4_boundary(f):
        return (f["cls"] * SPATIAL4_LEVELS + f["spatial4"]) * BOUNDARY_LEVELS + f["boundary"]

    def homog_surprise(f):
        return (f["cls"] * HOMOGENEITY_LEVELS + f["homog"]) * U_BINS + f["ubin"]

    def homog_spatial4(f):
        return (f["cls"] * HOMOGENEITY_LEVELS + f["homog"]) * SPATIAL4_LEVELS + f["spatial4"]

    def homog_boundary_surprise(f):
        head = (f["cls"] * HOMOGENEITY_LEVELS + f["homog"]) * BOUNDARY_LEVELS + f["boundary"]
        return head * U_BINS + f["ubin"]

    def spatial4_temporal(f):
        head = (f["cls"] * 2 + f["agree1"]) * 2 + f["agree2"]
        return head * SPATIAL4_LEVELS + f["spatial4"]

    def patch192_only(f):
        return f["patch192"]

    specs.update({"patch192_only": (192, patch192_only)})

    def tile48_groupbin8(f):
        return f["tile48"] * GROUP_BINS + f["groupbin8"]

    specs.update({"tile48_groupbin8": (48 * GROUP_BINS, tile48_groupbin8)})

    def groupbin8_only(f):
        return f["groupbin8"]

    def cls_groupbin8(f):
        return f["cls"] * GROUP_BINS + f["groupbin8"]

    def groupbin8_surprise(f):
        return (f["cls"] * GROUP_BINS + f["groupbin8"]) * U_BINS + f["ubin"]

    specs.update(
        {
            "groupbin8_only": (GROUP_BINS, groupbin8_only),
            "cls_groupbin8": (NUM_CLASSES * GROUP_BINS, cls_groupbin8),
            "groupbin8_surprise": (
                NUM_CLASSES * GROUP_BINS * U_BINS,
                groupbin8_surprise,
            ),
        }
    )

    specs.update(
        {
            "spatial4_surprise": (NUM_CLASSES * SPATIAL4_LEVELS * U_BINS, spatial4_surprise),
            "spatial4_boundary": (
                NUM_CLASSES * SPATIAL4_LEVELS * BOUNDARY_LEVELS,
                spatial4_boundary,
            ),
            "homog_surprise": (NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS, homog_surprise),
            "homog_spatial4": (
                NUM_CLASSES * HOMOGENEITY_LEVELS * SPATIAL4_LEVELS,
                homog_spatial4,
            ),
            "homog_boundary_surprise": (
                NUM_CLASSES * HOMOGENEITY_LEVELS * BOUNDARY_LEVELS * U_BINS,
                homog_boundary_surprise,
            ),
            "spatial4_temporal": (NUM_CLASSES * 2 * 2 * SPATIAL4_LEVELS, spatial4_temporal),
            "spatial4_surprise_fast256": (
                NUM_CLASSES * SPATIAL4_LEVELS * U_BINS,
                spatial4_surprise,
                256,
            ),
            "homog_surprise_fast256": (
                NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS,
                homog_surprise,
                256,
            ),
        }
    )
    return specs


def fx2_mixer_contexts() -> dict[str, tuple[int, object]]:
    """``ddm_fx1``'s weight-set selectors plus the ones R3 ranked into the lead."""
    contexts = dict(MIXER_CONTEXTS)
    contexts.update(
        {
            "cls_homog_ubin8": (
                NUM_CLASSES * HOMOGENEITY_LEVELS * 8,
                lambda f: (f["cls"] * HOMOGENEITY_LEVELS + f["homog"]) * 8
                + np.minimum(f["ubin"] >> 3, 7),
            ),
            "cls_spatial4_ubin8": (
                NUM_CLASSES * SPATIAL4_LEVELS * 8,
                lambda f: (f["cls"] * SPATIAL4_LEVELS + f["spatial4"]) * 8
                + np.minimum(f["ubin"] >> 3, 7),
            ),
            "cls_boundary_homog_ubin8": (
                NUM_CLASSES * BOUNDARY_LEVELS * HOMOGENEITY_LEVELS * 8,
                lambda f: (
                    (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * HOMOGENEITY_LEVELS + f["homog"]
                )
                * 8
                + np.minimum(f["ubin"] >> 3, 7),
            ),
            "cls_boundary_agree_ubin16": (
                NUM_CLASSES * BOUNDARY_LEVELS * 4 * 16,
                lambda f: (
                    (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * 4 + f["agree1"] * 2 + f["agree2"]
                )
                * 16
                + np.minimum(f["ubin"] >> 2, 15),
            ),
            "cls_boundary_agree_homog_ubin8": (
                NUM_CLASSES * BOUNDARY_LEVELS * 4 * HOMOGENEITY_LEVELS * 8,
                lambda f: (
                    (
                        (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * 4
                        + f["agree1"] * 2
                        + f["agree2"]
                    )
                    * HOMOGENEITY_LEVELS
                    + f["homog"]
                )
                * 8
                + np.minimum(f["ubin"] >> 3, 7),
            ),
            "cls_boundary_agree_spatial4_ubin8": (
                NUM_CLASSES * BOUNDARY_LEVELS * 4 * SPATIAL4_LEVELS * 8,
                lambda f: (
                    (
                        (f["cls"] * BOUNDARY_LEVELS + f["boundary"]) * 4
                        + f["agree1"] * 2
                        + f["agree2"]
                    )
                    * SPATIAL4_LEVELS
                    + f["spatial4"]
                )
                * 8
                + np.minimum(f["ubin"] >> 3, 7),
            ),
        }
    )
    return contexts


class Fx2ModelAxisMixer(FixedPointLogisticMixer):
    """``ddm_fx1``'s mixer with the widened causal template and the SSE stage.

    With ``sse_context='off'`` and an inherited member set and mixer context,
    this class is required to reproduce ``ddm_fx1`` BIT-FOR-BIT.  That is the
    control that makes every delta below readable as the new mechanism rather
    than as new plumbing, and the sister test asserts it on a real replay.
    """

    def __init__(
        self,
        plane: int,
        *,
        sse_context: str = "off",
        sse_learn_weight: bool = False,
        **kwargs,
    ) -> None:
        families = tuple(kwargs.pop("families", ("shipped_joint",)))
        mixer_context = str(kwargs.pop("mixer_context", "none"))

        # The parent validates its arguments against ITS module-level pools, which
        # do not know the widened ones.  So the parent is constructed on a pool it
        # does accept and the widened members are installed afterwards.  The
        # alternative -- mutating the inherited module's global dictionaries --
        # would leak into every sister object in the process, so it is refused.
        specs = fx2_family_specs()
        contexts = fx2_mixer_contexts()
        if not families:
            raise ValueError("at least one family is required")
        missing = [n for n in families if n not in specs]
        if missing:
            raise ValueError(f"unknown families: {missing}")
        if mixer_context not in contexts:
            raise ValueError(f"unknown mixer context {mixer_context!r}")

        super().__init__(plane, families=("shipped_joint",), mixer_context="none", **kwargs)

        self.families = [MixerFamily(name, *specs[name]) for name in families]
        self.mixer_context_name = mixer_context
        self.n_mixer_contexts, self._mixer_rule = contexts[mixer_context]
        self.n_weight_sets = self.n_mixer_contexts * self.count_buckets
        # Member 0 at weight 1.0, every other member at 0: the mixture BEGINS at
        # the incumbent law, so the learner must earn every byte away from a
        # known-good point and a failed learner degrades toward the incumbent.
        initial = [1.0 if f.name == "shipped_joint" else 0.0 for f in self.families]
        self.weights = np.zeros((self.n_weight_sets, len(self.families)), dtype=np.int64)
        for position, value in enumerate(initial):
            self.weights[:, position] = np.int64(round(value * WEIGHT_STORE_ONE))

        if sse_context not in SSE_CONTEXTS:
            raise ValueError(f"unknown sse context {sse_context!r}")
        self.sse_context_name = sse_context
        sse_size, self._sse_rule = SSE_CONTEXTS[sse_context]
        self.sse_learn_weight = bool(sse_learn_weight)
        self.sse: MixerFamily | None = (
            MixerFamily("sse", sse_size, self._sse_rule) if sse_size else None
        )
        # One exponent per MIXER CELL for the SSE stage, stored exactly like a
        # member's so the same integer learner drives it.  Starts at 1.0, i.e.
        # the SSE correction applied as-is, which is what an APM does.
        #
        # Sized by ``n_mixer_contexts``, NOT ``n_weight_sets``.  A member's weight
        # is also bucketed by that member's own evidence count; the SSE stage has
        # no member count to bucket on, so the extra factor would allocate cells
        # this index can never reach -- dead weight that reads as a live table.
        self.sse_weight = np.full(
            self.n_mixer_contexts, np.int64(WEIGHT_STORE_ONE), dtype=np.int64
        )

    # -- the widened causal geometry (R1 / R2 feature 2) ---------------------

    def _causal_neighbours(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decoded class and availability of each templated causal neighbour.

        Returns ``(classes, available)`` each of shape ``[4, n]``.  Availability
        is the runtime ``known`` mask AND in-bounds, so an offset whose group
        happens NOT to precede this one simply reads unavailable -- correctness
        never rests on the group arithmetic in the docstring.
        """
        x = flat % WIDTH
        y = flat // WIDTH
        classes = np.zeros((len(CAUSAL_OFFSETS), flat.size), dtype=np.int64)
        available = np.zeros((len(CAUSAL_OFFSETS), flat.size), dtype=bool)
        for slot, (dx, dy) in enumerate(CAUSAL_OFFSETS):
            nx = x + dx
            ny = y + dy
            inside = (nx >= 0) & (nx < WIDTH) & (ny >= 0) & (ny < HEIGHT)
            neighbour = np.clip(ny, 0, HEIGHT - 1) * WIDTH + np.clip(nx, 0, WIDTH - 1)
            available[slot] = inside & self.known[neighbour]
            classes[slot] = np.where(available[slot], self.current[neighbour].astype(np.int64), -1)
        return classes, available

    @staticmethod
    def _spatial4_level(
        classes: np.ndarray, available: np.ndarray, base_class: np.ndarray
    ) -> np.ndarray:
        """``0`` if nothing is decoded yet, else ``1 + #(neighbours == prediction)``."""
        agreeing = ((classes == base_class[None, :]) & available).sum(axis=0)
        any_available = available.any(axis=0)
        return np.where(any_available, np.minimum(agreeing + 1, SPATIAL4_LEVELS - 1), 0).astype(
            np.int64
        )

    @staticmethod
    def _homogeneity_level(classes: np.ndarray, available: np.ndarray) -> np.ndarray:
        """Distinct decoded classes among the available causal neighbours.

        The geometric feature: ``1`` means every decoded neighbour agrees with
        every other, i.e. a cell interior; ``>= 2`` means this position sits on a
        boundary of the already-decoded partition, which is where the argmax
        misses concentrate.  Counted by integer one-hot accumulation over the
        five classes, so it is exact and vectorised.
        """
        present = np.zeros((NUM_CLASSES, classes.shape[1]), dtype=bool)
        for slot in range(classes.shape[0]):
            usable = available[slot]
            for value in range(NUM_CLASSES):
                present[value] |= usable & (classes[slot] == value)
        distinct = present.sum(axis=0).astype(np.int64)
        return np.minimum(distinct, HOMOGENEITY_LEVELS - 1)

    def group_state(
        self, probability: np.ndarray, predicted: np.ndarray, positions: np.ndarray
    ) -> GroupState:
        # The rr4 grandparent is called DIRECTLY rather than through the fx1
        # parent.  The parent would build a feature dict from the narrow template
        # and evaluate every member rule against it, which the widened members
        # cannot read -- and it would then be thrown away and rebuilt here, so
        # skipping it also removes one full pass of index arithmetic per group.
        state = ShippedFreeCorrector.group_state(self, probability, predicted, positions)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)

        classes, available = self._causal_neighbours(flat)
        spatial4 = self._spatial4_level(classes, available, base_class)
        homog = self._homogeneity_level(classes, available)

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
            "spatial4": spatial4,
            "homog": homog,
            # ddm_gb1: the index of the decode step being taken, binned.
            # g(x, y) = (x mod 64) + 2 * (y mod 64) is the shipped group plan
            # (cpr1/inflate.py:33-34,275-287); the decoder selects the position
            # before it decodes the symbol there, so this is causal by
            # construction and costs zero transmitted bytes.
            "tile48": (
                ((flat // WIDTH) // 64) * (WIDTH // 64)
                + ((flat % WIDTH) // 64)
            ),
            "patch192": (
                ((flat // WIDTH) // 32) * (WIDTH // 32)
                + ((flat % WIDTH) // 32)
            ),
            "groupbin8": (
                (((flat % WIDTH) % 64) + 2 * ((flat // WIDTH) % 64)) * 8
            )
            // 190,
        }
        self._pending = {
            "flat": flat,
            "features": features,
            "indices": [family.rule(features) for family in self.families],
            "mixer": np.asarray(self._mixer_rule(features), dtype=np.int64).reshape(-1),
        }
        return state

    # -- the two-stage estimate ---------------------------------------------

    def coding_row(self, state: GroupState) -> np.ndarray:
        """Mix the members, then calibrate the mixture through the SSE stage."""
        multiplier = self.odds_multiplier(state)
        pending = self._pending
        assert pending is not None

        shifted = state.p_max * multiplier
        q_mix = np.clip(shifted / (shifted + state.one_minus), PROB_EPS, 1.0 - PROB_EPS)
        # The mixer's learner descends on its OWN output, exactly as a PAQ mixer
        # does; the SSE stage is a separate adaptive map, not another member.
        pending["q"] = q_mix

        if self.sse is not None:
            multiplier = self._apply_sse(state, multiplier, q_mix)
            shifted = state.p_max * multiplier
            q_mix = np.clip(shifted / (shifted + state.one_minus), PROB_EPS, 1.0 - PROB_EPS)

        row = state.row64.copy()
        active = multiplier != 1.0
        if np.any(active):
            scale = (1.0 - q_mix[active]) / state.one_minus[active]
            rows = row[active] * scale[:, None]
            rows[np.arange(rows.shape[0]), state.arg[active]] = q_mix[active]
            row[active] = rows
        return row.astype(np.float32)

    def _apply_sse(
        self, state: GroupState, multiplier: np.ndarray, q_mix: np.ndarray
    ) -> np.ndarray:
        """Secondary Symbol Estimation on the mixture's own output probability.

        Indexed by the frozen ``2^(-k/2)`` ladder on ``1 - q_mix`` -- exactly the
        surprise partition ``ddm_rr4`` already computes without a logarithm --
        crossed with a small context.  The correction is the shipped KT count
        ratio with ``q_mix`` as the accumulated expectation, so a calibrated bin
        returns exactly 1.0 and a cold bin returns exactly 1.0.
        """
        assert self.sse is not None
        pending = self._pending
        assert pending is not None

        one_minus_q = np.maximum(1.0 - q_mix, PROB_EPS)
        below = np.searchsorted(_SURPRISE_ASC, one_minus_q, side="left")
        qbin = np.clip((SSE_BINS - 1) - below, 0, SSE_BINS - 1).astype(np.int64)

        features = dict(pending["features"])
        features["qbin"] = qbin
        index = np.asarray(self._sse_rule(features), dtype=np.int64).reshape(-1)
        correction = self.sse.multiplier(index)

        if self.sse_learn_weight:
            grid_weight = round_shift(self.sse_weight[pending["mixer"]], self.store_shift)
            radicals: list[np.ndarray] = []
            root = correction
            for _ in range(self.power_bits):
                root = np.sqrt(root)
                radicals.append(root)
            pending["sse_stretch"] = stretch_from_radical(radicals[-1], self.power_bits)
            correction = dyadic_power(correction, radicals, grid_weight, self.power_bits)

        pending["sse_index"] = index
        # Fixed-point q_mix for the order-independent accumulator, matching the
        # shipped p_max_q construction exactly (one correctly rounded rint).
        pending["sse_expected_q"] = np.rint(q_mix * PHAT_SCALE).astype(np.int64)

        blended = multiplier * correction
        np.clip(blended, ODDS_LOW, ODDS_HIGH, out=blended)
        return blended

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("observe() called without a group_state()")
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        hit = (decoded == state.arg).astype(np.int64)

        if self.sse is not None and "sse_index" in pending:
            if self.sse_learn_weight and self.learn and "sse_stretch" in pending:
                self._update_sse_weight(hit, pending["q"], pending["mixer"], pending["sse_stretch"])
            self.sse.observe(pending["sse_index"], hit, pending["sse_expected_q"])

        super().observe(state, symbols)

    def _update_sse_weight(
        self,
        hit: np.ndarray,
        q: np.ndarray,
        mixer: np.ndarray,
        stretch: np.ndarray,
    ) -> None:
        """The same integer gradient step the members get, on the SSE exponent."""
        from .fx1_logistic_mixer_corrector import (
            ERR_SCALE,
            STRETCH_CLAMP,
            STRETCH_SCALE,
            WEIGHT_HIGH,
            WEIGHT_LOW,
        )

        residual = np.rint((hit.astype(np.float64) - q) * ERR_SCALE).astype(np.int64)
        quantised = np.clip(
            np.rint(stretch * STRETCH_SCALE), -STRETCH_CLAMP, STRETCH_CLAMP
        ).astype(np.int64)
        counts = np.bincount(mixer, minlength=self.n_mixer_contexts).astype(np.int64)
        gradient = np.zeros(self.n_mixer_contexts, dtype=np.int64)
        np.add.at(gradient, mixer, residual * quantised)
        if self.normalize:
            gradient = np.where(counts > 0, gradient // np.maximum(counts, 1), 0)
        step = round_shift(gradient, self.lr_shift)
        self.sse_weight = np.clip(self.sse_weight + step, WEIGHT_LOW, WEIGHT_HIGH)

    # -- observability ------------------------------------------------------

    def weight_report(self) -> dict[str, list[float]]:
        report = super().weight_report()
        if self.sse is not None and self.sse_learn_weight:
            scale = float(WEIGHT_STORE_ONE)
            report["sse"] = [float(v) / scale for v in self.sse_weight]
        return report


# --- the frozen shipped configuration ----------------------------------------

SHIPPED_CONFIG: dict = {
    "families": (
        # ddm_fx1's eleven, unchanged.
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
        # The two members the widened causal template unlocks.
        "spatial4_surprise",
        "homog_surprise",
        # ddm_fx5: the six that take D1's -710.84 B to E1's -797.42 B.  ddm_fx2
        # measured this exact set and withheld it on a PROJECTED 29 s decode
        # margin; ddm_rc2 MEASURED 323.5 s of real T4 slack, so it ships.
        "homog_boundary_surprise",
        "spatial4_boundary",
        "homog_spatial4",
        "spatial4_temporal",
        "homog_surprise_fast256",
        "spatial4_surprise_fast256",
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
}
"""The configuration this arm selected: **−710.84 B** on the full n600 field.

A decoder takes no arguments, so anything configurable at encode time but not at
decode time is a desynchronisation waiting to happen; the encoder reads this same
frozen dict.

WHY THIS ROW AND NOT THE LARGEST ONE.  A 19-member build measured −797.42 B, 87 B
better, and it is NOT frozen here.  ``ddm_fx1`` measured the real parse-back at
1,502.29 s base and 1,639.78 s for its own candidate against a 1,800 s contest
budget — 160.2 s of margin, the thinnest number in this lineage.  Serial timing
of these builds projects that margin at **118 s here and 29 s at 19 members**.
Twenty-nine seconds is not a margin, so the extra 87 B is refused.  The decode
cost is what picks the candidate, which is why it was measured serially rather
than inferred from the contended race times.

``sse_context`` stays ``off`` because the SSE/APM stage measured NEGATIVE in six
of six formulations (+9.91 B to +139.56 B); it is kept in the module because a
measured negative with a working implementation is worth more than a deleted one.
"""


class FreeCorrector(Fx2ModelAxisMixer):
    """Drop-in for ``ddm_rr4_free_corrector_v2.FreeCorrector``, frozen config."""

    def __init__(self, plane: int) -> None:
        super().__init__(plane, **SHIPPED_CONFIG)
