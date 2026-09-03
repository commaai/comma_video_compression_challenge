/* ddm_rr8 - the float64 free corrector, lowered to C, bit-identical by construction.
 *
 * WHY THIS EXISTS.  ``ddm_cd1`` MEASURED the shipped jg5 token stage on a contest T4 and
 * found the corrector is **917.929 s = 71.7%** of it -- ``corrector_coding_row`` 526.325 s,
 * ``corrector_observe`` 271.842 s, ``corrector_group_state`` 119.762 s.  ``ddm_rr7`` had
 * already lowered the OTHER 21% (the integer HPAC model) to C and MEASURED a 15.3%
 * REGRESSION, because it moved GPU work onto weak container vCPUs.  This port does the
 * opposite: the corrector is ALREADY on those vCPUs, in numpy.  Lowering it changes the
 * language, not the processor, so rr7's mechanism is absent from this one by construction.
 *
 * Break-even, pre-registered by ddm_cd1 §6.4 against the CI wall: **2.03x** (frame B) /
 * **2.77x** (frame A) on the port's OWN scope.  Below 2.03x this file has cleared nothing.
 *
 * WHAT IT REPRODUCES.  ``free_corrector.FreeCorrector`` -- i.e. ``Ma1WithinMissCorrector``
 * under the frozen ``SHIPPED_CONFIG``, whose MRO is
 *
 *     free_corrector.FreeCorrector          (ma1: the within-miss relative law)
 *       -> Fx2ModelAxisMixer                (fx2: widened causal template + SSE stage)
 *         -> FixedPointLogisticMixer        (fx1: the fixed-point log-odds mixer)
 *           -> rr4_free_corrector.FreeCorrector
 *
 * TWO STRUCTURAL FACTS ABOUT THE FROZEN CONFIG, both load-bearing and both verified against
 * the shipped sources rather than assumed:
 *
 *   1. ``sse_context = "off"`` makes ``self.sse`` None, so ``Fx2ModelAxisMixer._apply_sse``
 *      and ``_update_sse_weight`` are NEVER reached on the shipped path.  They are not
 *      ported.  A config that turned them on would need them, which is why the Python
 *      wrapper REFUSES to bind unless the live config matches the one compiled in here.
 *   2. ``FixedPointLogisticMixer.observe`` does NOT call ``super().observe``, so rr4's own
 *      ``counts``/``hits``/``phat_q`` (51,200 cells each) are written by nothing and read by
 *      nothing -- ``odds_multiplier`` is fully overridden.  They are dead state and are not
 *      allocated here.  The ``shipped_joint`` MEMBER carries the identical context rule and
 *      IS live; the two must not be confused.
 *
 * EXACTNESS -- the whole point.  Every value below is produced by IEEE-754 correctly rounded
 * operations only (+ - * / compare sqrt rint, and float32<->float64 conversion), in the same
 * order numpy produces them, so the result is bit-identical on every conforming platform.
 * There is no log, exp, log2, exp2 or pow anywhere -- that is the ``ddm_rr2`` refusal class
 * (S = 27.83, one libm ULP desynchronising the arithmetic decoder), and the Python sources
 * this file mirrors assert it by walking their own AST.  Specific hazards handled:
 *
 *   * ``rint`` (round-half-to-EVEN) is used everywhere numpy uses ``np.rint``.  C's
 *     ``round()`` is half-away-from-zero and would differ on exact .5 -- it appears nowhere.
 *   * Integer floor division and arithmetic right shift are written explicitly
 *     (``floor_div_i64`` / ``round_shift``), because numpy's ``//`` floors toward -inf while
 *     C's ``/`` truncates toward zero.  The learner's gradient IS negative half the time, so
 *     this is a live difference, not a theoretical one.
 *   * MUST be compiled with ``-ffp-contract=off -fno-fast-math``.  FMA contraction would
 *     fuse a multiply and an add into one rounding step and change results.  The Python
 *     wrapper cannot check the compiler flags, so the parity harness checks the RESULT.
 *
 * THE SPEEDUP MECHANISM, stated so it can be falsified rather than hoped for.  numpy runs
 * the mixer MEMBER-outer / position-inner: for each of the 19 members it materialises the
 * multiplier, six radicals, the stretch, and the dyadic-power accumulator as full-length
 * float64 temporaries -- on the order of a hundred heap arrays per group, times 114,000
 * groups.  This file runs it POSITION-outer / member-inner, so the same arithmetic happens
 * in registers with no allocation at all.  That reordering is legal because each member's
 * contribution is elementwise independent; the 13-way PRODUCT order (member 0..12) is
 * preserved exactly, which is the part that would change float results if it were not.
 *
 * RULE 118.  Generic decoder code.  Zero transmitted bytes, no table shipped, no
 * video-derived constant: every table below starts at zero or at a first-principles integer
 * and is filled only from symbols the receiver has already decoded.
 *
 * DISTORTION.  None, by construction.  Only the probability row handed to the RC64 coder is
 * produced here; the decoded token field is bit-identical, which is the falsifier for this
 * whole file rather than its justification.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define F26C_ABI_VERSION 1

/* --- frozen constants, mirrored from the shipped Python sources ------------------------ */
/* rr4_free_corrector.py */
#define NUM_CLASSES 5
#define U_BINS 64
#define RUN_LEVELS 8
#define RUN_CAP 255
#define BOUNDARY_LEVELS 5
#define KT_ALPHA 0.5
#define MIN_COUNT 32
#define ODDS_LOW 0.0625
#define ODDS_HIGH 16.0
#define PROB_EPS 1e-9
#define PHAT_SCALE 1073741824.0 /* 1 << 30 */
#define PHAT_SCALE_I ((int64_t)1 << 30)

/* fx1_logistic_mixer_corrector.py */
#define HEIGHT 384
#define WIDTH 512
#define SPATIAL_LEVELS 5
#define WEIGHT_STORE_BITS 20
#define WEIGHT_STORE_ONE ((int64_t)1 << 20)
#define POWER_BITS 6
#define INT_POWER_BITS 4
#define ERR_SCALE 1048576.0  /* 1 << 20 */
#define STRETCH_SCALE 1048576.0 /* 1 << 20 */
#define STRETCH_CLAMP 33554432.0 /* 32 * (1 << 20) */
#define COUNT_HALVING_PASSES 40
#define WEIGHT_LOW (-4 * WEIGHT_STORE_ONE)
#define WEIGHT_HIGH (8 * WEIGHT_STORE_ONE)
#define STORE_SHIFT (WEIGHT_STORE_BITS - POWER_BITS) /* 14 */
#define LR_SHIFT 24 /* LR_BASE_SHIFT (20) + 4 */

/* fx2_model_axis_corrector.py */
#define SPATIAL4_LEVELS 6
#define HOMOGENEITY_LEVELS 5
#define N_CAUSAL 4
#define GROUP_BINS 8 /* ddm_gb1 */

/* free_corrector.py (ddm_ma1) */
#define UNKNOWN NUM_CLASSES
#define MISS_KT_ALPHA 0.5
#define MISS_MIN_COUNT 1
#define MISS_CLAMP_HIGH 16.0
#define MISS_CLAMP_LOW 0.0625
#define MISS_BASE (NUM_CLASSES + 1) /* 6 */
#define N_MISS_CELLS (MISS_BASE * MISS_BASE * MISS_BASE * MISS_BASE) /* 1296, nb3_prev1 */

/* the live family set, in SHIPPED_CONFIG order.
 * ddm_fx5: 13 -> 19.  ddm_fx2 raced this member set as E1 and measured -797.42 B
 * against the live rr4 law, 86.58 B beyond the D1 build rc2 ships; it withheld E1
 * only because serial timing PROJECTED a 29 s parse-back margin.  ddm_rc2 then
 * MEASURED the real T4 wall at 498.476 s against an 822 s ceiling (323.5 s slack),
 * so the withholding precondition is discharged.  N_MIXER_CONTEXTS is unchanged:
 * E1 and D1 share the mixer context exactly. */
#define N_FAMILIES 23 /* ddm_afr1: +tile48_groupbin8 */
#define N_MIXER_CONTEXTS (NUM_CLASSES * BOUNDARY_LEVELS * 4 * HOMOGENEITY_LEVELS * 8) /* 4000 */
#define N_WEIGHT_SETS N_MIXER_CONTEXTS /* count_buckets == 1 */

#define JOINT_SIZE (NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS) /* 51200 */

/* CAUSAL_OFFSETS = ((-1,0), (0,-1), (1,-1), (-1,-1)); slot names from ddm_ma1. */
#define SLOT_LEFT 0
#define SLOT_UP 1
#define SLOT_UPRIGHT 2
#define SLOT_UPLEFT 3
static const int CAUSAL_DX[N_CAUSAL] = {-1, 0, 1, -1};
static const int CAUSAL_DY[N_CAUSAL] = {0, -1, -1, -1};

/* Family rule selectors.  The rule is what varies between members; the ESTIMATOR is
 * identical for all of them (MixerFamily.multiplier), which is what makes the mixer a test
 * of context rather than of two changes at once. */
enum {
    RULE_SHIPPED = 0,
    RULE_TEMPORAL_SPATIAL,
    RULE_SURPRISE_ONLY,
    RULE_SPATIAL_SURPRISE,
    RULE_SPATIAL_BOUNDARY,
    RULE_RUN_SURPRISE,
    RULE_BOUNDARY_SURPRISE,
    RULE_TEMPORAL_SURPRISE,
    RULE_SPATIAL4_SURPRISE,
    RULE_HOMOG_SURPRISE,
    /* ddm_fx5: the four rules E1's six new members need.  17 and 18 reuse
     * RULE_HOMOG_SURPRISE / RULE_SPATIAL4_SURPRISE at count_limit 256. */
    RULE_HOMOG_BOUNDARY_SURPRISE,
    RULE_SPATIAL4_BOUNDARY,
    RULE_HOMOG_SPATIAL4,
    RULE_SPATIAL4_TEMPORAL,
    /* ddm_gb1: decode-scan group conditioning. */
    RULE_GROUPBIN8_SURPRISE,      /* groupbin8_surprise, ddm_gb1 */
    RULE_CLS_GROUPBIN8,           /* cls_groupbin8, ddm_jt21 */
    RULE_PATCH192_ONLY,           /* patch192_only, ddm_lb1 */
    RULE_TILE48_GROUPBIN8          /* tile48_groupbin8, ddm_afr1 */
};

/* SHIPPED_CONFIG["families"], in order.  Member 0 is ``shipped_joint`` and starts at weight
 * exactly 1.0, so the mixture BEGINS at the incumbent law. */
static const int FAMILY_RULE[N_FAMILIES] = {
    RULE_SHIPPED,            /* shipped_joint      */
    RULE_TEMPORAL_SPATIAL,   /* temporal_spatial   */
    RULE_SURPRISE_ONLY,      /* surprise_only      */
    RULE_SPATIAL_SURPRISE,   /* spatial_surprise   */
    RULE_SPATIAL_BOUNDARY,   /* spatial_boundary   */
    RULE_RUN_SURPRISE,       /* run_surprise       */
    RULE_BOUNDARY_SURPRISE,  /* boundary_surprise  */
    RULE_TEMPORAL_SURPRISE,  /* temporal_surprise  */
    RULE_SHIPPED,            /* shipped_fast256    */
    RULE_SHIPPED,            /* shipped_fast4096   */
    RULE_SURPRISE_ONLY,      /* surprise_fast256   */
    RULE_SPATIAL4_SURPRISE,  /* spatial4_surprise  */
    RULE_HOMOG_SURPRISE,     /* homog_surprise     */
    /* ddm_fx5: E1's six. */
    RULE_HOMOG_BOUNDARY_SURPRISE, /* homog_boundary_surprise   */
    RULE_SPATIAL4_BOUNDARY,       /* spatial4_boundary         */
    RULE_HOMOG_SPATIAL4,          /* homog_spatial4            */
    RULE_SPATIAL4_TEMPORAL,       /* spatial4_temporal         */
    RULE_HOMOG_SURPRISE,          /* homog_surprise_fast256    */
    RULE_SPATIAL4_SURPRISE,       /* spatial4_surprise_fast256 */
    RULE_GROUPBIN8_SURPRISE,      /* groupbin8_surprise, ddm_gb1 */
    RULE_CLS_GROUPBIN8,           /* cls_groupbin8, ddm_jt21 */
    RULE_PATCH192_ONLY,           /* patch192_only, ddm_lb1 */
    RULE_TILE48_GROUPBIN8          /* tile48_groupbin8, ddm_afr1 */
};

static const int64_t FAMILY_SIZE[N_FAMILIES] = {
    JOINT_SIZE,
    NUM_CLASSES * 2 * 2 * SPATIAL_LEVELS,
    NUM_CLASSES * U_BINS,
    NUM_CLASSES * SPATIAL_LEVELS * U_BINS,
    NUM_CLASSES * SPATIAL_LEVELS * BOUNDARY_LEVELS,
    NUM_CLASSES * RUN_LEVELS * U_BINS,
    NUM_CLASSES * BOUNDARY_LEVELS * U_BINS,
    NUM_CLASSES * 2 * 2 * U_BINS,
    JOINT_SIZE,
    JOINT_SIZE,
    NUM_CLASSES * U_BINS,
    NUM_CLASSES * SPATIAL4_LEVELS * U_BINS,
    NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS,
    /* ddm_fx5: E1's six, transcribed from fx2_family_specs(). */
    NUM_CLASSES * HOMOGENEITY_LEVELS * BOUNDARY_LEVELS * U_BINS, /* 8000 */
    NUM_CLASSES * SPATIAL4_LEVELS * BOUNDARY_LEVELS,             /*  150 */
    NUM_CLASSES * HOMOGENEITY_LEVELS * SPATIAL4_LEVELS,          /*  150 */
    NUM_CLASSES * 2 * 2 * SPATIAL4_LEVELS,                       /*  120 */
    NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS,                   /* 1600 */
    NUM_CLASSES * SPATIAL4_LEVELS * U_BINS,                      /* 1920 */
    NUM_CLASSES * GROUP_BINS * U_BINS,                           /* 2560, ddm_gb1 */
    NUM_CLASSES * GROUP_BINS,                                    /* 40, ddm_jt21 */
    192,                                                         /* ddm_lb1 */
    48 * GROUP_BINS                                               /* 384, ddm_afr1 */
};

/* MixerFamily.count_limit.  Nonzero enables the repeated-halving recency window. */
static const int64_t FAMILY_COUNT_LIMIT[N_FAMILIES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 256, 4096, 256, 0, 0,
    /* ddm_fx5: E1's six -- the two ``_fast256`` members carry the recency window. */
    0, 0, 0, 0, 256, 256,
    /* ddm_gb1 + ddm_jt21 + ddm_lb1 + ddm_afr1 */
    0, 0, 0, 0
};

/* Initial weights: 1.0 on ``shipped_joint`` (member 0), 0.0 on every other member. */
static const int FAMILY_IS_SHIPPED_JOINT[N_FAMILIES] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /* ddm_fx5: E1's six all start at weight 0, so the mixture still BEGINS at the
     * incumbent law and the learner must earn every byte away from it. */
    0, 0, 0, 0, 0, 0,
    /* ddm_gb1 + ddm_jt21 + ddm_lb1 + ddm_afr1 */
    0, 0, 0, 0
};

/* --- exact scalar helpers -------------------------------------------------------------- */

/* numpy's ``//`` floors toward -infinity; C's ``/`` truncates toward zero.  The learner's
 * gradient is negative about half the time, so the difference is live. */
static inline int64_t floor_div_i64(int64_t numerator, int64_t denominator)
{
    int64_t quotient = numerator / denominator;
    int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        quotient -= 1;
    }
    return quotient;
}

/* ``round_shift`` from fx1: arithmetic right shift with round-half-up.  An arithmetic shift
 * IS floor division by a power of two, written explicitly so nothing rests on the
 * implementation-defined behaviour of ``>>`` on a negative signed integer. */
static inline int64_t round_shift(int64_t value, int bits)
{
    if (bits <= 0) {
        return value;
    }
    int64_t half = (int64_t)1 << (bits - 1);
    return floor_div_i64(value + half, (int64_t)1 << bits);
}

static inline double clamp_double(double value, double low, double high)
{
    /* np.clip == min(max(v, low), high) for the ordered bounds used here. */
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline int64_t clamp_i64(int64_t value, int64_t low, int64_t high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

/* --- the frozen 2^(-k/2) surprise ladder ------------------------------------------------ */

/* rr4._surprise_thresholds, reversed to ascending.  Built once, from ldexp and a pinned
 * sqrt(0.5), so it is the same 63 doubles on every conforming platform.  The odd entries
 * need 1/sqrt(2); IEEE requires sqrt to be correctly rounded, which is exactly why a radical
 * is permitted here where a logarithm is not. */
static double SURPRISE_ASC[U_BINS - 1];
static int SURPRISE_READY = 0;

static const uint64_t INV_SQRT2_BITS = 0x3FE6A09E667F3BCDull;

static int build_surprise_table(void)
{
    double inv_sqrt2 = sqrt(0.5);
    uint64_t bits;
    memcpy(&bits, &inv_sqrt2, sizeof(bits));
    if (bits != INV_SQRT2_BITS) {
        return -1; /* fail closed, exactly as rr4 does at import */
    }
    /* descending[k-1] for k = 1..63, then reversed */
    double descending[U_BINS - 1];
    for (int k = 1; k < U_BINS; ++k) {
        double value = ldexp(1.0, -(k / 2));
        if (k % 2) {
            value *= inv_sqrt2;
        }
        descending[k - 1] = value;
    }
    for (int i = 0; i < U_BINS - 1; ++i) {
        SURPRISE_ASC[i] = descending[U_BINS - 2 - i];
    }
    SURPRISE_READY = 1;
    return 0;
}

/* np.searchsorted(ascending, value, side="left") == count of entries strictly below value. */
static inline int64_t searchsorted_left(const double *ascending, int64_t size, double value)
{
    int64_t low = 0;
    int64_t high = size;
    while (low < high) {
        int64_t mid = low + ((high - low) >> 1);
        if (ascending[mid] < value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

/* fx1._assert_sqrt_is_correctly_rounded, mirrored.  The probes are mathematical facts, not
 * readings taken from this machine: sqrt of an exactly representable square is that square's
 * root exactly.  A table transcribed from the local libm would certify the platform against
 * itself and prove nothing. */
static int sqrt_is_correctly_rounded(void)
{
    static const double roots[8] = {1.0, 1.5, 2.0, 3.0, 4.0, 1.0625, 65536.0, 1e8};
    for (int i = 0; i < 8; ++i) {
        double squared = roots[i] * roots[i];
        if (sqrt(squared) != roots[i]) {
            return 0;
        }
    }
    double value = 1.0;
    for (int i = 0; i < 8; ++i) {
        value = value / 4.0;
    }
    for (int i = 0; i < 16; ++i) {
        if (sqrt(value * value) != value) {
            return 0;
        }
        value = value * 4.0;
    }
    return 1;
}

/* --- the corrector ---------------------------------------------------------------------- */

typedef struct {
    int64_t *counts;
    int64_t *hits;
    int64_t *phat_q;
    int64_t size;
    int64_t count_limit;
    int rule;
} Family;

typedef struct {
    int64_t plane;

    /* rr4 per-pixel temporal memory */
    uint8_t *prev1;
    uint8_t *prev2;
    int64_t *run;
    int64_t *boundary;
    int have_prev;

    /* fx1 decoded-plane state */
    uint8_t *current;
    uint8_t *known;

    Family families[N_FAMILIES];
    int64_t *weights; /* [N_WEIGHT_SETS][N_FAMILIES], row major, matching numpy */

    /* ma1 within-miss tables */
    int64_t *miss_counts; /* [N_MISS_CELLS][NUM_CLASSES] */
    int64_t *miss_expect; /* [N_MISS_CELLS][NUM_CLASSES] */
    int64_t *miss_seen;   /* [N_MISS_CELLS] */

    /* per-group scratch, grown on demand */
    int64_t capacity;
    int64_t n;
    int group_open; /* 1 between group_state() and observe() */

    double *row64;      /* [n][5] the receiver's own row, widened */
    int64_t *arg;       /* [n] */
    double *p_max;      /* [n] */
    double *one_minus;  /* [n] */
    int64_t *p_max_q;   /* [n] */
    int64_t *flat;      /* [n] */
    int64_t *fam_index; /* [N_FAMILIES][n] */
    int64_t *mixer;     /* [n] */
    int64_t *miss_cell; /* [n] */
    double *stretch;    /* [N_FAMILIES][n] */
    double *q;          /* [n] */
    double *blended;    /* [n] */
    double *out64;      /* [n][5] */

    /* learner scratch, fixed size */
    int64_t *ws_counts;   /* [N_WEIGHT_SETS] */
    int64_t *ws_gradient; /* [N_WEIGHT_SETS] */
    int64_t *residual;    /* [n] quantised (y - q), hoisted out of the member loop */
    int64_t *hit;         /* [n] */
} Corrector;

static void family_free(Family *family)
{
    free(family->counts);
    free(family->hits);
    free(family->phat_q);
    family->counts = NULL;
    family->hits = NULL;
    family->phat_q = NULL;
}

static int family_init(Family *family, int rule, int64_t size, int64_t count_limit)
{
    family->rule = rule;
    family->size = size;
    family->count_limit = count_limit;
    family->counts = (int64_t *)calloc((size_t)size, sizeof(int64_t));
    family->hits = (int64_t *)calloc((size_t)size, sizeof(int64_t));
    family->phat_q = (int64_t *)calloc((size_t)size, sizeof(int64_t));
    if (!family->counts || !family->hits || !family->phat_q) {
        family_free(family);
        return -1;
    }
    return 0;
}

void f26_corrector_destroy(void *handle)
{
    Corrector *self = (Corrector *)handle;
    if (!self) {
        return;
    }
    for (int i = 0; i < N_FAMILIES; ++i) {
        family_free(&self->families[i]);
    }
    free(self->prev1);
    free(self->prev2);
    free(self->run);
    free(self->boundary);
    free(self->current);
    free(self->known);
    free(self->weights);
    free(self->miss_counts);
    free(self->miss_expect);
    free(self->miss_seen);
    free(self->row64);
    free(self->arg);
    free(self->p_max);
    free(self->one_minus);
    free(self->p_max_q);
    free(self->flat);
    free(self->fam_index);
    free(self->mixer);
    free(self->miss_cell);
    free(self->stretch);
    free(self->q);
    free(self->blended);
    free(self->out64);
    free(self->ws_counts);
    free(self->ws_gradient);
    free(self->residual);
    free(self->hit);
    free(self);
}

static int ensure_capacity(Corrector *self, int64_t n)
{
    if (n <= self->capacity) {
        return 0;
    }
    int64_t capacity = self->capacity ? self->capacity : 1024;
    while (capacity < n) {
        capacity *= 2;
    }
    /* Allocate into locals first so a mid-way failure leaves the handle intact and the
     * caller can fall back to the Python corrector rather than inherit a torn state. */
    double *row64 = (double *)malloc((size_t)capacity * NUM_CLASSES * sizeof(double));
    int64_t *arg = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));
    double *p_max = (double *)malloc((size_t)capacity * sizeof(double));
    double *one_minus = (double *)malloc((size_t)capacity * sizeof(double));
    int64_t *p_max_q = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));
    int64_t *flat = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));
    int64_t *fam_index =
        (int64_t *)malloc((size_t)capacity * N_FAMILIES * sizeof(int64_t));
    int64_t *mixer = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));
    int64_t *miss_cell = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));
    double *stretch = (double *)malloc((size_t)capacity * N_FAMILIES * sizeof(double));
    double *q = (double *)malloc((size_t)capacity * sizeof(double));
    double *blended = (double *)malloc((size_t)capacity * sizeof(double));
    double *out64 = (double *)malloc((size_t)capacity * NUM_CLASSES * sizeof(double));
    int64_t *residual = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));
    int64_t *hit = (int64_t *)malloc((size_t)capacity * sizeof(int64_t));

    if (!row64 || !arg || !p_max || !one_minus || !p_max_q || !flat || !fam_index ||
        !mixer || !miss_cell || !stretch || !q || !blended || !out64 || !residual || !hit) {
        free(row64); free(arg); free(p_max); free(one_minus); free(p_max_q);
        free(flat); free(fam_index); free(mixer); free(miss_cell); free(stretch);
        free(q); free(blended); free(out64); free(residual); free(hit);
        return -1;
    }

    free(self->row64); free(self->arg); free(self->p_max); free(self->one_minus);
    free(self->p_max_q); free(self->flat); free(self->fam_index); free(self->mixer);
    free(self->miss_cell); free(self->stretch); free(self->q); free(self->blended);
    free(self->out64); free(self->residual); free(self->hit);

    self->row64 = row64;
    self->arg = arg;
    self->p_max = p_max;
    self->one_minus = one_minus;
    self->p_max_q = p_max_q;
    self->flat = flat;
    self->fam_index = fam_index;
    self->mixer = mixer;
    self->miss_cell = miss_cell;
    self->stretch = stretch;
    self->q = q;
    self->blended = blended;
    self->out64 = out64;
    self->residual = residual;
    self->hit = hit;
    self->capacity = capacity;
    return 0;
}

void *f26_corrector_create(int64_t plane)
{
    if (plane != (int64_t)HEIGHT * WIDTH) {
        return NULL; /* the mixer assumes the shipped 384x512 plane */
    }
    if (!SURPRISE_READY && build_surprise_table() != 0) {
        return NULL;
    }
    if (!sqrt_is_correctly_rounded()) {
        return NULL;
    }

    Corrector *self = (Corrector *)calloc(1, sizeof(Corrector));
    if (!self) {
        return NULL;
    }
    self->plane = plane;

    self->prev1 = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->prev2 = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->run = (int64_t *)calloc((size_t)plane, sizeof(int64_t));
    self->boundary = (int64_t *)malloc((size_t)plane * sizeof(int64_t));
    self->current = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->known = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->weights =
        (int64_t *)calloc((size_t)N_WEIGHT_SETS * N_FAMILIES, sizeof(int64_t));
    self->miss_counts =
        (int64_t *)calloc((size_t)N_MISS_CELLS * NUM_CLASSES, sizeof(int64_t));
    self->miss_expect =
        (int64_t *)calloc((size_t)N_MISS_CELLS * NUM_CLASSES, sizeof(int64_t));
    self->miss_seen = (int64_t *)calloc((size_t)N_MISS_CELLS, sizeof(int64_t));
    self->ws_counts = (int64_t *)calloc((size_t)N_WEIGHT_SETS, sizeof(int64_t));
    self->ws_gradient = (int64_t *)calloc((size_t)N_WEIGHT_SETS, sizeof(int64_t));

    if (!self->prev1 || !self->prev2 || !self->run || !self->boundary || !self->current ||
        !self->known || !self->weights || !self->miss_counts || !self->miss_expect ||
        !self->miss_seen || !self->ws_counts || !self->ws_gradient) {
        f26_corrector_destroy(self);
        return NULL;
    }

    for (int i = 0; i < N_FAMILIES; ++i) {
        if (family_init(&self->families[i], FAMILY_RULE[i], FAMILY_SIZE[i],
                        FAMILY_COUNT_LIMIT[i]) != 0) {
            f26_corrector_destroy(self);
            return NULL;
        }
    }

    /* rr4.__init__: boundary starts at BOUNDARY_LEVELS - 1 everywhere. */
    for (int64_t i = 0; i < plane; ++i) {
        self->boundary[i] = BOUNDARY_LEVELS - 1;
    }
    /* Member 0 (shipped_joint) at weight exactly 1.0, everything else at 0. */
    for (int64_t ws = 0; ws < N_WEIGHT_SETS; ++ws) {
        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            self->weights[ws * N_FAMILIES + pos] =
                FAMILY_IS_SHIPPED_JOINT[pos] ? WEIGHT_STORE_ONE : 0;
        }
    }
    if (ensure_capacity(self, 2048) != 0) {
        f26_corrector_destroy(self);
        return NULL;
    }
    return self;
}

/* --- driving ---------------------------------------------------------------------------- */

int f26_corrector_begin_frame(void *handle, const int64_t *boundary, int64_t size)
{
    Corrector *self = (Corrector *)handle;
    if (!self || size != self->plane) {
        return -1;
    }
    memcpy(self->boundary, boundary, (size_t)size * sizeof(int64_t));
    /* fx1.begin_frame resets the decoded-plane state after the rr4 boundary pin. */
    memset(self->known, 0, (size_t)self->plane * sizeof(uint8_t));
    memset(self->current, 0, (size_t)self->plane * sizeof(uint8_t));
    self->group_open = 0;
    return 0;
}

/* MixerFamily.multiplier, one position. */
static inline double family_multiplier(const Family *family, int64_t index)
{
    int64_t raw_count = family->counts[index];
    double count = (double)raw_count;
    double denominator = count + 2.0 * KT_ALPHA;
    double hit_numerator = (double)family->hits[index] + KT_ALPHA;
    double hit_denominator = denominator - hit_numerator;
    double expected = (double)family->phat_q[index] / PHAT_SCALE;
    double exp_numerator = expected + KT_ALPHA;
    double exp_denominator = denominator - exp_numerator;

    double multiplier = 1.0;
    if (raw_count >= MIN_COUNT && hit_numerator > 0.0 && hit_denominator > 0.0 &&
        exp_numerator > 0.0 && exp_denominator > 0.0) {
        multiplier = (hit_numerator * exp_denominator) / (hit_denominator * exp_numerator);
    }
    return clamp_double(multiplier, ODDS_LOW, ODDS_HIGH);
}

/* fx1.dyadic_power, one position.  ``radicals[i]`` is ``value ** (1 / 2**(i+1))``. */
static inline double dyadic_power(double value, const double *radicals, int64_t weight)
{
    int negative = weight < 0;
    int64_t magnitude = negative ? -weight : weight;
    int64_t integer_part = magnitude >> POWER_BITS;
    int64_t fraction = magnitude & (((int64_t)1 << POWER_BITS) - 1);

    double accumulator = 1.0;
    double base = value;
    int64_t remaining = integer_part;
    for (int i = 0; i < INT_POWER_BITS; ++i) {
        if ((remaining & 1) == 1) {
            accumulator = accumulator * base;
        }
        base = base * base;
        remaining >>= 1;
    }
    for (int index = 0; index < POWER_BITS; ++index) {
        int64_t bit = (fraction >> (POWER_BITS - 1 - index)) & 1;
        if (bit == 1) {
            accumulator = accumulator * radicals[index];
        }
    }
    return negative ? 1.0 / accumulator : accumulator;
}

static inline int64_t family_rule_index(int rule, int64_t cls, int64_t ubin, int64_t agree1,
                                        int64_t agree2, int64_t run, int64_t boundary,
                                        int64_t spatial, int64_t spatial4, int64_t homog,
                                        int64_t groupbin8, int64_t patch192,
                                        int64_t tile48_groupbin8)
{
    switch (rule) {
    case RULE_SHIPPED: {
        int64_t head = ((cls * U_BINS + ubin) * 2 + agree1) * 2 + agree2;
        return (head * RUN_LEVELS + run) * BOUNDARY_LEVELS + boundary;
    }
    case RULE_TEMPORAL_SPATIAL: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * SPATIAL_LEVELS + spatial;
    }
    case RULE_SURPRISE_ONLY:
        return cls * U_BINS + ubin;
    case RULE_SPATIAL_SURPRISE:
        return (cls * SPATIAL_LEVELS + spatial) * U_BINS + ubin;
    case RULE_SPATIAL_BOUNDARY:
        return (cls * SPATIAL_LEVELS + spatial) * BOUNDARY_LEVELS + boundary;
    case RULE_RUN_SURPRISE:
        return (cls * RUN_LEVELS + run) * U_BINS + ubin;
    case RULE_BOUNDARY_SURPRISE:
        return (cls * BOUNDARY_LEVELS + boundary) * U_BINS + ubin;
    case RULE_TEMPORAL_SURPRISE: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * U_BINS + ubin;
    }
    case RULE_SPATIAL4_SURPRISE:
        return (cls * SPATIAL4_LEVELS + spatial4) * U_BINS + ubin;
    case RULE_HOMOG_SURPRISE:
        return (cls * HOMOGENEITY_LEVELS + homog) * U_BINS + ubin;
    /* ddm_fx5: E1's four new rules.  Each is transcribed from the matching closure
     * in ``fx2_model_axis_corrector.fx2_family_specs`` and is pure int64 index
     * arithmetic -- exact on every conforming platform, no transcendental. */
    case RULE_HOMOG_BOUNDARY_SURPRISE: {
        int64_t head = (cls * HOMOGENEITY_LEVELS + homog) * BOUNDARY_LEVELS + boundary;
        return head * U_BINS + ubin;
    }
    case RULE_SPATIAL4_BOUNDARY:
        return (cls * SPATIAL4_LEVELS + spatial4) * BOUNDARY_LEVELS + boundary;
    case RULE_HOMOG_SPATIAL4:
        return (cls * HOMOGENEITY_LEVELS + homog) * SPATIAL4_LEVELS + spatial4;
    case RULE_SPATIAL4_TEMPORAL: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * SPATIAL4_LEVELS + spatial4;
    }
    /* ddm_gb1: transcribed from fx2 ``groupbin8_surprise``.  Pure int64 index
     * arithmetic; all operands are non-negative so C division == Python //. */
    case RULE_GROUPBIN8_SURPRISE:
        return (cls * GROUP_BINS + groupbin8) * U_BINS + ubin;
    case RULE_CLS_GROUPBIN8:
        return cls * GROUP_BINS + groupbin8;
    case RULE_PATCH192_ONLY:
        return patch192;
    case RULE_TILE48_GROUPBIN8:
        return tile48_groupbin8;
    default:
        return 0;
    }
}

int64_t f26_tile48_groupbin8_context(int64_t x, int64_t y)
{
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return -1;
    int64_t tile48 = (y / 64) * (WIDTH / 64) + (x / 64);
    int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * GROUP_BINS) / 190;
    return tile48 * GROUP_BINS + groupbin8;
}

int f26_corrector_group_state(void *handle, const float *probability,
                              const int64_t *predicted, const int64_t *positions, int64_t n)
{
    Corrector *self = (Corrector *)handle;
    if (!self || n <= 0) {
        return -1;
    }
    if (ensure_capacity(self, n) != 0) {
        return -1;
    }
    self->n = n;

    for (int64_t i = 0; i < n; ++i) {
        double *row = &self->row64[i * NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; ++c) {
            row[c] = (double)probability[i * NUM_CLASSES + c];
        }
        /* numpy argmax returns the FIRST maximum, so the comparison is strict. */
        int64_t arg = 0;
        for (int c = 1; c < NUM_CLASSES; ++c) {
            if (row[c] > row[arg]) {
                arg = c;
            }
        }
        double p_max = row[arg];
        double one_minus = 1.0 - p_max;
        if (!(one_minus > PROB_EPS)) {
            one_minus = PROB_EPS; /* np.maximum */
        }

        int64_t below = searchsorted_left(SURPRISE_ASC, U_BINS - 1, one_minus);
        int64_t ubin = clamp_i64((U_BINS - 1) - below, 0, U_BINS - 1);

        int64_t base_class = predicted[i];
        int64_t flat = positions[i];
        int64_t agree1 = 0;
        int64_t agree2 = 0;
        if (self->have_prev) {
            agree1 = ((int64_t)self->prev1[flat] == base_class) ? 1 : 0;
            agree2 = ((int64_t)self->prev2[flat] == base_class) ? 1 : 0;
        }
        int64_t run = min_i64(self->run[flat], RUN_LEVELS - 1);

        int64_t head = ((base_class * U_BINS + ubin) * 2 + agree1) * 2 + agree2;
        int64_t context = (head * RUN_LEVELS + run) * BOUNDARY_LEVELS + self->boundary[flat];

        self->arg[i] = arg;
        self->p_max[i] = p_max;
        self->one_minus[i] = one_minus;
        self->p_max_q[i] = (int64_t)rint(p_max * PHAT_SCALE);
        self->flat[i] = flat;

        /* fx1/fx2 unpack the sub-features straight back out of the packed context.  It is
         * reproduced rather than short-circuited: the round trip is only an identity while
         * every component is inside its modulus, and reproducing it faithfully means this
         * file agrees with numpy even if some future boundary bucket were not. */
        int64_t packed = context;
        int64_t boundary_f = packed % BOUNDARY_LEVELS;
        int64_t rest = packed / BOUNDARY_LEVELS;
        int64_t run_f = rest % RUN_LEVELS;
        rest /= RUN_LEVELS;
        int64_t agree2_f = rest % 2;
        rest /= 2;
        int64_t agree1_f = rest % 2;
        rest /= 2;
        int64_t ubin_f = rest % U_BINS;
        int64_t cls = rest / U_BINS;

        /* fx2._causal_neighbours: the widened four-neighbour template. */
        int64_t x = flat % WIDTH;
        int64_t y = flat / WIDTH;
        int64_t classes[N_CAUSAL];
        int available[N_CAUSAL];
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            int64_t nx = x + CAUSAL_DX[slot];
            int64_t ny = y + CAUSAL_DY[slot];
            int inside = (nx >= 0) && (nx < WIDTH) && (ny >= 0) && (ny < HEIGHT);
            int64_t cy = clamp_i64(ny, 0, HEIGHT - 1);
            int64_t cx = clamp_i64(nx, 0, WIDTH - 1);
            int64_t neighbour = cy * WIDTH + cx;
            available[slot] = inside && self->known[neighbour];
            classes[slot] = available[slot] ? (int64_t)self->current[neighbour] : -1;
        }

        /* fx2._spatial4_level */
        int64_t agreeing = 0;
        int any_available = 0;
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            if (available[slot]) {
                any_available = 1;
                if (classes[slot] == base_class) {
                    agreeing += 1;
                }
            }
        }
        int64_t spatial4 =
            any_available ? min_i64(agreeing + 1, SPATIAL4_LEVELS - 1) : 0;

        /* fx2._homogeneity_level: distinct decoded classes among available neighbours. */
        int present[NUM_CLASSES] = {0, 0, 0, 0, 0};
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            if (!available[slot]) {
                continue;
            }
            for (int value = 0; value < NUM_CLASSES; ++value) {
                if (classes[slot] == value) {
                    present[value] = 1;
                }
            }
        }
        int64_t distinct = 0;
        for (int value = 0; value < NUM_CLASSES; ++value) {
            distinct += present[value];
        }
        int64_t homog = min_i64(distinct, HOMOGENEITY_LEVELS - 1);

        /* fx1._spatial_level: the narrow left/up template the inherited members still read. */
        int64_t left = flat - 1 > 0 ? flat - 1 : 0;
        int64_t up = flat - WIDTH > 0 ? flat - WIDTH : 0;
        int has_left = (x > 0) && self->known[left];
        int has_up = (y > 0) && self->known[up];
        int agree_left = has_left && ((int64_t)self->current[left] == base_class);
        int agree_up = has_up && ((int64_t)self->current[up] == base_class);
        int64_t available2 = (int64_t)has_left + (int64_t)has_up;
        int64_t agreeing2 = (int64_t)agree_left + (int64_t)agree_up;
        int64_t spatial = (available2 == 0) ? 0 : agreeing2 + 1;

        /* ddm_gb1: the decode-scan group index, binned to GROUP_BINS levels.
         * g(x, y) = (x mod 64) + 2 * (y mod 64) is the shipped group plan; the
         * decoder selects the position BEFORE it decodes the symbol there, so
         * this is causal by construction and costs zero transmitted bytes. */
        int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * 8) / 190;
        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);
        int64_t tile48_groupbin8 = f26_tile48_groupbin8_context(x, y);
        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            self->fam_index[(int64_t)pos * self->capacity + i] =
                family_rule_index(self->families[pos].rule, cls, ubin_f, agree1_f, agree2_f,
                                  run_f, boundary_f, spatial, spatial4, homog, groupbin8,
                                  patch192, tile48_groupbin8);
        }

        /* mixer context ``cls_boundary_agree_homog_ubin8`` */
        int64_t ubin8 = min_i64(ubin_f >> 3, 7);
        int64_t mixer_head =
            ((cls * BOUNDARY_LEVELS + boundary_f) * 4 + agree1_f * 2 + agree2_f);
        self->mixer[i] = (mixer_head * HOMOGENEITY_LEVELS + homog) * 8 + ubin8;

        /* ma1._miss_cell (``nb3_prev1``): unavailable neighbours read UNKNOWN, a distinct
         * level rather than a fold into a real class. */
        int64_t nb[N_CAUSAL];
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            nb[slot] = available[slot] ? classes[slot] : UNKNOWN;
        }
        int64_t prev1_value = self->have_prev ? (int64_t)self->prev1[flat] : UNKNOWN;
        int64_t miss_head =
            (nb[SLOT_UP] * MISS_BASE + nb[SLOT_UPRIGHT]) * MISS_BASE + nb[SLOT_LEFT];
        self->miss_cell[i] = miss_head * MISS_BASE + prev1_value;
    }

    self->group_open = 1;
    return 0;
}

/* ma1._miss_multiplier, one cell. */
static inline void miss_multiplier(const Corrector *self, int64_t cell, double *out)
{
    if (self->miss_seen[cell] >= MISS_MIN_COUNT) {
        const int64_t *counts = &self->miss_counts[cell * NUM_CLASSES];
        const int64_t *expect = &self->miss_expect[cell * NUM_CLASSES];
        for (int k = 0; k < NUM_CLASSES; ++k) {
            double ratio = ((double)counts[k] + MISS_KT_ALPHA) /
                           ((double)expect[k] / PHAT_SCALE + MISS_KT_ALPHA);
            out[k] = clamp_double(ratio, MISS_CLAMP_LOW, MISS_CLAMP_HIGH);
        }
    } else {
        for (int k = 0; k < NUM_CLASSES; ++k) {
            out[k] = 1.0;
        }
    }
}

int f26_corrector_coding_row(void *handle, float *output, int64_t n)
{
    Corrector *self = (Corrector *)handle;
    if (!self || !self->group_open || n != self->n) {
        return -1;
    }

    for (int64_t i = 0; i < n; ++i) {
        /* fx1.odds_multiplier -- the weighted GEOMETRIC blend, member 0..12 in order.
         * Member-inner is the reordering that removes numpy's temporaries; the PRODUCT
         * order is preserved, which is the part float arithmetic is sensitive to. */
        double blended = 1.0;
        int64_t mixer_index = self->mixer[i];
        int64_t weight_index = mixer_index; /* count_buckets == 1 */
        const int64_t *weight_row = &self->weights[weight_index * N_FAMILIES];

        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            const Family *family = &self->families[pos];
            int64_t index = self->fam_index[(int64_t)pos * self->capacity + i];
            double multiplier = family_multiplier(family, index);
            int64_t grid_weight = round_shift(weight_row[pos], STORE_SHIFT);

            /* ``learn`` is True in the shipped config, so the radicals are ALWAYS taken --
             * the learner needs the stretch at every position, including those the
             * transport leaves untouched. */
            double radicals[POWER_BITS];
            double root = multiplier;
            for (int r = 0; r < POWER_BITS; ++r) {
                root = sqrt(root);
                radicals[r] = root;
            }
            /* stretch_from_radical: ~ln(m), with no logarithm. */
            self->stretch[(int64_t)pos * self->capacity + i] =
                (radicals[POWER_BITS - 1] - 1.0) * (double)(1 << POWER_BITS);

            blended = blended * dyadic_power(multiplier, radicals, grid_weight);
        }
        blended = clamp_double(blended, ODDS_LOW, ODDS_HIGH);
        self->blended[i] = blended;

        /* fx2.coding_row (sse is None under the frozen config, so no second stage). */
        double p_max = self->p_max[i];
        double one_minus = self->one_minus[i];
        double shifted = p_max * blended;
        double q = clamp_double(shifted / (shifted + one_minus), PROB_EPS, 1.0 - PROB_EPS);
        self->q[i] = q;

        const double *row = &self->row64[i * NUM_CLASSES];
        double *out = &self->out64[i * NUM_CLASSES];
        int64_t arg = self->arg[i];
        if (blended != 1.0) {
            double scale = (1.0 - q) / one_minus;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                out[c] = row[c] * scale;
            }
            out[arg] = q;
        } else {
            for (int c = 0; c < NUM_CLASSES; ++c) {
                out[c] = row[c];
            }
        }
        /* The float32 narrowing is REAL and observable: ma1 reads this row back as float64,
         * so the round trip is part of the shipped arithmetic, not a presentation step. */
        float narrowed[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; ++c) {
            narrowed[c] = (float)out[c];
        }

        /* ma1.coding_row -- the mass-preserving within-miss reweight. */
        double m[NUM_CLASSES];
        miss_multiplier(self, self->miss_cell[i], m);
        m[arg] = 1.0;
        int active = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            if (m[c] != 1.0) {
                active = 1;
            }
        }
        if (!active) {
            for (int c = 0; c < NUM_CLASSES; ++c) {
                output[i * NUM_CLASSES + c] = narrowed[c];
            }
            continue;
        }

        double row64b[NUM_CLASSES];
        double weighted[NUM_CLASSES];
        double base[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; ++c) {
            row64b[c] = (double)narrowed[c];
            weighted[c] = row64b[c] * m[c];
            base[c] = row64b[c];
        }
        weighted[arg] = 0.0;
        base[arg] = 0.0;
        /* Lane by lane in a FIXED order, so the reduction is unambiguous rather than
         * dependent on a library's pairwise blocking. */
        double big_w = 0.0;
        double big_s = 0.0;
        for (int lane = 0; lane < NUM_CLASSES; ++lane) {
            big_w += weighted[lane];
            big_s += base[lane];
        }
        if (!(big_w > 0.0) || !(big_s > 0.0)) {
            for (int c = 0; c < NUM_CLASSES; ++c) {
                output[i * NUM_CLASSES + c] = narrowed[c];
            }
            continue;
        }
        double scale2 = big_s / big_w;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            output[i * NUM_CLASSES + c] = (float)(weighted[c] * scale2);
        }
        output[i * NUM_CLASSES + arg] = (float)row64b[arg];
    }
    return 0;
}

/* MixerFamily.observe's recency window: halve REPEATEDLY, so a large group cannot leave a
 * bin permanently above its limit and silently degrade the member into a duplicate of the
 * cumulative one. */
static inline void family_halve(Family *family, int64_t index)
{
    if (!family->count_limit) {
        return;
    }
    for (int pass = 0; pass < COUNT_HALVING_PASSES; ++pass) {
        if (family->counts[index] <= family->count_limit) {
            break;
        }
        family->counts[index] >>= 1;
        family->hits[index] >>= 1;
        family->phat_q[index] >>= 1;
    }
}

int f26_corrector_observe(void *handle, const int64_t *symbols, int64_t n)
{
    Corrector *self = (Corrector *)handle;
    if (!self || !self->group_open || n != self->n) {
        return -1;
    }

    /* ma1.observe runs FIRST, exactly as the MRO drives it.  The tables are disjoint from
     * the mixer's, so ordering does not change values -- it is preserved anyway, because a
     * reader should not have to prove that to follow the code. */
    for (int64_t i = 0; i < n; ++i) {
        int64_t arg = self->arg[i];
        int64_t decoded = symbols[i];
        if (decoded == arg) {
            continue;
        }
        int64_t cell = self->miss_cell[i];
        const double *row = &self->row64[i * NUM_CLASSES];
        double one_minus = self->one_minus[i];
        int64_t *expect = &self->miss_expect[cell * NUM_CLASSES];
        for (int k = 0; k < NUM_CLASSES; ++k) {
            double relative = (k == arg) ? 0.0 : row[k] / one_minus;
            expect[k] += (int64_t)rint(relative * PHAT_SCALE);
        }
        self->miss_counts[cell * NUM_CLASSES + decoded] += 1;
        self->miss_seen[cell] += 1;
    }

    /* fx1.observe: the learner steps BEFORE the members fold this group in, so the gradient
     * is taken against the statistics that actually produced the row. */
    memset(self->ws_counts, 0, (size_t)N_WEIGHT_SETS * sizeof(int64_t));
    for (int64_t i = 0; i < n; ++i) {
        self->ws_counts[self->mixer[i]] += 1;
        int64_t hit = (symbols[i] == self->arg[i]) ? 1 : 0;
        self->hit[i] = hit;
        /* numpy forms the residual ONCE, outside the member loop; hoisting it here keeps
         * that structure rather than recomputing an identical rint 13 times. */
        self->residual[i] = (int64_t)rint(((double)hit - self->q[i]) * ERR_SCALE);
    }

    for (int pos = 0; pos < N_FAMILIES; ++pos) {
        memset(self->ws_gradient, 0, (size_t)N_WEIGHT_SETS * sizeof(int64_t));
        const double *stretch = &self->stretch[(int64_t)pos * self->capacity];
        for (int64_t i = 0; i < n; ++i) {
            double quantised_d =
                clamp_double(rint(stretch[i] * STRETCH_SCALE), -STRETCH_CLAMP, STRETCH_CLAMP);
            self->ws_gradient[self->mixer[i]] += self->residual[i] * (int64_t)quantised_d;
        }
        for (int64_t ws = 0; ws < N_WEIGHT_SETS; ++ws) {
            int64_t gradient = self->ws_gradient[ws];
            int64_t count = self->ws_counts[ws];
            /* ``normalize`` is True: numpy's ``//`` FLOORS, and the gradient is negative
             * about half the time, so truncation here would be a real bias. */
            gradient = (count > 0) ? floor_div_i64(gradient, count) : 0;
            int64_t step = round_shift(gradient, LR_SHIFT);
            int64_t *slot = &self->weights[ws * N_FAMILIES + pos];
            *slot = clamp_i64(*slot + step, WEIGHT_LOW, WEIGHT_HIGH);
        }
    }

    for (int pos = 0; pos < N_FAMILIES; ++pos) {
        Family *family = &self->families[pos];
        const int64_t *indices = &self->fam_index[(int64_t)pos * self->capacity];
        for (int64_t i = 0; i < n; ++i) {
            int64_t index = indices[i];
            family->counts[index] += 1;
            family->hits[index] += self->hit[i];
            family->phat_q[index] += self->p_max_q[i];
        }
        if (family->count_limit) {
            /* numpy takes np.unique first; halving is per-cell and idempotent once the cell
             * is at or below the limit, so visiting a duplicate index costs a comparison and
             * changes nothing. */
            for (int64_t i = 0; i < n; ++i) {
                family_halve(family, indices[i]);
            }
        }
    }

    for (int64_t i = 0; i < n; ++i) {
        int64_t flat = self->flat[i];
        self->current[flat] = (uint8_t)symbols[i];
        self->known[flat] = 1;
    }
    self->group_open = 0;
    return 0;
}

int f26_corrector_end_frame(void *handle, const uint8_t *tokens, int64_t size)
{
    Corrector *self = (Corrector *)handle;
    if (!self || size != self->plane) {
        return -1;
    }
    if (self->have_prev) {
        for (int64_t i = 0; i < self->plane; ++i) {
            if (tokens[i] == self->prev1[i]) {
                int64_t next = self->run[i] + 1;
                self->run[i] = next < RUN_CAP ? next : RUN_CAP;
            } else {
                self->run[i] = 0;
            }
        }
        memcpy(self->prev2, self->prev1, (size_t)self->plane * sizeof(uint8_t));
    }
    memcpy(self->prev1, tokens, (size_t)self->plane * sizeof(uint8_t));
    self->have_prev = 1;
    return 0;
}

/* --- introspection, for the parity harness only ----------------------------------------- */

int32_t f26_corrector_abi_version(void) { return F26C_ABI_VERSION; }

/* Expose the live table set so the differential harness can compare STATE, not just output.
 * An output-only comparison would pass for a long time on a corrector whose tables have
 * already diverged, because a cold cell emits exactly 1.0 either way. */
int f26_corrector_table(void *handle, int32_t which, int32_t position, int64_t *out,
                        int64_t capacity, int64_t *size_out)
{
    Corrector *self = (Corrector *)handle;
    if (!self) {
        return -1;
    }
    const int64_t *source = NULL;
    int64_t size = 0;
    switch (which) {
    case 0: /* family counts */
    case 1: /* family hits */
    case 2: /* family phat_q */
        if (position < 0 || position >= N_FAMILIES) {
            return -1;
        }
        size = self->families[position].size;
        source = which == 0   ? self->families[position].counts
                 : which == 1 ? self->families[position].hits
                              : self->families[position].phat_q;
        break;
    case 3:
        source = self->weights;
        size = (int64_t)N_WEIGHT_SETS * N_FAMILIES;
        break;
    case 4:
        source = self->miss_counts;
        size = (int64_t)N_MISS_CELLS * NUM_CLASSES;
        break;
    case 5:
        source = self->miss_expect;
        size = (int64_t)N_MISS_CELLS * NUM_CLASSES;
        break;
    case 6:
        source = self->miss_seen;
        size = N_MISS_CELLS;
        break;
    case 7:
        source = self->run;
        size = self->plane;
        break;
    default:
        return -1;
    }
    if (size_out) {
        *size_out = size;
    }
    if (!out) {
        return 0;
    }
    if (capacity < size) {
        return -1;
    }
    memcpy(out, source, (size_t)size * sizeof(int64_t));
    return 0;
}
