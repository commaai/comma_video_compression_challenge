/* tcm.c - Token Context-Mixing coder, prototype v1 (internal measurement tool).
 *
 * Lossless coder for the 600 x 384 x 512 token maps (classes 0..4, map-major raster uint8).
 *   ./tcm e in.u8 out.tcm [key=value ...]     encode (params stored in the header)
 *   ./tcm d in.tcm out.u8                     decode
 *
 * Everything that affects coded bits is integer / fixed point (tables from gen_tables1.py are constants),
 * so encoder and decoder are bit-identical on any machine.  -DSTATS adds float diagnostics (encoder only).
 *
 * Probability path:  counters 32-bit p + 10-bit hit count (rate 1/(n+1.5), n <= limit)
 *                    stretch = exact-ish integer logit (clz + 4096-entry log2 table), Q8 nats, clamp +-4095
 *                    mixers: int32 Q16 weights, int64 dot;  squash table -> 32-bit probability
 *                    final clamp [2^12, 2^32-2^12]  (floor 2^-20 per binary decision)
 *                    coder: 64-bit carryless binary arithmetic coder, 32-bit probabilities.
 *
 * Binarisation per pixel (bin=0, the S0 spec): node0 "tok == c1" with c1 = co-located previous-map token;
 * node1 "tok == c2" with c2 = first of W,N,NE,NW,... that differs from c1; nodes 2,3 = binary tree over the
 * remaining 3 classes (ascending).  bin=1: c1 = W;  bin=2: c1 = local-motion-compensated prev;  bin=3: global-shift prev.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "tcm1_tables.h"
#ifdef STATS
#include <math.h>
#endif

#define H 384
#define W 512
#define B 8                      /* padding border of the padded map buffers */
#define PW (W + 2 * B)
#define PH (H + 2 * B)
#define OOB 5
#define NONE 9

/* ------------------------------------------------------------------ parameters (stored in header) */
enum { P_STAGE, P_LR, P_LIMIT, P_BIN, P_HB, P_APMR, P_LR2, P_APMW, P_W0, P_MV, P_GS, P_TYP, P_MM, P_MDY, P_MW, P_C2O, P_APMC, P_FSEL, P_DUAL, P_M5, P_M6, P_M7, P_SM, P_SML, P_M8, P_M9, P_M10, P_M11, P_M12, P_M13, P_LIMX, P_M14, P_M15, P_M16, P_M17, P_M18, P_M19, P_M20, P_MLRD, P_APMN, P_M21, P_M22, P_M23, P_A1B, P_F2, P_F3, P_M24, P_F4, P_MC, P_MCX, P_MCH, P_MCC, P_MIXQ, P_MCR, P_MCS, P_MC2, P_MC3, P_MCL, P_M26, P_C2M, P_M27, P_MCSR, P_HBS, P_MCV, P_MCD, P_M28, P_APM4, P_MCW, P_MLR, P_LD, P_APM5, P_M30, P_OBJ, P_OBJC, P_BD, P_OBJM, P_F5, P_FMIX, P_APM6, P_APM7, P_APM9, P_XC, P_APM10, P_PINIT, P_FMIXLR, P_MCSW, NPARAM_USED };
#define NPARAM 96
static int prm[NPARAM];
static uint64_t mask_off = 0;     /* bit k set = context k disabled (ablation); stored in header */
static const char *pname[NPARAM_USED] = {"stage", "lr", "limit", "bin", "hb", "apmr", "lr2", "apmw", "w0", "mv", "gs", "typ", "mm", "mdy", "mw", "c2o", "apmc", "fsel", "dual", "m5", "m6", "m7", "sm", "sml", "m8", "m9", "m10", "m11", "m12", "m13", "limx", "m14", "m15", "m16", "m17", "m18", "m19", "m20", "mlrd", "apmn", "m21", "m22", "m23", "a1b", "f2", "f3", "m24", "f4", "mc", "mcx", "mch", "mcc", "mixq", "mcr", "mcs", "mc2", "mc3", "mcl", "m26", "c2m", "m27", "mcsr", "hbs", "mcv", "mcd", "m28", "apm4", "mcw", "mlr", "ld", "apm5", "m30", "obj", "objc", "bd", "objm", "f5", "fmix", "apm6", "apm7", "apm9", "xc", "apm10", "pinit", "fmixlr", "mcsw"};
static void prm_default(void) {
    memset(prm, 0, sizeof prm);
    prm[P_STAGE] = 5; prm[P_LR] = 24; prm[P_LIMIT] = 255;  /* must be 4k+3 */ prm[P_BIN] = 0; prm[P_HB] = 20; prm[P_APMR] = 6;
    prm[P_LR2] = 8; prm[P_SML] = 255; prm[P_MCD] = 6; prm[P_OBJC] = 3; prm[P_APMW] = 2; prm[P_W0] = 20; prm[P_MV] = 3; prm[P_GS] = 3; prm[P_TYP] = 0; prm[P_MM] = 0; prm[P_MDY] = 1; prm[P_MW] = 2; prm[P_C2O] = 0;
}

/* ------------------------------------------------------------------ arithmetic coder (64-bit, carryless) */
typedef struct {
    uint64_t x1, x2, x;
    uint8_t *buf; size_t pos, cap;
    const uint8_t *in; size_t in_len, in_pos;
    int dec;
} Coder;
static void put_byte(Coder *c, uint8_t b) {
    if (c->pos == c->cap) { c->cap = c->cap ? c->cap * 2 : 1 << 20; c->buf = realloc(c->buf, c->cap); }
    c->buf[c->pos++] = b;
}
static inline uint8_t get_byte(Coder *c) { return c->in_pos < c->in_len ? c->in[c->in_pos++] : 0; }
/* p = P(bit==1) as 32-bit fraction, in [1, 2^32-1] */
static inline int code_bit(Coder *c, int bit, uint32_t p) {
    uint64_t xmid = c->x1 + (uint64_t)(((unsigned __int128)(c->x2 - c->x1) * p) >> 32);
    if (c->dec) bit = c->x <= xmid;
    if (bit) c->x2 = xmid; else c->x1 = xmid + 1;
    while (((c->x1 ^ c->x2) >> 56) == 0) {
        if (c->dec) c->x = (c->x << 8) | get_byte(c);
        else put_byte(c, (uint8_t)(c->x2 >> 56));
        c->x1 <<= 8; c->x2 = (c->x2 << 8) | 255;
    }
    return bit;
}
static void coder_flush(Coder *c) {   /* shortest byte string v (zero padded) with x1 <= v <= x2 */
    for (int k = 1; k <= 8; k++) {
        int sh = 64 - 8 * k;
        uint64_t v = sh ? ((c->x1 >> sh) << sh) : c->x1;
        if (v < c->x1) { uint64_t inc = (uint64_t)1 << sh; if (v + inc < v) continue; v += inc; }
        if (v <= c->x2) { for (int i = 0; i < k; i++) put_byte(c, (uint8_t)(v >> (56 - 8 * i))); return; }
    }
}

/* ------------------------------------------------------------------ probability primitives */
static inline int32_t lg2q16(uint32_t x) {          /* log2(x) in Q16, x >= 1 */
    int k = 31 - __builtin_clz(x);
    uint32_t m = k >= 12 ? (x >> (k - 12)) & 4095 : (x << (12 - k)) & 4095;
    return (k << 16) + (int32_t)log2_tab[m];
}
static inline int clamp_s(int s) { return s < -4095 ? -4095 : (s > 4095 ? 4095 : s); }
static inline int stretch(uint32_t p) {             /* ln(p/(1-p)) in Q8, p 32-bit */
    uint32_t q = 0u - p;
    if (p == 0) p = 1;
    if (q == 0) q = 1;
    int32_t d = lg2q16(p) - lg2q16(q);
    return clamp_s((int)(((int64_t)d * 45426) >> 24));   /* * ln2 * 256 / 65536 */
}
static inline uint32_t squash(int s) { return squash_tab[clamp_s(s) + 4095]; }

static int LIMIT = 255;
typedef struct { uint32_t p[4]; uint16_t n[4]; uint32_t chk; uint32_t pad; } Bucket;   /* 32 bytes, 4 nodes */
static inline void ctr_update(uint32_t *p, uint16_t *n, int bit) {
    uint32_t r = rate_tab[*n];
    if (bit) *p += (uint32_t)(((uint64_t)(~*p) * r) >> 16);
    else *p -= (uint32_t)(((uint64_t)*p * r) >> 16);
    if (*n < LIMIT) (*n)++;
}
static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 31; x *= 0x7fb5d329728ea185ULL; x ^= x >> 27; x *= 0x81dadef4bc2dd44dULL; x ^= x >> 33;
    return x;
}
typedef struct { Bucket *t; int bits; uint32_t prior[4]; } HashTab;
static int PINIT = 0, PINIT_R = 10;   /* pinit=1: new context slots start from the context type's running bit rate (per node) instead of 1/2 */
static Bucket *ht_lookup(HashTab *h, uint64_t ctx) {
    uint64_t v = hash64(ctx);
    Bucket *b = &h->t[v >> (64 - h->bits)];
    uint32_t chk = (uint32_t)v | 1;
    if (b->chk != chk) {
        for (int i = 0; i < 4; i++) { b->p[i] = PINIT ? h->prior[i] : 1u << 31; b->n[i] = 0; }
        b->pad = 1u << 31;
        b->chk = chk;
    }
    return b;
}

/* ------------------------------------------------------------------ mixer */
#define MAXIN 128
typedef struct { int32_t *w; uint16_t *cnt; int n, nsets, lr; } Mixer;
static int MLRD = 0, MIXQ = 0;   /* per-weight-set learning-rate boost lr*(1 + MLRD*32/(32+hits)), 0 = off */
static void mixer_init(Mixer *m, int n, int nsets, int lr, int32_t w0) {
    m->n = n; m->nsets = nsets; m->lr = lr; m->w = malloc(sizeof(int32_t) * n * nsets); m->cnt = calloc(nsets, 2);
    for (int i = 0; i < n * nsets; i++) m->w[i] = w0;
}
static inline int mixer_dot(const Mixer *m, int set, const int *x) {
    const int32_t *w = m->w + (size_t)set * m->n; int64_t d = 0;
    for (int i = 0; i < m->n; i++) d += (int64_t)w[i] * x[i];
    return clamp_s((int)(d >> 16));
}
static inline void mixer_update(Mixer *m, int set, const int *x, int bit, uint32_t p) {
    int32_t *w = m->w + (size_t)set * m->n;
    int64_t err = ((int64_t)bit << 32) - (int64_t)p;        /* +-2^32 */
    int64_t e = (err >> 8) * m->lr;                          /* +-2^24 * lr */
    if (MLRD) { int h = m->cnt[set]; e += e * MLRD * 32 / (32 + h); if (h < 65535) m->cnt[set] = (uint16_t)(h + 1); }
    if (MIXQ) {   /* mixq=1: 32-bit update (vectorisable): e16 = e >> 16 (|e16| < 2^18), w += (x * e16) >> 14 */
        int32_t e16 = (int32_t)(e >> 16);
        if (e16 > 262143) e16 = 262143; if (e16 < -262143) e16 = -262143;
        for (int i = 0; i < m->n; i++) w[i] += (x[i] * e16) >> 14;
    } else
        for (int i = 0; i < m->n; i++) w[i] += (int32_t)((x[i] * e) >> 30);
}

/* ------------------------------------------------------------------ APM / SSE (33 knots over the stretch domain) */
#define APM_N 33
typedef struct { uint32_t *t; uint8_t *c; int idx, frac, rate; } APM;
static int APMN = 0;   /* 1: count-based APM step max(1/(n+1.5), 2^-rate) per knot */
static void apm_init(APM *a, int nctx, int rate) {
    a->rate = rate; a->t = malloc(sizeof(uint32_t) * (size_t)nctx * APM_N); a->c = calloc((size_t)nctx * APM_N, 1);
    for (int c = 0; c < nctx; c++) for (int j = 0; j < APM_N; j++) a->t[(size_t)c * APM_N + j] = squash((j - 16) * 256);
}
static inline uint32_t apm_p(APM *a, int ctx, int s) {
    int v = clamp_s(s) + 4096; int j = v >> 8, f = v & 255;
    if (j >= APM_N - 1) { j = APM_N - 2; f = 256; }
    a->idx = ctx * APM_N + j; a->frac = f;
    return (uint32_t)(((uint64_t)a->t[a->idx] * (256 - f) + (uint64_t)a->t[a->idx + 1] * f) >> 8);
}
static inline void apm_update(APM *a, int bit) {
    int64_t tgt = bit ? 0xffffffffLL : 0;
    uint32_t *t = a->t + a->idx;
    int64_t d0, d1;
    if (APMN) {
        uint8_t *c = a->c + a->idx; int64_t fl = 65536 >> a->rate;
        int64_t r0 = rate_tab[c[0]] > fl ? rate_tab[c[0]] : fl, r1 = rate_tab[c[1]] > fl ? rate_tab[c[1]] : fl;
        d0 = ((tgt - (int64_t)t[0]) * (256 - a->frac) >> 8) * r0 >> 16;
        d1 = ((tgt - (int64_t)t[1]) * a->frac >> 8) * r1 >> 16;
        if (a->frac < 192 && c[0] < 255) c[0]++;
        if (a->frac > 64 && c[1] < 255) c[1]++;
    } else {
        d0 = ((tgt - (int64_t)t[0]) * (256 - a->frac)) >> (8 + a->rate);
        d1 = ((tgt - (int64_t)t[1]) * a->frac) >> (8 + a->rate);
    }
    t[0] = (uint32_t)((int64_t)t[0] + d0); t[1] = (uint32_t)((int64_t)t[1] + d1);
}

/* ------------------------------------------------------------------ maps / features */
static inline int AT(const uint8_t *m, int y, int x) { return m[(y + B) * PW + x + B]; }  /* padded access */

/* distance (4-neighbour growth, capped at 4) to the nearest class boundary of map t (padded) */
static void boundary_buckets(const uint8_t *t, uint8_t *out) {
    static uint8_t act[H * W], nxt[H * W];
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int v = AT(t, y, x), e = 0;
        if (y > 0 && AT(t, y - 1, x) != v) e = 1;
        if (y < H - 1 && AT(t, y + 1, x) != v) e = 1;
        if (x > 0 && AT(t, y, x - 1) != v) e = 1;
        if (x < W - 1 && AT(t, y, x + 1) != v) e = 1;
        act[y * W + x] = (uint8_t)e; out[y * W + x] = e ? 0 : 4;
    }
    for (int d = 1; d < 4; d++) {
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            int i = y * W + x;
            nxt[i] = act[i] | (y > 0 && act[i - W]) | (y < H - 1 && act[i + W]) | (x > 0 && act[i - 1]) | (x < W - 1 && act[i + 1]);
        }
        for (int i = 0; i < H * W; i++) { if (out[i] == 4 && nxt[i]) out[i] = (uint8_t)d; act[i] = nxt[i]; }
    }
}

/* signed offset (-4..4) of the transition nearest x in row yy of padded map m (transition at j: m[j] != m[j-1]),
 * returned +4 (0..8), NONE (9) if none within 4. For the current map only rows < y may be passed. */
static inline int edge_off(const uint8_t *m, int yy, int x) {
    if (yy < 0 || yy >= H) return NONE;
    const uint8_t *r = m + (yy + B) * PW + B;
    for (int d = 0; d <= 4; d++) {
        int j = x + d; if (j >= 1 && j < W && r[j] != r[j - 1]) return d + 4;
        j = x - d; if (d && j >= 1 && j < W && r[j] != r[j - 1]) return -d + 4;
    }
    return NONE;
}

/* decoder-side local motion (per row, from the previous decoded row):  for every x choose (dx,dy) in
 * [-MV,MV]x[-1,1] minimising mismatches between cur row y-1 and pv row y-1+dy shifted by dx over x-6..x+6. */
#define MWIN 6
static int8_t mc_dx[W], mc_dy[W];
static void motion_row(const uint8_t *cur, const uint8_t *pv, int y, int mv) {
    static int32_t pre[64][W + 1];
    int dxs[64], dys[64], no = 0;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -mv; dx <= mv; dx++, no++) {
        dxs[no] = dx; dys[no] = dy; pre[no][0] = 0;
        for (int x = 0; x < W; x++) pre[no][x + 1] = pre[no][x] + (AT(cur, y - 1, x) != AT(pv, y - 1 + dy, x + dx));
    }
    for (int x = 0; x < W; x++) {
        int a = x - MWIN < 0 ? 0 : x - MWIN, b = x + MWIN + 1 > W ? W : x + MWIN + 1;
        int best = 0, bc = 1 << 30, bp = 1 << 30;
        for (int o = 0; o < no; o++) {
            int cst = pre[o][b] - pre[o][a];
            int pen = abs(dxs[o]) + abs(dys[o]);
            if (cst < bc || (cst == bc && pen < bp)) { bc = cst; bp = pen; best = o; }
        }
        mc_dx[x] = (int8_t)dxs[best]; mc_dy[x] = (int8_t)dys[best];
    }
}

/* per-pixel causal motion (mm=1): window = cur rows y-2,y-1 at x-6..x+6 plus cur row y at x-6..x-1 (weight mw),
 * offsets dx in [-mv,mv], dy in [-mdy,mdy]; ties prefer the smaller |dx|+|dy|.  Uses decoded pixels only. */
static int32_t mp1[128][W + 1], mp2[128][W + 1], mp0[128][W + 1];
static int mo_dx[128], mo_dy[128], mo_pen[128], mo_n;
static void motion_pix_row(const uint8_t *cur, const uint8_t *pv, int y, int mv, int mdy) {
    mo_n = 0;
    for (int dy = -mdy; dy <= mdy; dy++) for (int dx = -mv; dx <= mv; dx++) {
        int o = mo_n++; mo_dx[o] = dx; mo_dy[o] = dy; mo_pen[o] = abs(dx) + abs(dy);
        mp1[o][0] = mp2[o][0] = mp0[o][0] = 0;
        for (int x = 0; x < W; x++) {
            mp1[o][x + 1] = mp1[o][x] + (y >= 1 && AT(cur, y - 1, x) != AT(pv, y - 1 + dy, x + dx));
            mp2[o][x + 1] = mp2[o][x] + (y >= 2 && AT(cur, y - 2, x) != AT(pv, y - 2 + dy, x + dx));
        }
    }
}
static void motion_pix(const uint8_t *cur, const uint8_t *pv, int y, int x, int mw) {
    for (int o = 0; o < mo_n; o++)   /* add pixel x-1 of the current row */
        mp0[o][x] = x ? mp0[o][x - 1] + (AT(cur, y, x - 1) != AT(pv, y + mo_dy[o], x - 1 + mo_dx[o])) : 0;
    int a = x - MWIN < 0 ? 0 : x - MWIN, b = x + MWIN + 1 > W ? W : x + MWIN + 1;
    int best = 0, bc = 1 << 30, bp = 1 << 30;
    for (int o = 0; o < mo_n; o++) {
        int cst = mw * (mp0[o][x] - mp0[o][a]) + mp1[o][b] - mp1[o][a] + mp2[o][b] - mp2[o][a];
        if (cst < bc || (cst == bc && mo_pen[o] < bp)) { bc = cst; bp = mo_pen[o]; best = o; }
    }
    mc_dx[x] = (int8_t)mo_dx[best]; mc_dy[x] = (int8_t)mo_dy[best];
}

/* Matched edge transfer (per row, decoder side).  For every transition (j, L|R) of cur row y-1 find the nearest
 * transition with the same class pair in prev row y-1 (k1, |k1-j|<=XR1), then the same pair in prev row y nearest k1
 * (k0, |k0-k1|<=XR2):  temporal prediction of the row-y transition = k0 + (j - k1).  Spatial: same pair in cur row
 * y-2 nearest j (j2) -> 2j - j2.  Unmatched prev row-y transitions are added as temporal predictions (shifted by the
 * displacement of the nearest matched edge).  Per x we keep the offset of the nearest prediction and its pair. */
#define XR1 16
#define XR2 24
#define MAXT 512
static uint8_t xo_o[W], xp_o[W], xo_c2[W], xp_c2[W], xo_c[W], xp_c[W], xo_u[W], xp_u[W], xo_t[W], xp_t[W], xo_s[W], xp_s[W], xo_v[2][W], xp_v[2][W];
static int row_trans(const uint8_t *m, int yy, int *pos, int *pair) {
    int n = 0;
    if (yy < 0 || yy >= H) return 0;
    const uint8_t *r = m + (yy + B) * PW + B;
    for (int j = 1; j < W; j++) if (r[j] != r[j - 1]) { pos[n] = j; pair[n] = r[j - 1] * 6 + r[j]; n++; }
    return n;
}
static int off_bucket(int o) { return o < -8 ? 17 : o > 8 ? 18 : o + 8; }   /* 0..18, 19 = none */
static uint16_t xa_s[W], xa_t[W];
static void nearest_fill(const int *pp, const int *pq, int np, uint8_t *xo, uint8_t *xp, uint16_t *xa) {
    int k = 0;
    for (int x = 0; x < W; x++) {
        if (!np) { xo[x] = 19; xp[x] = 36; if (xa) xa[x] = 0xfff; continue; }
        while (k + 1 < np && abs(pp[k + 1] - x) <= abs(pp[k] - x)) k++;
        xo[x] = (uint8_t)off_bucket(pp[k] - x); xp[x] = (uint8_t)(pq[k] & 63); if (xa) xa[x] = (uint16_t)(pq[k] >> 6);
    }
}
static int match_near(const int *pj, const int *pq, int n, int j, int q, int r) {   /* nearest same-pair position or -1 */
    int best = -1, bd = r + 1;
    for (int t = 0; t < n; t++) if (pq[t] == q && abs(pj[t] - j) < bd) { bd = abs(pj[t] - j); best = pj[t]; }
    return best;
}
static int stepc(int d) { return d < -3 ? 0 : d > 3 ? 6 : d + 3; }
static void sort_preds(int *pp, int *pq, int n) {   /* insertion sort by position */
    for (int i = 1; i < n; i++) { int a = pp[i], b = pq[i], j = i - 1; while (j >= 0 && pp[j] > a) { pp[j + 1] = pp[j]; pq[j + 1] = pq[j]; j--; } pp[j + 1] = a; pq[j + 1] = b; }
}
static void edge_pred_row(const uint8_t *cur, const uint8_t *pv, int y, int dv, uint8_t *o_t, uint8_t *q_t, int full) {
    static int cj[MAXT], cq[MAXT], c2j[MAXT], c2q[MAXT], p1j[MAXT], p1q[MAXT], p0j[MAXT], p0q[MAXT], used[MAXT];
    static int tp[2 * MAXT], tq[2 * MAXT], sp[MAXT], sq[MAXT];
    static int c3j[MAXT], c3q[MAXT], c4j[MAXT], c4q[MAXT];
    int nc = row_trans(cur, y - 1, cj, cq), nc2 = row_trans(cur, y - 2, c2j, c2q);
    int nc3 = !full ? 0 : row_trans(cur, y - 3, c3j, c3q), nc4 = !full ? 0 : row_trans(cur, y - 4, c4j, c4q);
    static int pmj[MAXT], pmq[MAXT], ppj[MAXT], ppq[MAXT];
    int npm = !full ? 0 : row_trans(pv, y - 2, pmj, pmq), npp = !full ? 0 : row_trans(pv, y + 1, ppj, ppq);
    int np1 = row_trans(pv, y - 1 + dv, p1j, p1q), np0 = row_trans(pv, y + dv, p0j, p0q);
    int nt = 0, ns = 0;
    memset(used, 0, sizeof(int) * (np0 ? np0 : 1));
    int lastd = 0;
    for (int i = 0; i < nc; i++) {
        int j = cj[i], q = cq[i], k1 = -1, bd = XR1 + 1, ts = 7;
        for (int t = 0; t < np1; t++) if (p1q[t] == q && abs(p1j[t] - j) < bd) { bd = abs(p1j[t] - j); k1 = p1j[t]; }
        if (k1 >= 0) {
            int k0 = -1, bi = -1; bd = XR2 + 1;
            for (int t = 0; t < np0; t++) if (p0q[t] == q && abs(p0j[t] - k1) < bd) { bd = abs(p0j[t] - k1); k0 = p0j[t]; bi = t; }
            if (k0 >= 0) {
                int aux = 0;
                if (full) {   /* displacement + consistency with row y-2, prev edge's next step */
                    int j2 = match_near(c2j, c2q, nc2, j, q, 8), k2 = j2 >= 0 ? match_near(pmj, pmq, npm, k1, q, 8) : -1;
                    int kp = match_near(ppj, ppq, npp, k0, q, 8);
                    aux = stepc(j - k1) | (k2 >= 0 ? stepc((j - k1) - (j2 - k2)) : 7) << 3 | (kp >= 0 ? stepc(kp - k0) : 7) << 6;
                }
                tp[nt] = k0 + j - k1; tq[nt] = q | aux << 6; nt++; used[bi] = 1; lastd = j - k1; ts = stepc(k0 - k1); }
        }
        int j2 = -1; bd = XR2 + 1;
        for (int t = 0; t < nc2; t++) if (c2q[t] == q && abs(c2j[t] - j) < bd) { bd = abs(c2j[t] - j); j2 = c2j[t]; }
        int s1 = 7, s2 = 7, s3 = 7;
        if (full && j2 >= 0) {
            s1 = stepc(j - j2);
            int j3 = match_near(c3j, c3q, nc3, j2, q, 8);
            if (j3 >= 0) { s2 = stepc(j2 - j3); int j4 = match_near(c4j, c4q, nc4, j3, q, 8); if (j4 >= 0) s3 = stepc(j3 - j4); }
        }
        sp[ns] = j2 >= 0 ? 2 * j - j2 : j; sq[ns] = q | (s1 | s2 << 3 | s3 << 6 | ts << 9) << 6; ns++;
    }
    for (int t = 0; t < np0; t++) if (!used[t]) { tp[nt] = p0j[t] + lastd; tq[nt] = p0q[t] | 0x1ff << 6; nt++; }
    sort_preds(tp, tq, nt); sort_preds(sp, sq, ns);
    nearest_fill(tp, tq, nt, o_t, q_t, !full ? NULL : xa_t);
    if (full) nearest_fill(sp, sq, ns, xo_s, xp_s, xa_s);
}

/* In-row transfer: when a transition (x0, pair) occurs in the current row, match it to the same pair in prev rows
 * y-1, y, y+1 (nearest, |k-x0| <= 16) and predict the NEXT transition of the current row at k' + (x0 - k), k' = the
 * prev transition following k.  dv=0 prediction and the best-matching-dv prediction are kept. Row start acts as x0=0. */
static int ir_j[3][MAXT], ir_q[3][MAXT], ir_n[3];
static int ir_pred0 = -999, ir_q0 = 36, ir_predb = -999, ir_qb = 36, ir_dvb = 1;
static void inrow_reset(const uint8_t *pv, int y) {
    for (int d = 0; d < 3; d++) ir_n[d] = row_trans(pv, y - 1 + d, ir_j[d], ir_q[d]);
    ir_pred0 = ir_n[1] ? ir_j[1][0] : -999; ir_q0 = ir_n[1] ? ir_q[1][0] : 36;
    ir_predb = ir_pred0; ir_qb = ir_q0; ir_dvb = 1;
}
static void inrow_update(const uint8_t *row, int x) {     /* call after pixel x of the current row is known */
    if (x < 1 || row[x] == row[x - 1]) return;
    int q = row[x - 1] * 6 + row[x], bd = 1 << 30;
    ir_pred0 = -999; ir_q0 = 36; ir_predb = -999; ir_qb = 36; ir_dvb = 3;
    for (int d = 0; d < 3; d++) {
        int bi = -1, b = 17;
        for (int t = 0; t < ir_n[d]; t++) if (ir_q[d][t] == q && abs(ir_j[d][t] - x) < b) { b = abs(ir_j[d][t] - x); bi = t; }
        if (bi < 0) continue;
        int pr = bi + 1 < ir_n[d] ? ir_j[d][bi + 1] + x - ir_j[d][bi] : 2000, pq = bi + 1 < ir_n[d] ? ir_q[d][bi + 1] : 36;
        if (d == 1) { ir_pred0 = pr; ir_q0 = pq; }
        int cost = b * 4 + (d != 1);
        if (cost < bd) { bd = cost; ir_predb = pr; ir_qb = pq; ir_dvb = d; }
    }
}
static int ir_bucket(int pr, int x) { return pr == -999 ? 19 : pr >= 2000 ? 20 : off_bucket(pr - x); }

/* Column (vertical) matched transfer.  pv column transition lists are built per map; for cur we track the last
 * transition row of every column.  Prediction for column x: match the last cur transition (row lt, pair) to the
 * nearest pv transition with the same pair (r1, |r1-lt|<=16) -> dv = lt - r1;  next pv transition below r1 -> r0;
 * predicted row r0 + dv;  vo = predicted - y.  If unmatched: dv = 0, r0 = first pv transition >= y-3. */
#define MAXCT 96
static int16_t pcr[W][MAXCT]; static uint8_t pcq[W][MAXCT]; static int pcn[W];
static int16_t clt[W]; static uint8_t clq[W];
static void col_trans_build(const uint8_t *pv) {
    for (int x = 0; x < W; x++) { pcn[x] = 0; clt[x] = -1; clq[x] = 0; }
    for (int y = 1; y < H; y++) for (int x = 0; x < W; x++) {
        int a = AT(pv, y - 1, x), b = AT(pv, y, x);
        if (a != b && pcn[x] < MAXCT) { pcr[x][pcn[x]] = (int16_t)y; pcq[x][pcn[x]] = (uint8_t)(a * 6 + b); pcn[x]++; }
    }
}
static int vo_bucket(int o) { return o < -4 ? 0 : o > 12 ? 18 : o + 5; }    /* 0..18, 19 none */
static void col_pred(int y, int x, int *vo, int *vq) {
    int n = pcn[x], r1i = -1;
    if (clt[x] >= 0) {
        int bd = 17;
        for (int t = 0; t < n; t++) if (pcq[x][t] == clq[x] && abs(pcr[x][t] - clt[x]) < bd) { bd = abs(pcr[x][t] - clt[x]); r1i = t; }
    }
    if (r1i >= 0) {
        int dv = clt[x] - pcr[x][r1i];
        if (r1i + 1 < n) { *vo = vo_bucket(pcr[x][r1i + 1] + dv - y); *vq = pcq[x][r1i + 1]; }
        else { *vo = 19; *vq = 36; }
    } else {
        int t = 0; while (t < n && pcr[x][t] < y - 3) t++;
        if (t < n) { *vo = vo_bucket(pcr[x][t] - y); *vq = pcq[x][t]; } else { *vo = 19; *vq = 36; }
    }
}

/* Road-plane reprojection of a map (same integer formula as PR #154 container/mc_context.py composite()):
 * only ground pixels (class <= 1) change: lane (class 1) is re-sampled from the radially re-projected position.
 * mc=1: the parameters (tau, v0, u0, dx, dy) are FITTED DECODER-SIDE on (map t-2 -> map t-1) by integer coordinate
 * descent (start = previous map's fit) and then applied to map t-1 (constant-velocity assumption).  Zero side info. */
static inline int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b) && ((a < 0) != (b < 0))) q--; return q; }
#define NMCP 7
static int NQ = 5, NQF = 5;   /* NQ: params refined/coded by mcs; NQF: params fitted decoder-side */   /* active reprojection params: 5, or 7 with the left/right split (mlr=1) */
static int mcp[NMCP] = {44, 348, 261, 0, 0, 0, 0};    /* tau, v0, u0, dx, dy, right-half tau offset, right-half dx offset */
static uint8_t *cmap;                       /* padded composite of pv */
static int MCSW = 0;   /* set only inside the encoder's mcs_refine; decoder-side fits always use plain counts */
static int gl_n = -1;   /* ground-pixel list valid flag; reset by mc_fit for every map */
static int gl_maxrow = H;   /* cost only over rows < gl_maxrow (online refit on the decoded part) */
static void composite(const uint8_t *src, uint8_t *dst, const int *q, int only_cost, const uint8_t *ref, int64_t *cost) {
    /* only_cost: iterate only over the ground pixels (class <= 1) of src (list gl_*), bit-identical to the full loop */
    static int32_t gl_v[H * W], gl_u[H * W]; static const uint8_t *gl_src = NULL;
    int tau = q[0], v0 = q[1], u0 = q[2], dx = q[3], dy = q[4]; int64_t c = 0;
    int tauR = tau + q[5], dxR = dx + q[6]; if (tauR < 0) tauR = 0;   /* right half (2u+1 >= 2*u0) */
    if (only_cost) {
        if (gl_src != src || gl_n < 0) {   /* (re)build the ground-pixel list of this src map */
            gl_n = 0; gl_src = src;
            for (int v = 0; v < H; v++) for (int u = 0; u < W; u++) if (AT(src, v, u) <= 1) { gl_v[gl_n] = v; gl_u[gl_n] = u; gl_n++; }
        }
        int64_t dd[2][H], aa[H]; int vpr[2][H];
        for (int v = 0; v < H; v++) {
            int64_t a = 2 * v + 1 - v0; aa[v] = a;
            for (int sd = 0; sd < 2; sd++) {
                int64_t d = 16000 + (int64_t)(sd ? tauR : tau) * (a > 0 ? a : 0);
                int vp = (int)(fdiv((int64_t)v0 * d + 16000 * a, 2 * d) - dy); if (vp < 0) vp = 0; if (vp > H - 1) vp = H - 1;
                dd[sd][v] = d; vpr[sd][v] = vp;
            }
        }
        for (int i = 0; i < gl_n; i++) {
            int v = gl_v[i], u = gl_u[i], pvv = AT(src, v, u);
            if (v >= gl_maxrow) break;   /* list is in raster order */
            int sd = 2 * u + 1 >= 2 * u0; int64_t d = dd[sd][v];
            int up = (int)(fdiv(2 * (int64_t)u0 * d + (int64_t)(2 * u + 1 - 2 * u0) * 16000, 2 * d) - (sd ? dxR : dx)); if (up < 0) up = 0; if (up > W - 1) up = W - 1;
            int lane = AT(src, vpr[sd][v], up) == 1 && aa[v] > 0;
            int o = lane ? 1 : (pvv == 1 ? 0 : pvv), rv = AT(ref, v, u);
            if (o != rv) c += (MCSW && (o == 1 || rv == 1)) ? MCSW : 1;   /* MCSW: encoder-only lane weighting (mcs refinement) */
        }
        if (cost) *cost = c;
        return;
    }
    for (int v = 0; v < H; v++) {
        int64_t a = 2 * v + 1 - v0, dL = 16000 + (int64_t)tau * (a > 0 ? a : 0), dR = 16000 + (int64_t)tauR * (a > 0 ? a : 0);
        int vpL = (int)(fdiv((int64_t)v0 * dL + 16000 * a, 2 * dL) - dy); if (vpL < 0) vpL = 0; if (vpL > H - 1) vpL = H - 1;
        int vpR = (int)(fdiv((int64_t)v0 * dR + 16000 * a, 2 * dR) - dy); if (vpR < 0) vpR = 0; if (vpR > H - 1) vpR = H - 1;
        for (int u = 0; u < W; u++) {
            int pvv = AT(src, v, u), o = pvv;
            if (pvv <= 1) {
                int sd = 2 * u + 1 >= 2 * u0; int64_t d = sd ? dR : dL; int vp = sd ? vpR : vpL;
                int up = (int)(fdiv(2 * (int64_t)u0 * d + (int64_t)(2 * u + 1 - 2 * u0) * 16000, 2 * d) - (sd ? dxR : dx)); if (up < 0) up = 0; if (up > W - 1) up = W - 1;
                int lane = AT(src, vp, up) == 1 && a > 0;
                o = lane ? 1 : (pvv == 1 ? 0 : pvv);
            }
            dst[(v + B) * PW + u + B] = (uint8_t)o;
        }
    }
    if (cost) *cost = c;
}
static void mc_fit(const uint8_t *p2, const uint8_t *pv) {
    gl_n = -1; gl_maxrow = H;
    static const int steps[NMCP][4] = {{8, 4, 2, 1}, {4, 2, 1, 0}, {4, 2, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {4, 2, 1, 0}, {1, 0, 0, 0}};
    static const int lo[NMCP] = {0, 300, 200, -3, -2, -24, -3}, hi[NMCP] = {80, 400, 320, 3, 2, 24, 3};
    int64_t best; composite(p2, NULL, mcp, 1, pv, &best);
    for (int round = 0; round < 2; round++)
        for (int k = 0; k < NQF; k++) for (int si = 0; si < 4 && steps[k][si]; si++) {
            for (int sg = -1; sg <= 1; sg += 2) {
                int q[NMCP]; memcpy(q, mcp, sizeof q); q[k] += sg * steps[k][si];
                if (q[k] < lo[k] || q[k] > hi[k]) continue;
                int64_t c; composite(p2, NULL, q, 1, pv, &c);
                if (c < best) { best = c; memcpy(mcp, q, sizeof q); }
            }
        }
}
/* mc2=1: full ground-plane radial warp of t-1 below the horizon (every class, not only lane pixels), same params */
static uint8_t *cmap2, *cmap3, *cmapv[16], *cmapw[2];
static void warp_full(const uint8_t *src, uint8_t *dst, const int *q) {
    int tau = q[0], v0 = q[1], u0 = q[2], dx = q[3], dy = q[4];
    for (int v = 0; v < H; v++) {
        int64_t a = 2 * v + 1 - v0, d = 16000 + (int64_t)tau * (a > 0 ? a : 0);
        int vp = (int)(fdiv((int64_t)v0 * d + 16000 * a, 2 * d) - dy); if (vp < 0) vp = 0; if (vp > H - 1) vp = H - 1;
        for (int u = 0; u < W; u++) {
            int o = AT(src, v, u);
            if (a > 0) { int up = (int)(fdiv(2 * (int64_t)u0 * d + (int64_t)(2 * u + 1 - 2 * u0) * 16000, 2 * d) - dx); if (up < 0) up = 0; if (up > W - 1) up = W - 1; o = AT(src, vp, up); }
            dst[(v + B) * PW + u + B] = (uint8_t)o;
        }
    }
}

/* Lane-dash periodicity (ld=1): walk up the radial line through the vanishing point (fitted u0, v0) from the
 * current pixel over already decoded rows of the current map; measure, in ground-plane depth units Z = 2^20 / a
 * (a = 2r+1-v0, i.e. twice the row distance below the horizon), the length of the current run (distance since the
 * last class change), the previous run and the run before it (same class as the current one = last full period part).
 * Returns packed bucket features; only computed near lane pixels. */
static inline int ld_b(int64_t d) { if (d <= 0) return 0; int b = (int)(lg2q16((uint32_t)(d > 0xffffff ? 0xffffff : d)) >> 15); return b > 47 ? 47 : b; }
static void lane_dash(const uint8_t *cur, int y, int x, int *f) {
    int v0 = mcp[1], u0 = mcp[2]; int64_t ay = 2 * y + 1 - v0;
    f[0] = f[1] = f[2] = f[3] = f[4] = 0; f[5] = 7;
    if (ay < 8) return;
    int64_t zy = ((int64_t)1 << 20) / ay;
    int runs = 0, cls = -1, prevcls = -1, curc = -1; int64_t zstart = zy, len[3] = {0, 0, 0};
    for (int r = y - 1; r >= 0 && r >= y - 128; r--) {
        int64_t ar = 2 * r + 1 - v0; if (ar < 4) break;
        int u = (int)(u0 + fdiv((int64_t)(2 * x + 1 - 2 * u0) * ar, 2 * ay)); if (u < 0 || u >= W) break;
        int c = AT(cur, r, u); int64_t zr = ((int64_t)1 << 20) / ar;
        if (curc < 0) { curc = cls = c; continue; }
        if (c != curc) {                 /* run boundary between r+1 and r */
            len[runs] = zr - zstart; zstart = zr; curc = c;
            if (runs == 0) prevcls = c;
            if (++runs == 3) break;
        }
    }
    if (cls < 0) return;
    f[0] = 1 + (cls == 1) + 2 * (prevcls == 1);   /* 1..4 */
    f[1] = runs; f[2] = ld_b(len[0]); f[3] = ld_b(len[1]); f[4] = ld_b(len[2]);
    f[5] = runs >= 3 && len[2] > 0 ? (int)(len[0] * 6 / len[2]) : 7; if (f[5] > 7) f[5] = 7;   /* phase vs last same-class run */
}

/* ld=2: dash-edge transfer along the lane ray.  cur's nearest boundary above (depth distance Dc) is matched to the
 * nearest boundary of the reprojected map cmap along the same ray (Dm): err = Dc - Dm.  cmap's next boundary at or
 * below the current row (Db <= 0) shifted by err predicts cur's next dash edge; returned as a signed row offset. */
static int lane_edge_pred(const uint8_t *cur, const uint8_t *cm, int y, int x) {
    int v0 = mcp[1], u0 = mcp[2]; int64_t ay = 2 * y + 1 - v0;
    if (ay < 8) return 20;
    int64_t zy = ((int64_t)1 << 20) / ay;
    int64_t Dc = -1, bm[8]; int nbm = 0, cc = -1, mc = -1;
    for (int r = y - 1; r >= 0 && r >= y - 96; r--) {
        int64_t ar = 2 * r + 1 - v0; if (ar < 4) break;
        int u = (int)(u0 + fdiv((int64_t)(2 * x + 1 - 2 * u0) * ar, 2 * ay)); if (u < 0 || u >= W) break;
        int c = AT(cur, r, u), m = AT(cm, r, u); int64_t D = ((int64_t)1 << 20) / ar - zy;
        if (cc >= 0 && c != cc && Dc < 0) Dc = D;
        if (mc >= 0 && m != mc && nbm < 8) bm[nbm++] = D;
        cc = c; mc = m;
        if (Dc >= 0 && nbm >= 2) break;
    }
    if (Dc < 0 || !nbm) return 21;
    int64_t Dm = bm[0]; for (int i = 1; i < nbm; i++) if (llabs(bm[i] - Dc) < llabs(Dm - Dc)) Dm = bm[i];
    int64_t err = Dc - Dm, Db = 1; int pm = AT(cm, y - 1, x);
    for (int r = y; r < H && r <= y + 32; r++) {
        int64_t ar = 2 * r + 1 - v0;
        int u = (int)(u0 + fdiv((int64_t)(2 * x + 1 - 2 * u0) * ar, 2 * ay)); if (u < 0 || u >= W) break;
        int m = AT(cm, r, u);
        if (m != pm) { Db = ((int64_t)1 << 20) / ar - zy; break; }
        pm = m;
    }
    if (Db > 0) return 22;   /* no cmap boundary below within 32 rows */
    int64_t rows = -(Db + err) * ay * ay / ((int64_t)1 << 21);   /* predicted edge, rows below the current one */
    return rows < -8 ? 0 : rows > 8 ? 18 : (int)rows + 9;
}

/* ld=4: dash period transfer. Along the ray, the reprojected map gives the spacing P (depth units) between
 * consecutive dash starts (boundary not-1 above / 1 below) nearest the current pixel; the last dash start above in
 * the current map, minus P, predicts the next dash start; returned as a signed row offset bucket. */
static int lane_period(const uint8_t *cur, const uint8_t *cm, int y, int x) {
    int v0 = mcp[1], u0 = mcp[2]; int64_t ay = 2 * y + 1 - v0;
    if (ay < 8) return 20;
    int64_t zy = ((int64_t)1 << 20) / ay, st[16]; int ns = 0, pm = -1;
    for (int r = y - 128 < 0 ? 0 : y - 128; r < H && r <= y + 64; r++) {      /* top -> bottom along the ray */
        int64_t ar = 2 * r + 1 - v0; if (ar < 4) continue;
        int u = (int)(u0 + fdiv((int64_t)(2 * x + 1 - 2 * u0) * ar, 2 * ay)); if (u < 0 || u >= W) { pm = -1; continue; }
        int m = AT(cm, r, u);
        if (pm >= 0 && pm != 1 && m == 1 && ns < 16) st[ns++] = ((int64_t)1 << 20) / ar - zy;   /* D > 0 above, < 0 below */
        pm = m;
    }
    if (ns < 2) return 21;
    int64_t P = -1, best = INT64_MAX;
    for (int i = 0; i + 1 < ns; i++) { int64_t mid = (st[i] + st[i + 1]) / 2; if (llabs(mid) < best) { best = llabs(mid); P = st[i] - st[i + 1]; } }
    int64_t Ds = -1; int pc = -1;
    for (int r = y - 1; r >= 0 && r >= y - 128; r--) {                     /* last dash start above in cur */
        int64_t ar = 2 * r + 1 - v0; if (ar < 4) break;
        int u = (int)(u0 + fdiv((int64_t)(2 * x + 1 - 2 * u0) * ar, 2 * ay)); if (u < 0 || u >= W) break;
        int c = AT(cur, r, u);
        if (pc == 1 && c != 1) { Ds = ((int64_t)1 << 20) / (ar + 2) - zy; break; }   /* start is between r and r+1 */
        pc = c;
    }
    if (Ds < 0 || P <= 0) return 22;
    int64_t rows = -(Ds - P) * ay * ay / ((int64_t)1 << 21);
    return rows < -8 ? 0 : rows > 8 ? 18 : (int)rows + 9;
}

/* obj=1: per-object motion for a class (default class 3).  Decoder-side: 4-connected components of that class in
 * t-1 and t-2; each t-1 component (>= 8 px) is matched to the t-2 component with the nearest centroid (<= 24 px,
 * size ratio within [1/2, 2]); its displacement (constant velocity) moves it in an object map omap built from the
 * reprojected map: old pixels get the most frequent bordering class, shifted pixels get the object class. */
#define MAXCOMP 4096
typedef struct { int n; int64_t sx, sy; int fill[6]; } Comp;
static int32_t lab1[H * W], lab2[H * W];
static int ccl(const uint8_t *m, int cls, int32_t *lab, Comp *cp) {
    static int32_t stk[H * W];
    int nc = 0;
    for (int i = 0; i < H * W; i++) lab[i] = -1;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        if (AT(m, y, x) != cls || lab[y * W + x] >= 0 || nc >= MAXCOMP) continue;
        Comp *c = &cp[nc]; memset(c, 0, sizeof *c);
        int sp = 0; stk[sp++] = y * W + x; lab[y * W + x] = nc;
        while (sp) {
            int i = stk[--sp], yy = i / W, xx = i % W;
            c->n++; c->sx += xx; c->sy += yy;
            const int dy4[4] = {-1, 1, 0, 0}, dx4[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int ny = yy + dy4[d], nx = xx + dx4[d];
                if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                int v = AT(m, ny, nx), j = ny * W + nx;
                if (v == cls) { if (lab[j] < 0) { lab[j] = nc; stk[sp++] = j; } }
                else c->fill[v]++;
            }
        }
        nc++;
    }
    return nc;
}
static uint8_t *omap, *pdc;
static int obj_moved, OBJM = 0;   /* number of moved components in this map (diagnostic) */
static void obj_build(const uint8_t *pv, const uint8_t *p2, const uint8_t *base, int cls) {
    static Comp c1[MAXCOMP], c2[MAXCOMP];
    memcpy(omap, base, PW * PH); obj_moved = 0;
    int n1 = ccl(pv, cls, lab1, c1), n2 = ccl(p2, cls, lab2, c2);
    static int dxs[MAXCOMP], dys[MAXCOMP];
    if (OBJM) {   /* objm=1: displacement by template matching: shift d maximising #pixels of the t-1 component whose
                   * position - d is the object class in t-2 (ties: smaller |dx|+|dy|) */
        static int32_t first[MAXCOMP], nxt[H * W];
        for (int a = 0; a < n1; a++) first[a] = -1;
        for (int i = H * W - 1; i >= 0; i--) if (lab1[i] >= 0) { nxt[i] = first[lab1[i]]; first[lab1[i]] = i; }
        for (int a = 0; a < n1; a++) {
            dxs[a] = dys[a] = 0;
            if (c1[a].n < 8) continue;
            int bs = -1, bp = 1 << 30;
            for (int ddy = -8; ddy <= 8; ddy++) for (int ddx = -8; ddx <= 8; ddx++) {
                int sc = 0;
                for (int i = first[a]; i >= 0; i = nxt[i]) { int yy = i / W - ddy, xx = i % W - ddx; if (yy >= 0 && yy < H && xx >= 0 && xx < W && AT(p2, yy, xx) == cls) sc++; }
                int pen = abs(ddx) + abs(ddy);
                if (sc > bs || (sc == bs && pen < bp)) { bs = sc; bp = pen; dxs[a] = ddx; dys[a] = ddy; }
            }
            if (bs * 2 < c1[a].n) dxs[a] = dys[a] = 0;   /* poor match: do not move */
        }
    } else
    for (int a = 0; a < n1; a++) {
        dxs[a] = dys[a] = 0;
        if (c1[a].n < 8) continue;
        int best = -1; int64_t bd = (int64_t)24 * 24 + 1;
        for (int b = 0; b < n2; b++) {
            if (c2[b].n * 2 < c1[a].n || c2[b].n > c1[a].n * 2) continue;
            int64_t ddx = fdiv(c1[a].sx * c2[b].n - c2[b].sx * c1[a].n, (int64_t)c1[a].n * c2[b].n);
            int64_t ddy = fdiv(c1[a].sy * c2[b].n - c2[b].sy * c1[a].n, (int64_t)c1[a].n * c2[b].n);
            int64_t d2 = ddx * ddx + ddy * ddy;
            if (d2 < bd) { bd = d2; best = b; }
        }
        if (best < 0) continue;
        int64_t den = (int64_t)c1[a].n * c2[best].n;
        int ddx = (int)fdiv(2 * (c1[a].sx * c2[best].n - c2[best].sx * c1[a].n) + den, 2 * den);
        int ddy = (int)fdiv(2 * (c1[a].sy * c2[best].n - c2[best].sy * c1[a].n) + den, 2 * den);
        if (ddx < -16) ddx = -16; if (ddx > 16) ddx = 16; if (ddy < -16) ddy = -16; if (ddy > 16) ddy = 16;
        dxs[a] = ddx; dys[a] = ddy;
    }
    for (int a = 0; a < n1; a++) {   /* erase moved objects: fill class = most frequent bordering class */
        if (!dxs[a] && !dys[a]) continue;
        int f = 0; for (int v = 1; v < 6; v++) if (c1[a].fill[v] > c1[a].fill[f]) f = v;
        if (f == OOB) f = 0;
        c1[a].fill[0] = f;   /* reuse slot 0 as the chosen fill class */
    }
    for (int i = 0; i < H * W; i++) { int a = lab1[i]; if (a >= 0 && (dxs[a] || dys[a])) omap[(i / W + B) * PW + i % W + B] = (uint8_t)c1[a].fill[0]; }
    for (int i = 0; i < H * W; i++) {
        int a = lab1[i]; if (a < 0 || (!dxs[a] && !dys[a])) continue;
        int ny = i / W + dys[a], nx = i % W + dxs[a];
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        omap[(ny + B) * PW + nx + B] = (uint8_t)cls; obj_moved++;
    }
}

/* mcs=1: per-map side info = correction of the decoder-side fit. The ENCODER refines the params against the true
 * current map (coordinate descent, cost = mismatches on ground pixels of t-1) and codes delta = best - fit for each
 * of the 5 params in the main arithmetic stream (zero flag, sign, unary magnitude; small adaptive binary models).
 * Counted in T automatically. */
static const int mcs_lim[NMCP] = {15, 7, 7, 3, 2, 15, 3};
static uint32_t mcs_p[NMCP][2 + 16];
static void mcs_reset(void) { for (int j = 0; j < NMCP; j++) for (int i = 0; i < 18; i++) mcs_p[j][i] = 1u << 31; }
static int mcs_bit(Coder *c, uint32_t *p, int bit) {
    uint32_t q = *p < 4096 ? 4096 : (*p > 0xffffffffu - 4096 ? 0xffffffffu - 4096 : *p);
    bit = code_bit(c, bit, q);
    if (bit) *p += (~*p) >> 4; else *p -= *p >> 4;
    return bit;
}
static int mcs_code(Coder *c, int j, int d) {      /* encoder passes d; decoder gets it back */
    int nz = mcs_bit(c, &mcs_p[j][0], d != 0);
    if (!nz) return 0;
    int neg = mcs_bit(c, &mcs_p[j][1], d < 0), mag = abs(d), r = 1;
    while (r < mcs_lim[j] && mcs_bit(c, &mcs_p[j][2 + (r < 15 ? r : 15)], mag > r)) r++;
    return neg ? -r : r;
}
static void mcs_refine(const uint8_t *pv, const uint8_t *curfull, int *q_out) {
    static const int steps[NMCP][4] = {{8, 4, 2, 1}, {4, 2, 1, 0}, {4, 2, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {4, 2, 1, 0}, {1, 0, 0, 0}};
    int q[NMCP]; memcpy(q, mcp, sizeof q);
    gl_n = -1; gl_maxrow = H; MCSW = prm[P_MCSW];
    int64_t best; composite(pv, NULL, q, 1, curfull, &best);
    for (int round = 0; round < 2 + prm[P_MCSR]; round++)
        for (int k = 0; k < NQ; k++) for (int si = 0; si < 4 && steps[k][si]; si++)
            for (int sg = -1; sg <= 1; sg += 2) {
                if (k == 5 && prm[P_MLR] == 3) continue;   /* mlr=3: right-half dx correction only */
                int t[NMCP]; memcpy(t, q, sizeof t); t[k] += sg * steps[k][si];
                if (abs(t[k] - mcp[k]) > mcs_lim[k]) continue;
                int64_t cc; composite(pv, NULL, t, 1, curfull, &cc);
                if (cc < best) { best = cc; memcpy(q, t, sizeof q); }
            }
    gl_n = -1;
    MCSW = 0;   /* back to plain counts before any decoder-side fit runs */
    memcpy(q_out, q, sizeof q);
}

/* mcr: online refit of the reprojection params on the already decoded rows (< y) of the current map, src = t-1 */
static void mc_refit(const uint8_t *pv, const uint8_t *cur, int y) {
    static const int steps[5][3] = {{4, 2, 1}, {2, 1, 0}, {2, 1, 0}, {1, 0, 0}, {1, 0, 0}};
    static const int lo[5] = {0, 300, 200, -3, -2}, hi[5] = {80, 400, 320, 3, 2};   /* (mcr: unused experiment, left half only) */
    gl_n = -1; gl_maxrow = y;
    int64_t best; composite(pv, NULL, mcp, 1, cur, &best);
    for (int k = 0; k < 5; k++) for (int si = 0; si < 3 && steps[k][si]; si++)
        for (int sg = -1; sg <= 1; sg += 2) {
            int q[NMCP]; memcpy(q, mcp, sizeof q); q[k] += sg * steps[k][si];
            if (q[k] < lo[k] || q[k] > hi[k]) continue;
            int64_t c; composite(pv, NULL, q, 1, cur, &c);
            if (c < best) { best = c; memcpy(mcp, q, sizeof q); }
        }
    gl_maxrow = H; gl_n = -1;
    composite(pv, cmap, mcp, 0, NULL, NULL);
}

/* decoder-side global shift per map: accumulate mismatch counts of every shift over decoded rows */
#define GMAX 4
static int64_t gcnt[2 * GMAX + 1][2 * GMAX + 1];
static int gdx = 0, gdy = 0, gprev_dx = 0, gprev_dy = 0;
static void gshift_row(const uint8_t *cur, const uint8_t *pv, int y, int gs) {   /* add row y-1 */
    for (int dy = -gs; dy <= gs; dy++) for (int dx = -gs; dx <= gs; dx++) {
        int c = 0;
        const uint8_t *a = cur + (y - 1 + B) * PW + B, *b = pv + (y - 1 + dy + B) * PW + B + dx;
        for (int x = 0; x < W; x++) c += a[x] != b[x];
        gcnt[dy + GMAX][dx + GMAX] += c;
    }
    int64_t bc = INT64_MAX; int bp = 1 << 30;
    for (int dy = -gs; dy <= gs; dy++) for (int dx = -gs; dx <= gs; dx++) {
        int64_t c = gcnt[dy + GMAX][dx + GMAX];
        int pen = (dx == gprev_dx && dy == gprev_dy) ? 0 : 1 + abs(dx) + abs(dy);
        if (c < bc || (c == bc && pen < bp)) { bc = c; bp = pen; gdx = dx; gdy = dy; }
    }
}

/* ------------------------------------------------------------------ model */
#define MAXCTX 96
#define SMN 32
#define SMQ 65
typedef struct {
    HashTab ht[MAXCTX];
    int nctx, na, act[MAXCTX];
    Mixer m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, mf, mf2, mf3, mf4, mf5, mf6, mf7, mf8, mf9, mfx, mfy;
    int kx12, kx5, kx8, kmc1, kx7;
    APM a1, a2, a3, a4, a5, a6, a7, a8, a9, a10;
    uint32_t *smp; uint16_t *smc;   /* indirect state maps: (ctx, node, min(n,SMN-1), quantised p) -> p */
} Model;

/* context ids (bit k of mask_off disables context k); stage at which each is introduced */
enum {
    C_S0, C_S1, C_S2, C_S3, C_S4, C_S5, C_S6,                 /* spatial */
    C_T0, C_T1,                                               /* co-located prev (S0) */
    C_P3, C_P3S, C_P5, C_P3S4, C_PROW, C_PCOL, C_PD, C_U0, C_U1, /* prev non-causal + t-2 (S1) */
    C_G0, C_G1, C_L0, C_L1, C_LE, C_ET, C_L5, C_AG,           /* warps (S2) */
    C_X1, C_X2, C_X3, C_X4, C_V1, C_V2, C_X5, C_X6, C_X7, C_X8, C_X9, C_X10, C_X11, C_X12, C_X13, C_X14, C_X15, C_X16, C_R1, C_R2, C_MC1, C_MC2, C_MC3, C_MC4, C_MC5, C_MC6, C_MC7, C_MC8, C_MC9, C_MC10, C_MC11, C_MC12, C_MC13, C_MC14, C_MC15, C_MC16, C_LD1, C_LD2, C_LD3, C_LD4, C_OB1, C_OB2, C_BD1, C_BD2, C_MC17, C_XC1, C_XC2, C_XC3, C_XC4, C_XC5, C_MC18,                      /* matched edge transfer / extrapolation (S2) */
    NCTXDEF
};
static const int ctx_stage[NCTXDEF] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
static int ctx_on[NCTXDEF];

#define NSET1 (4 * 2 * 5 * 3)
#define NSET2 (4 * 5 * 2 * 2)
#define NSET3 (4 * 10 * 3)
#define NSETF (4 * 7 * 2 * 6 * 5)
#define NSET4 (4 * 20 * 2 * 2)
#define NSET5 (4 * 20 * 8 * 2)
#define NSET6 (4 * 48 * 16)
#define NSET7 (4 * 20 * 20)
#define NSET8 (4 * 20 * 8 * 8)
#define NSET9 (4 * 16 * 10 * 2)
#define NSET10 (4 * 1024)
#define NSET11 (4 * 1296)
#define NSET12 (4 * 343)
#define NSET13 (4 * 7 * 7 * 7)
#define NSET14 (4 * 16384)
#define NSET15 (4 * 4096)
#define NSET16 (4 * 4096)
#define NSET17 (4 * 4096)
#define NSET18 (4 * 4096)
#define NSET19 (4 * 4096)
#define NSET20 (4 * 4096)
#define NSET21 (4 * 4096)
#define NSET22 (4 * 4096)
#define NSET23 (4 * 4096)
#define NSET24 (4 * 4096)
#define NSET25 (4 * 80)
#define NSET26 (4 * 4096)
#define NSET27 (4 * 4096)
#define NSET28 (4 * 360)
#define NSET29 (4 * 120)
#define NSET30 (4 * 4096)
#define NAPM1 (4 * 65536)
#define NSETF2 (4 * 1024)
#define NSETF3 (4 * 4096)
#define NSETF4 (4 * 4096)
#define NSETF5 (4 * 4096)
#define NAPM2 (4 * 8000)

/* hbs=1: per-context hash-table bits, ~4x the bucket occupancy measured on the full 600 maps (snap_T99527 config) */
static const uint8_t hbs_tab[96] = {12,12,13,15,15,16,14,20,20,15,20,20,20,16,15,20,20,16,16,20,16,20,19,20,20,20,20,15,18,16,20,20,19,20,17,21,21,21,19,19,19,19,21,17,16,20,16,17,13,19,19,16,17,17,16,18,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20};
/* hbs=2 / hbs=3: per-context table bits = ceil(log2(16x / 8x the bucket occupancy measured on the full 600 maps with
 * the snap_T98385 config at hb=21)), clamped to [12, 21]; contexts without a measurement get 20 */
static const uint8_t hbs_tab16[96] = {12,13,15,17,17,18,16,20,20,17,20,20,20,18,17,20,20,18,18,20,18,20,21,20,20,20,20,17,20,18,20,20,21,20,19,21,21,21,21,21,21,21,21,19,18,20,18,19,15,21,21,18,19,19,18,20,17,17,15,17,20,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20};
static const uint8_t hbs_tab32[96] = {12,14,16,18,18,19,17,21,21,18,21,21,21,19,18,21,21,19,19,21,19,21,22,21,21,21,21,18,21,19,21,21,22,21,20,22,22,22,22,22,22,22,22,20,19,21,19,20,16,22,22,19,20,20,19,21,18,18,16,18,21,20,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21};   /* hbs=4: 32x occupancy, clamp [12, 22] */
static const uint8_t hbs_tab16b[96] = {12,13,15,17,17,18,16,20,20,17,20,20,20,18,17,20,20,18,18,20,18,20,21,20,20,20,20,17,20,18,20,20,21,20,19,22,22,22,21,21,21,21,22,19,18,20,18,19,15,21,21,18,19,19,18,20,17,17,15,17,20,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20};   /* hbs=5: 16x occupancy, clamp [12, 22] */
static const uint8_t hbs_tab8[96] = {12,12,14,16,16,17,15,20,20,16,20,20,20,17,16,20,20,17,17,20,17,20,20,20,20,20,20,16,19,17,20,20,20,20,18,21,21,21,20,20,20,20,21,18,17,20,17,18,14,20,20,17,18,18,17,19,16,16,14,16,19,18,19,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20};
static void model_init(Model *m) {
    MLRD = prm[P_MLRD]; APMN = prm[P_APMN]; MIXQ = prm[P_MIXQ]; PINIT = prm[P_PINIT]; PINIT_R = prm[P_PINIT] ? prm[P_PINIT] : 10;
    m->nctx = 0; m->na = 0;
    for (int k = 0; k < NCTXDEF; k++) {
        ctx_on[k] = ctx_stage[k] <= prm[P_STAGE] && !(k < 64 && (mask_off >> k & 1)) && (k < C_MC1 || prm[P_MC]) && (k < C_MC4 || k > C_MC5 || prm[P_MCC]) && (k < C_MC6 || k > C_MC7 || prm[P_MC2]) && (k < C_MC8 || k > C_MC9 || prm[P_MC3]) && (k < C_MC10 || k > C_MC10 || prm[P_MCL]) && (k < C_MC11 || prm[P_MCV]) && (k < C_MC13 || prm[P_MCV] >= 2) && (k < C_MC15 || k > C_MC15 || prm[P_MCV] >= 3) && (k < C_MC16 || k > C_MC16 || (prm[P_MCW] && prm[P_MC2])) && (k < C_LD1 || prm[P_LD]) && (k < C_LD3 || prm[P_LD] >= 2) && (k < C_LD4 || k > C_LD4 || prm[P_LD] >= 4) && (k < C_OB1 || k > C_OB2 || prm[P_OBJ]) && (k < C_BD1 || (prm[P_BD] && prm[P_MC])) && (k < C_BD2 || k > C_BD2 || prm[P_BD] >= 2) && (k < C_MC17 || k > C_MC17 || prm[P_MCV] >= 4) && (k < C_XC1 || k > C_XC3 || (prm[P_XC] && prm[P_MC] && (k < C_XC3 || prm[P_LD]))) && (k < C_XC4 || k > C_XC5 || (prm[P_XC] >= 2 && prm[P_MC])) && (k < C_MC18 || prm[P_MCV] >= 5);
        if (ctx_on[k]) {
            m->ht[k].bits = prm[P_HBS] == 5 ? hbs_tab16b[k] : prm[P_HBS] == 4 ? hbs_tab32[k] : prm[P_HBS] == 3 ? hbs_tab8[k] : prm[P_HBS] == 2 ? hbs_tab16[k] : prm[P_HBS] ? hbs_tab[k] : prm[P_HB];
            m->ht[k].t = calloc((size_t)1 << m->ht[k].bits, sizeof(Bucket));   /* (was 2^hb: only the first 2^bits were ever touched) */
            for (int i = 0; i < 4; i++) m->ht[k].prior[i] = 1u << 31;
            m->nctx++;
            m->act[m->na++] = k;
        }
    }
    int nin = m->na + 1;     /* active contexts + bias */
    int32_t w0 = (int32_t)(65536 * prm[P_W0] / 100 / (m->nctx ? m->nctx : 1) * 4);
    mixer_init(&m->m1, nin, NSET1, prm[P_LR], w0);
    mixer_init(&m->m2, nin, NSET2, prm[P_LR], w0);
    mixer_init(&m->m3, nin, NSET3, prm[P_LR], w0);
    mixer_init(&m->m4, nin, NSET4, prm[P_LR], w0);
    mixer_init(&m->m5, nin, NSET5, prm[P_LR], w0);
    mixer_init(&m->m6, nin, NSET6, prm[P_LR], w0);
    mixer_init(&m->m7, nin, NSET7, prm[P_LR], w0);
    mixer_init(&m->m8, nin, NSET8, prm[P_LR], w0);
    mixer_init(&m->m9, nin, NSET9, prm[P_LR], w0);
    mixer_init(&m->m10, nin, NSET10, prm[P_LR], w0);
    mixer_init(&m->m11, nin, NSET11, prm[P_LR], w0);
    mixer_init(&m->m12, nin, NSET12, prm[P_LR], w0);
    mixer_init(&m->m13, nin, NSET13, prm[P_LR], w0);
    m->kx12 = m->kx5 = m->kx8 = -1;
    m->kmc1 = m->kx7 = -1;
    for (int k = 0; k < m->na; k++) { if (m->act[k] == C_MC1) m->kmc1 = k; if (m->act[k] == C_X7) m->kx7 = k; }
    for (int k = 0; k < m->na; k++) { if (m->act[k] == C_X12) m->kx12 = k; if (m->act[k] == C_X5) m->kx5 = k; if (m->act[k] == C_X8) m->kx8 = k; }
    mixer_init(&m->m14, nin, NSET14, prm[P_LR], w0);
    mixer_init(&m->m15, nin, NSET15, prm[P_LR], w0);
    mixer_init(&m->m16, nin, NSET16, prm[P_LR], w0);
    mixer_init(&m->m17, nin, NSET17, prm[P_LR], w0);
    mixer_init(&m->m18, nin, NSET18, prm[P_LR], w0);
    mixer_init(&m->m19, nin, NSET19, prm[P_LR], w0);
    mixer_init(&m->m20, nin, NSET20, prm[P_LR], w0);
    mixer_init(&m->m21, nin, NSET21, prm[P_LR], w0);
    mixer_init(&m->m22, nin, NSET22, prm[P_LR], w0);
    mixer_init(&m->m23, nin, NSET23, prm[P_LR], w0);
    mixer_init(&m->m24, nin, NSET24, prm[P_LR], w0);
    mixer_init(&m->m25, nin, NSET25, prm[P_LR], w0);
    mixer_init(&m->m26, nin, NSET26, prm[P_LR], w0);
    mixer_init(&m->m27, nin, NSET27, prm[P_LR], w0);
    mixer_init(&m->m28, nin, NSET28, prm[P_LR], w0);
    mixer_init(&m->m29, nin, NSET29, prm[P_LR], w0);
    mixer_init(&m->m30, nin, NSET30, prm[P_LR], w0);
    mixer_init(&m->mf, 31, NSETF, prm[P_LR2], 65536 / (4 + (prm[P_M5] != 0) + (prm[P_M6] != 0) + (prm[P_M7] != 0) + (prm[P_M8] != 0) + (prm[P_M9] != 0) + (prm[P_M10] != 0) + (prm[P_M11] != 0) + (prm[P_M12] != 0) + (prm[P_M13] != 0) + (prm[P_M14] != 0) + (prm[P_M15] != 0) + (prm[P_M16] != 0) + (prm[P_M17] != 0) + (prm[P_M18] != 0) + (prm[P_M19] != 0) + (prm[P_M20] != 0) + (prm[P_M21] != 0) + (prm[P_M22] != 0) + (prm[P_M23] != 0) + (prm[P_M24] != 0) + (prm[P_MC] == 2) + (prm[P_M26] != 0) + (prm[P_M27] != 0) + (prm[P_M28] != 0) + (prm[P_LD] >= 3) + (prm[P_M30] != 0)));
    mixer_init(&m->mf2, 31, NSETF2, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf3, 31, NSETF3, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf4, 31, NSETF4, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf5, 31, NSETF5, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf6, 31, NSETF5, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf7, 31, NSETF5, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf8, 31, NSETF5, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mf9, 31, NSETF5, prm[P_LR2], m->mf.w[0]);
    mixer_init(&m->mfx, 7, 4 * 6, prm[P_LR2], 65536 / 4);
    mixer_init(&m->mfy, 9, 4 * 6, prm[P_FMIXLR] ? prm[P_FMIXLR] : prm[P_LR2], 0);
    for (int st6 = 0; st6 < 4 * 6; st6++) m->mfy.w[st6 * 9] = 65536;   /* start as the plain average */
    apm_init(&m->a1, NAPM1, prm[P_APMR]);
    apm_init(&m->a2, NAPM2, prm[P_APMR]);
    apm_init(&m->a3, NAPM2, prm[P_APMR]);
    apm_init(&m->a4, NAPM2, prm[P_APMR]);
    apm_init(&m->a5, NAPM2, prm[P_APMR]);
    if (prm[P_APM6]) apm_init(&m->a6, 4 * 8192, prm[P_APMR]);
    if (prm[P_APM9]) apm_init(&m->a9, 4 * 1089, prm[P_APMR]);
    if (prm[P_APM10]) apm_init(&m->a10, 4 * 1089, prm[P_APMR]);
    if (prm[P_APM7]) { apm_init(&m->a7, 4 * 8192, prm[P_APMR]); apm_init(&m->a8, 4 * 8192, prm[P_APMR]); }
    if (prm[P_SM] || prm[P_DUAL]) { fprintf(stderr, "sm/dual experiments were removed (both hurt)\n"); exit(2); }
}

#define PK2(a, b) (((uint64_t)(a) << 3) | (uint64_t)(b))
#define ID(k) ((uint64_t)(k) << 58)

typedef struct {
    uint64_t ctx[NCTXDEF];
    int c1, c2, others[3];
    int sel1, sel2, sel3, self, apm1, apm2;
} PixCtx;

#ifdef STATS
static double st_cost[8][4], st_ev[8]; static long st_cnt[8][4], st_nev[8];   /* [category][node] */
#endif

_Static_assert(NCTXDEF <= MAXCTX, "too many contexts");
static int code_pixel(Coder *c, Model *m, const uint8_t *cur, const uint8_t *pv, const uint8_t *p2,
                      const uint8_t *pd, int y, int x, int tok) {
    const int st = prm[P_STAGE];
    int Wc = AT(cur, y, x - 1), Nc = AT(cur, y - 1, x), NW = AT(cur, y - 1, x - 1), NE = AT(cur, y - 1, x + 1);
    int WW = AT(cur, y, x - 2), NN = AT(cur, y - 2, x), NNE = AT(cur, y - 2, x + 1), NEE = AT(cur, y - 1, x + 2);
    int NWW = AT(cur, y - 1, x - 2), NNW = AT(cur, y - 2, x - 1);
    if (y == 0) { Nc = NW = NE = NN = NNE = NEE = NWW = NNW = OOB; }
    else if (y == 1) { NN = NNE = NNW = OOB; }
    int PV = AT(pv, y, x);
    int e1 = y >= 1 ? edge_off(cur, y - 1, x) : NONE, e2 = y >= 2 ? edge_off(cur, y - 2, x) : NONE;
    int run = 0; while (run < 15 && x - 1 - run >= 0 && AT(cur, y, x - 1 - run) == Wc) run++;
    int vrun = 0; while (vrun < 15 && y - 1 - vrun >= 0 && AT(cur, y - 1 - vrun, x) == Nc) vrun++;
    PixCtx q;
    uint64_t *cx = q.ctx;
    cx[C_S0] = ID(C_S0) | PK2(Wc, Nc);
    cx[C_S1] = ID(C_S1) | (PK2(PK2(PK2(Wc, Nc), NW), NE));
    cx[C_S2] = ID(C_S2) | (PK2(PK2(PK2(PK2(PK2(Wc, Nc), NW), NE), WW), NN));
    cx[C_S3] = ID(C_S3) | (PK2(PK2(PK2(PK2(PK2(PK2(Wc, Nc), NE), NEE), NNE), NN), NWW) << 4 | (uint64_t)NNW);
    cx[C_S4] = ID(C_S4) | ((uint64_t)e1 << 20 | (uint64_t)e2 << 16 | PK2(PK2(PK2(Wc, Nc), NE), NEE));
    cx[C_S5] = ID(C_S5) | ((uint64_t)run << 24 | (uint64_t)e1 << 16 | (uint64_t)vrun << 8 | PK2(Wc, Nc));
    cx[C_S6] = ID(C_S6) | ((uint64_t)(y >> 4) << 20 | (uint64_t)(x >> 6) << 16 | PK2(PK2(Wc, Nc), NE));
    cx[C_T0] = ID(C_T0) | PK2(PK2(PV, Wc), Nc);
    cx[C_T1] = ID(C_T1) | PK2(PK2(PK2(PK2(PV, Wc), Nc), NW), NE);

    /* S1: non-causal previous-map templates */
    uint64_t pv3 = 0, pv5 = 0; int D = 4, typ = 0;
    if (st >= 1) {
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) pv3 = pv3 * 8 + AT(pv, y + dy, x + dx);
        if (ctx_on[C_P5] || prm[P_M19]) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) pv5 = (pv5 ^ (uint64_t)AT(pv, y + dy, x + dx)) * 0x100000001b3ULL;
        uint64_t prow = 0, pcol = 0;
        for (int dx = -3; dx <= 3; dx++) prow = prow * 8 + AT(pv, y, x + dx);
        for (int dy = -2; dy <= 3; dy++) pcol = pcol * 8 + AT(pv, y + dy, x);
        D = pd[y * W + x];
        int P2 = AT(p2, y, x);
        uint64_t p23 = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) p23 = p23 * 8 + AT(p2, y + dy, x + dx);
        cx[C_P3] = ID(C_P3) | pv3;
        cx[C_P3S] = ID(C_P3S) | pv3 << 6 | PK2(Wc, Nc);
        cx[C_P5] = ID(C_P5) | (pv5 & 0x3ffffffffffffffULL);
        cx[C_P3S4] = ID(C_P3S4) | pv3 << 12 | PK2(PK2(PK2(Wc, Nc), NW), NE);
        cx[C_PROW] = ID(C_PROW) | prow << 12 | (uint64_t)AT(pv, y + 1, x) << 9 | (uint64_t)AT(pv, y + 2, x) << 6 | PK2(Wc, Nc);
        cx[C_PCOL] = ID(C_PCOL) | pcol << 9 | PK2(PK2(Wc, Nc), NE);
        cx[C_PD] = ID(C_PD) | (uint64_t)D << 12 | PK2(PK2(PK2(PV, Wc), Nc), NE);
        cx[C_U0] = ID(C_U0) | PK2(PK2(PK2(P2, PV), Wc), Nc);
        cx[C_U1] = ID(C_U1) | p23 << 6 | PK2(PV, Wc);
        int a = AT(pv, y - 1, x - 1);
        typ = 1;
        for (int dy = -1; dy <= 1 && typ; dy++) for (int dx = -1; dx <= 1; dx++) if (AT(pv, y + dy, x + dx) != a) { typ = 0; break; }
    }
    /* S2: warp candidates (decoder-side, no side info) */
    uint64_t cm3_g = 0; int ld_sel2 = 0, apm5c_g = 0, ld_sel = 0, apm4c_g = 0, mc_sel4 = 0, mc_sel3 = 0, mc_sel2 = 0, mc_sel = 0, otc = 0, PG = PV, PL = PV, mflag = 0, pe0 = NONE, ot = 19, qt = 36, os = 19, qs = 36; uint64_t pl3_g = 0;
    if (st >= 2) {
        PG = AT(pv, y + gdy, x + gdx);
        uint64_t pg3 = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) pg3 = pg3 * 8 + AT(pv, y + gdy + dy, x + gdx + dx);
        int ldx = (y || prm[P_MM]) ? mc_dx[x] : 0, ldy = (y || prm[P_MM]) ? mc_dy[x] : 0;
        mflag = (ldx != 0) * 2 + (ldy != 0);
        PL = AT(pv, y + ldy, x + ldx);
        uint64_t pl3 = 0, pl5 = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) pl3 = pl3 * 8 + AT(pv, y + ldy + dy, x + ldx + dx);
        pl3_g = pl3;
        if (ctx_on[C_L5]) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) pl5 = (pl5 ^ (uint64_t)AT(pv, y + ldy + dy, x + ldx + dx)) * 0x100000001b3ULL;
        pe0 = edge_off(pv, y + ldy, x + ldx); int pe1 = edge_off(pv, y + ldy + 1, x + ldx), pem = edge_off(pv, y + ldy - 1, x + ldx);
        int et = NONE;
        if (e1 != NONE && pe0 != NONE && pem != NONE) { et = e1 + pe0 - pem; if (et < 0) et = 0; if (et > 8) et = 8; }
        cx[C_G0] = ID(C_G0) | pg3 << 6 | PK2(Wc, Nc);
        cx[C_G1] = ID(C_G1) | (uint64_t)((gdx + GMAX) * 9 + gdy + GMAX) << 20 | PK2(PK2(PK2(PK2(PG, Wc), Nc), NW), NE);
        cx[C_L0] = ID(C_L0) | pl3 << 6 | PK2(Wc, Nc);
        cx[C_L1] = ID(C_L1) | (uint64_t)mflag << 24 | PK2(PK2(PK2(PK2(PK2(PL, Wc), Nc), NW), NE), NN);
        cx[C_LE] = ID(C_LE) | (uint64_t)pe0 << 24 | (uint64_t)pe1 << 20 | (uint64_t)pem << 16 | (uint64_t)e1 << 12 | PK2(PK2(PK2(PL, Wc), Nc), NE);
        cx[C_ET] = ID(C_ET) | (uint64_t)et << 16 | (uint64_t)e1 << 12 | PK2(PK2(PK2(PL, Wc), Nc), NE);
        cx[C_L5] = ID(C_L5) | (pl5 & 0x3ffffffffffffffULL);
        ot = y ? xo_t[x] : 19; qt = y ? xp_t[x] : 36; os = y ? xo_s[x] : 19; qs = y ? xp_s[x] : 36;
        cx[C_X1] = ID(C_X1) | (uint64_t)ot << 24 | (uint64_t)qt << 12 | PK2(PK2(Wc, Nc), NE);
        cx[C_X2] = ID(C_X2) | (uint64_t)os << 24 | (uint64_t)qs << 12 | PK2(PK2(Wc, Nc), NE);
        cx[C_X3] = ID(C_X3) | (uint64_t)ot << 24 | (uint64_t)os << 16 | (uint64_t)e1 << 8 | PK2(Wc, Nc);
        cx[C_X4] = ID(C_X4) | (uint64_t)(y >> 5) << 28 | (uint64_t)(x >> 6) << 24 | (uint64_t)ot << 16 | (uint64_t)qt << 6 | (uint64_t)Wc;
        int om = y ? xo_v[0][x] : 19, qm = y ? xp_v[0][x] : 36, op = y ? xo_v[1][x] : 19, qp = y ? xp_v[1][x] : 36;
        cx[C_X5] = ID(C_X5) | (uint64_t)om << 40 | (uint64_t)qm << 32 | (uint64_t)op << 24 | (uint64_t)qp << 16 | PK2(PK2(Wc, Nc), NE);
        cx[C_X6] = ID(C_X6) | (uint64_t)om << 32 | (uint64_t)op << 24 | (uint64_t)ot << 16 | (uint64_t)(qt / 6 == Wc) << 8 | PK2(Wc, Nc);
        cx[C_X7] = ID(C_X7) | (uint64_t)(y >> 3) << 32 | (uint64_t)(x >> 5) << 24 | (uint64_t)e1 << 16 | PK2(PK2(Wc, Nc), NE);
        cx[C_X8] = ID(C_X8) | (uint64_t)ot << 40 | (uint64_t)om << 32 | (uint64_t)op << 24 | (uint64_t)os << 16 | (uint64_t)e1 << 8 | PK2(Wc, Nc);
        otc = ot == 19 ? 0 : ot >= 17 ? 1 : abs(ot - 8) <= 1 ? 2 + (ot - 7) : 5 + (ot > 8);
        cx[C_X9] = ID(C_X9) | (uint64_t)y << 32 | (uint64_t)x << 16 | PK2(Wc, Nc);
        cx[C_X10] = ID(C_X10) | (uint64_t)(y >> 1) << 32 | (uint64_t)(x >> 1) << 16 | (uint64_t)otc << 8 | PK2(PK2(Wc, Nc), NE);
        cx[C_X11] = ID(C_X11) | (uint64_t)(y >> 2) << 32 | (uint64_t)(x >> 2) << 16 | (uint64_t)e1 << 8 | PK2(Wc, Nc);
        int xa = y ? xa_s[x] : 0xfff;
        cx[C_X12] = ID(C_X12) | (uint64_t)os << 32 | (uint64_t)xa << 16 | (uint64_t)(qs / 6 == Wc) << 8 | PK2(Wc, Nc);
        cx[C_X13] = ID(C_X13) | (uint64_t)ot << 32 | (uint64_t)(xa >> 9) << 24 | (uint64_t)(xa & 7) << 20 | (uint64_t)os << 12 | PK2(Wc, Nc);
        int xt = y ? xa_t[x] : 0x1ff;
        cx[C_X14] = ID(C_X14) | (uint64_t)ot << 32 | (uint64_t)xt << 16 | (uint64_t)(qt / 6 == Wc) << 8 | PK2(Wc, Nc);
        cx[C_X15] = ID(C_X15) | (uint64_t)ot << 40 | (uint64_t)os << 32 | (uint64_t)(xt >> 3) << 24 | (uint64_t)(xa & 63) << 12 | PK2(Wc, Nc);
        int ou = y ? xo_u[x] : 19, qu = y ? xp_u[x] : 36;
        cx[C_X16] = ID(C_X16) | (uint64_t)ou << 32 | (uint64_t)ot << 24 | (uint64_t)(qu / 6 == Wc) << 16 | (uint64_t)(qt / 6 == Wc) << 12 | PK2(Wc, Nc);
        int rb0 = ir_bucket(ir_pred0, x), rbb = ir_bucket(ir_predb, x);
        cx[C_R1] = ID(C_R1) | (uint64_t)rb0 << 32 | (uint64_t)ir_q0 << 20 | (uint64_t)rbb << 12 | PK2(Wc, Nc);
        cx[C_R2] = ID(C_R2) | (uint64_t)rbb << 32 | (uint64_t)ir_dvb << 28 | (uint64_t)ot << 20 | (uint64_t)os << 12 | (uint64_t)(ir_qb / 6 == Wc) << 8 | PK2(Wc, Nc);
        if (prm[P_MC]) {
            int CM = AT(cmap, y, x), oc = y ? xo_c[x] : 19, qc = y ? xp_c[x] : 36; uint64_t cm3 = 0;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) cm3 = cm3 * 8 + AT(cmap, y + dy, x + dx);
            cm3_g = cm3;
            uint64_t cm5 = 0;
            if (ctx_on[C_MC4]) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) cm5 = (cm5 ^ (uint64_t)AT(cmap, y + dy, x + dx)) * 0x100000001b3ULL;
            cx[C_MC4] = ID(C_MC4) | (cm5 & 0x3ffffffffffffffULL);
            if (prm[P_MC2]) {
                uint64_t c23 = 0; int oc2 = y ? xo_c2[x] : 19, qc2 = y ? xp_c2[x] : 36;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) c23 = c23 * 8 + AT(cmap2, y + dy, x + dx);
                cx[C_MC6] = ID(C_MC6) | c23 << 6 | PK2(Wc, Nc);
                if (prm[P_MCW]) {
                    int w0v = AT(cmapw[0], y, x), w1v = AT(cmapw[1], y, x), C2v = AT(cmap2, y, x);
                    cx[C_MC16] = ID(C_MC16) | PK2(PK2(PK2(PK2(PK2(PK2(PK2(w0v, C2v), w1v), AT(cmapw[0], y, x + 1)), AT(cmapw[1], y, x + 1)), Wc), Nc), NE) | (uint64_t)oc2 << 32;
                }
                cx[C_MC7] = ID(C_MC7) | (uint64_t)oc2 << 32 | (uint64_t)qc2 << 20 | (uint64_t)oc << 12 | (uint64_t)(qc2 / 6 == Wc) << 8 | PK2(Wc, Nc);
            }
            if (prm[P_MC3]) {
                uint64_t c33 = 0; int C3 = AT(cmap3, y, x);
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) c33 = c33 * 8 + AT(cmap3, y + dy, x + dx);
                cx[C_MC8] = ID(C_MC8) | c33 << 12 | (uint64_t)CM << 6 | PK2(Wc, Nc);
                cx[C_MC9] = ID(C_MC9) | PK2(PK2(PK2(PK2(PK2(C3, CM), PV), Wc), Nc), NE) | (uint64_t)oc << 24;
            }
            {   /* lane-radial context: row distance below the fitted horizon, column offset from the vanishing point */
                int rv = 2 * y + 1 - mcp[1]; rv = rv < 0 ? 0 : rv >> 4; if (rv > 31) rv = 31;
                int ru = (x - mcp[2]) >> 5; ru = ru < -8 ? -8 : ru > 7 ? 7 : ru;
                if (prm[P_M27]) mc_sel3 = (int)((hash64((uint64_t)rv << 32 | (uint64_t)(ru + 8) << 24 | (uint64_t)CM << 16 | (uint64_t)(oc) << 8 | PK2(Wc, Nc)) >> 52) & 4095);
                cx[C_MC10] = ID(C_MC10) | (uint64_t)rv << 40 | (uint64_t)(ru + 8) << 32 | (cm3 & 0x7ffffff) << 4 | (uint64_t)Wc;
            }
            if (prm[P_MCV]) {
                int a0 = AT(cmapv[0], y, x), a1 = AT(cmapv[1], y, x), b0 = AT(cmapv[2], y, x), b1 = AT(cmapv[3], y, x);
                int a0e = AT(cmapv[0], y, x + 1), a1e = AT(cmapv[1], y, x + 1), a0s = AT(cmapv[0], y + 1, x), a1s = AT(cmapv[1], y + 1, x);
                cx[C_MC11] = ID(C_MC11) | PK2(PK2(PK2(PK2(PK2(PK2(PK2(a0, CM), a1), a0e), a1e), Wc), Nc), NE) | (uint64_t)PK2(a0s, a1s) << 30;
                cx[C_MC12] = ID(C_MC12) | PK2(PK2(PK2(PK2(PK2(b0, CM), b1), Wc), Nc), NE) | (uint64_t)oc << 24;
                if (prm[P_MCV] >= 2) {
                    int c0 = AT(cmapv[4], y, x), c1v = AT(cmapv[5], y, x), d0 = AT(cmapv[6], y, x), d1 = AT(cmapv[7], y, x);
                    cx[C_MC13] = ID(C_MC13) | PK2(PK2(PK2(PK2(PK2(PK2(PK2(c0, c1v), d0), d1), CM), Wc), Nc), NE);
                    int agree = (a0 == CM) + (a1 == CM) + (b0 == CM) + (b1 == CM) + (c0 == CM) + (c1v == CM) + (d0 == CM) + (d1 == CM);
                    mc_sel4 = (agree * 20 + oc) * 2 + (CM == Wc);
                    apm4c_g = ((agree * 20 + oc) * 6 + CM) * 6 + (Wc == OOB ? 5 : Wc);   /* < 9*20*36 = 6480 */
                    if (prm[P_MCV] >= 5) {
                        int h0 = AT(cmapv[12], y, x), h1 = AT(cmapv[13], y, x), k0 = AT(cmapv[14], y, x), k1 = AT(cmapv[15], y, x);
                        cx[C_MC18] = ID(C_MC18) | PK2(PK2(PK2(PK2(PK2(PK2(PK2(h0, h1), k0), k1), CM), Wc), Nc), NE) | (uint64_t)oc << 32;
                    }
                    if (prm[P_MCV] >= 4) {
                        int e0 = AT(cmapv[8], y, x), e1v = AT(cmapv[9], y, x), g0 = AT(cmapv[10], y, x), g1 = AT(cmapv[11], y, x);
                        cx[C_MC17] = ID(C_MC17) | PK2(PK2(PK2(PK2(PK2(PK2(PK2(e0, e1v), g0), g1), CM), Wc), Nc), NE) | (uint64_t)(a0 == CM) << 30 | (uint64_t)(a1 == CM) << 31;
                    }
                    if (prm[P_MCV] >= 3) {
                        uint64_t h0 = 0, h1 = 0;
                        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { h0 = h0 * 8 + AT(cmapv[0], y + dy, x + dx); h1 = h1 * 8 + AT(cmapv[1], y + dy, x + dx); }
                        cx[C_MC15] = ID(C_MC15) | ((h0 << 27 | h1) & 0x3fffffffffffffULL) << 3 | (uint64_t)Wc;
                    }
                    cx[C_MC14] = ID(C_MC14) | (uint64_t)agree << 20 | (uint64_t)oc << 12 | PK2(PK2(CM, Wc), Nc);
                }
            }
            if (prm[P_LD]) {
                int f[6] = {0, 0, 0, 0, 0, 7};
                int near = Wc == 1 || Nc == 1 || NE == 1 || NW == 1 || CM == 1 || PV == 1 || AT(cmap, y + 1, x) == 1 || AT(cmap, y, x + 1) == 1;
                if (near) lane_dash(cur, y, x, f);
                if (prm[P_M30]) ld_sel2 = (int)((hash64((uint64_t)f[0] << 40 | (uint64_t)f[1] << 36 | (uint64_t)f[2] << 28 | (uint64_t)f[3] << 20 | (uint64_t)f[5] << 12 | PK2(Wc, CM)) >> 52) & 4095);
                if (prm[P_XC]) cx[C_XC3] = ID(C_XC3) | (uint64_t)f[0] << 40 | (uint64_t)f[2] << 32 | (uint64_t)f[3] << 24 | (uint64_t)(y >> 4) << 12 | (uint64_t)CM << 4 | (uint64_t)Wc;
                cx[C_LD1] = ID(C_LD1) | (uint64_t)f[0] << 40 | (uint64_t)f[1] << 36 | (uint64_t)f[2] << 28 | (uint64_t)f[3] << 20 | (uint64_t)f[4] << 12 | PK2(Wc, Nc);
                if (prm[P_LD] >= 2) {
                    int le = near ? lane_edge_pred(cur, cmap, y, x) : 23;
                    cx[C_LD3] = ID(C_LD3) | (uint64_t)le << 24 | (uint64_t)f[0] << 16 | (uint64_t)CM << 8 | PK2(Wc, Nc);
                    ld_sel = le * 5 + f[0];
                    apm5c_g = ((le * 5 + f[0]) * 8 + f[5]) * 6 + (CM == OOB ? 5 : CM);   /* < 24*5*8*6 = 5760 */
                }
                if (prm[P_LD] >= 4) {
                    int lp = near ? lane_period(cur, cmap, y, x) : 23;
                    cx[C_LD4] = ID(C_LD4) | (uint64_t)lp << 24 | (uint64_t)f[0] << 16 | (uint64_t)CM << 8 | PK2(Wc, Nc);
                }
                cx[C_LD2] = ID(C_LD2) | (uint64_t)f[0] << 40 | (uint64_t)f[5] << 32 | (uint64_t)f[3] << 24 | (uint64_t)CM << 16 | (uint64_t)oc << 8 | PK2(Wc, Nc);
            }
            if (prm[P_XC]) {
                cx[C_XC1] = ID(C_XC1) | (uint64_t)os << 40 | (uint64_t)((y ? xa_s[x] : 0xfff) & 0x1ff) << 28 | (uint64_t)oc << 20 | (uint64_t)CM << 12 | PK2(Wc, Nc);
                cx[C_XC2] = ID(C_XC2) | (uint64_t)(y >> 3) << 40 | (uint64_t)(x >> 5) << 32 | (uint64_t)oc << 20 | (uint64_t)CM << 12 | PK2(Wc, Nc);
                if (prm[P_XC] >= 2) {   /* more cross contexts: transfer offsets triple; boundary distance x reprojected edge */
                    cx[C_XC4] = ID(C_XC4) | (uint64_t)ot << 40 | (uint64_t)os << 32 | (uint64_t)oc << 24 | (uint64_t)(qt / 6 == Wc) << 20 | (uint64_t)(qc / 6 == Wc) << 16 | PK2(Wc, Nc);
                    cx[C_XC5] = ID(C_XC5) | (uint64_t)D << 40 | (uint64_t)oc << 32 | (uint64_t)e1 << 24 | (uint64_t)CM << 16 | (uint64_t)NE << 8 | PK2(Wc, Nc);
                }
            }
            cx[C_MC5] = ID(C_MC5) | (uint64_t)oc << 32 | (uint64_t)os << 24 | (uint64_t)((y ? xa_s[x] : 0xfff) & 63) << 12 | PK2(Wc, Nc);
            cx[C_MC1] = ID(C_MC1) | cm3 << 6 | PK2(Wc, Nc);
            cx[C_MC2] = ID(C_MC2) | (uint64_t)oc << 32 | (uint64_t)qc << 20 | (uint64_t)ot << 12 | (uint64_t)(qc / 6 == Wc) << 8 | PK2(Wc, Nc);
            cx[C_MC3] = ID(C_MC3) | (uint64_t)(CM != PV) << 24 | PK2(PK2(PK2(PK2(PK2(CM, PV), Wc), Nc), NW), NE);
            mc_sel = (oc * 2 + (CM == Wc)) * 2 + (CM != PV);
            if (prm[P_M26]) {
                int oc2 = prm[P_MC2] && y ? xo_c2[x] : 19, C3v = prm[P_MC3] ? AT(cmap3, y, x) : CM;
                mc_sel2 = (int)((hash64((uint64_t)oc << 32 | (uint64_t)oc2 << 24 | (uint64_t)(qc / 6 == Wc) << 20 | (uint64_t)(C3v == CM) << 16 | (uint64_t)(CM == Wc) << 12 | (uint64_t)(CM != PV) << 8 | PK2(Wc, Nc)) >> 52) & 4095);
            }
        }
        if (prm[P_BD] && prm[P_MC]) {
            int Dc = pdc[y * W + x], CMv = AT(cmap, y, x);
            int dir = (AT(cmap, y, x + 1) != CMv) | (AT(cmap, y, x - 1) != CMv) << 1 | (AT(cmap, y + 1, x) != CMv) << 2 | (AT(cmap, y - 1, x) != CMv) << 3;
            int dir2 = (AT(cmap, y, x + 2) != CMv) | (AT(cmap, y, x - 2) != CMv) << 1 | (AT(cmap, y + 2, x) != CMv) << 2 | (AT(cmap, y - 2, x) != CMv) << 3;
            cx[C_BD1] = ID(C_BD1) | (uint64_t)Dc << 32 | (uint64_t)dir << 24 | (uint64_t)dir2 << 16 | (uint64_t)CMv << 8 | PK2(Wc, Nc);
            if (prm[P_BD] >= 2) {   /* boundary normal orientation of the reprojected map in a 5x5 window (8 bins) */
                int gx = 0, gy = 0;
                for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) if (AT(cmap, y + dy, x + dx) != CMv) { gx += dx; gy += dy; }
                int ang = (gx == 0 && gy == 0) ? 8 : (gx >= 0 ? (gy >= 0 ? (gx >= gy ? 0 : 1) : (gx >= -gy ? 7 : 6)) : (gy >= 0 ? (-gx >= gy ? 3 : 2) : (-gx >= -gy ? 4 : 5)));
                cx[C_BD2] = ID(C_BD2) | (uint64_t)Dc << 32 | (uint64_t)ang << 24 | (uint64_t)(y ? xo_c[x] : 19) << 16 | (uint64_t)CMv << 8 | PK2(Wc, Nc);
            }
        }
        if (prm[P_OBJ]) {
            uint64_t o3 = 0; int OM = AT(omap, y, x), oo = y ? xo_o[x] : 19, qo = y ? xp_o[x] : 36;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) o3 = o3 * 8 + AT(omap, y + dy, x + dx);
            cx[C_OB1] = ID(C_OB1) | o3 << 6 | PK2(Wc, Nc);
            cx[C_OB2] = ID(C_OB2) | (uint64_t)oo << 32 | (uint64_t)qo << 20 | (uint64_t)ot << 12 | (uint64_t)(OM != PV) << 8 | PK2(Wc, Nc);
        }
        int vo = 19, vq = 36; if (ctx_on[C_V1] || ctx_on[C_V2]) col_pred(y, x, &vo, &vq);
        cx[C_V1] = ID(C_V1) | (uint64_t)vo << 24 | (uint64_t)vq << 12 | PK2(PK2(Wc, Nc), NE);
        cx[C_V2] = ID(C_V2) | (uint64_t)vo << 24 | (uint64_t)ot << 16 | (uint64_t)vrun << 8 | PK2(Wc, Nc);
        cx[C_AG] = ID(C_AG) | PK2(PK2(PK2(PK2(PK2(PV, PG), PL), Wc), Nc), AT(p2, y, x));
    }

    /* binarisation */
    int c1;
    switch (prm[P_BIN]) {
        case 1: c1 = Wc; break;
        case 2: c1 = PL; break;
        case 3: c1 = PG; break;
        default: c1 = PV; break;
    }
    if (c1 == OOB) c1 = Wc != OOB ? Wc : (Nc != OOB ? Nc : 0);
    int cand[8] = {Wc, Nc, NE, NW, PV, PL, WW, NN}, c2 = -1;
    if (prm[P_C2O] == 1) { cand[1] = PL; cand[2] = Nc; cand[3] = NE; cand[4] = NW; cand[5] = PV; }
    if (prm[P_C2M] && prm[P_MC]) { int CMv = AT(cmap, y, x); if (CMv != Wc && CMv != OOB) { cand[0] = CMv; cand[1] = Wc; } }
    for (int i = 0; i < 8; i++) if (cand[i] != OOB && cand[i] != c1) { c2 = cand[i]; break; }
    if (c2 < 0) c2 = c1 == 0 ? 1 : 0;
    int others[3], k = 0;
    for (int v = 0; v < 5; v++) if (v != c1 && v != c2) others[k++] = v;

    /* mixer selectors */
    int agree = (Wc == c1) + (Nc == c1) + (NW == c1) + (NE == c1);          /* 0..4 */
    int ag3 = (PV == c1) + (PL == c1) + (PG == c1); if (ag3 > 2) ag3 = 2;     /* 0..2 */
    int sel1 = 0, sel2 = 0, sel3 = 0, sel4 = 0, sel5 = 0, sel6 = 0, sel7 = 0, sel8 = 0, sel9 = 0, sel10 = 0, sel11 = 0, sel12 = 0, sel14 = 0, sel15 = 0, sel16 = 0, sel17 = 0, sel18 = 0, sel19 = 0, sel20 = 0, sel21 = 0, sel22 = 0, sel23 = 0, sel24 = 0, self = 0;
    if (st >= 3) {
        sel1 = (typ * 5 + agree) * 3 + ag3;
        sel2 = (D * 2 + (Wc == Nc)) * 2 + (Wc == c1);
        sel3 = e1 * 3 + (mflag ? 1 + (mflag & 1) : 0);
        self = prm[P_FSEL] == 2 ? ((otc * 2 + (Wc == Nc)) * 6 + typ * 3 + ag3) * 5 + D : prm[P_FSEL] == 1 ? (otc * 2 + (Wc == Nc)) * 6 + typ * 3 + ag3 : typ * 3 + ag3;
        sel4 = (ot * 2 + (qt / 6 == Wc)) * 2 + (os == 8);
        { int xa8 = y ? xa_s[x] : 0xfff; sel8 = (os * 8 + (xa8 & 7)) * 8 + ((xa8 >> 3) & 7); }
        sel9 = (vrun * 10 + e1) * 2 + (Wc == Nc);
        sel10 = (int)((hash64(prm[P_MCH] ? cm3_g : pv3) >> 54) & 1023);
        sel11 = ((Wc * 6 + Nc) * 6 + NW) * 6 + NE;
        sel14 = !prm[P_M14] ? 0 : (int)((hash64(PK2(PK2(PK2(PK2(PK2(PK2(PK2(Wc, Nc), NW), NE), WW), NN), NNE), NEE)) >> 50) & (prm[P_M14] == 2 ? 16383 : 4095));
        { int WWW = AT(cur, y, x - 3), NNN = y >= 3 ? AT(cur, y - 3, x) : OOB;
          sel18 = !prm[P_M18] ? 0 : (int)((hash64(PK2(PK2(PK2(PK2(PK2(PK2(PK2(PK2(PK2(PK2(PK2(Wc, Nc), NW), NE), WW), NN), NNE), NEE), NWW), NNW), WWW), NNN) * 3 + 1) >> 52) & 4095); }
        sel21 = !prm[P_M21] ? 0 : (int)((hash64(pl3_g << 8 | PK2(Wc, Nc)) >> 52) & 4095);
        sel22 = !prm[P_M22] ? 0 : (int)((hash64((uint64_t)ot << 40 | (uint64_t)qt << 32 | (uint64_t)os << 24 | (uint64_t)qs << 16 | (uint64_t)Wc << 8 | 7) >> 52) & 4095);
        sel23 = !prm[P_M23] ? 0 : (int)((hash64((uint64_t)D << 40 | (uint64_t)e1 << 32 | (uint64_t)e2 << 24 | (uint64_t)vrun << 16 | (uint64_t)(run > 7 ? 7 : run) << 8 | PK2(Wc, Nc)) >> 52) & 4095);
        sel24 = !prm[P_M24] ? 0 : (int)((hash64((uint64_t)(y ? xo_v[0][x] : 19) << 40 | (uint64_t)(y ? xp_v[0][x] : 36) << 32 | (uint64_t)(y ? xo_v[1][x] : 19) << 24 | (uint64_t)(y ? xp_v[1][x] : 36) << 16 | PK2(Wc, Nc)) >> 52) & 4095);
        sel19 = !prm[P_M19] ? 0 : (int)((hash64(pv5 ^ 0x5bd1e995ULL) >> 52) & 4095);
        sel20 = !prm[P_M20] ? 0 : (int)((hash64((uint64_t)(y >> 3) << 24 | (uint64_t)(x >> 4) << 12 | PK2(Wc, Nc)) >> 52) & 4095);
        sel16 = !prm[P_M16] ? 0 : (int)((hash64(pv3 << 12 | PK2(PK2(PK2(Wc, Nc), NW), NE)) >> 52) & 4095);
        sel17 = !prm[P_M17] ? 0 : (int)((hash64((uint64_t)ot << 40 | (uint64_t)os << 32 | (uint64_t)(y ? xo_v[0][x] : 19) << 24 | (uint64_t)(y ? xo_v[1][x] : 19) << 16 | PK2(Wc, Nc)) >> 52) & 4095);
        sel15 = !prm[P_M15] ? 0 : (int)((hash64(((uint64_t)os << 32) | ((uint64_t)(y ? xa_s[x] : 0xfff) << 8) | (uint64_t)((qs / 6 == Wc) * 2 + (Wc == Nc))) >> 52) & 4095);
        { int a1 = y ? xo_v[0][x] : 19, a2 = y ? xo_v[1][x] : 19;
#define OB7(o) ((o) == 19 ? 0 : (o) >= 17 ? 1 : abs((o) - 8) <= 1 ? 2 + ((o) - 7) : 5 + ((o) > 8))
          sel12 = (OB7(ot) * 7 + OB7(a1)) * 7 + OB7(a2); }
        sel6 = prm[P_M6] == 2 ? (y >> 3) * 16 + (x >> 5) : (y >> 4) * 8 + (x >> 6);
        sel7 = (y ? xo_v[0][x] : 19) * 20 + (y ? xo_v[1][x] : 19);
        sel5 = (os * 8 + (y ? (xa_s[x] & 7) : 7)) * 2 + (qs / 6 == Wc);
    }
    int apm1c = prm[P_A1B] ? (int)((hash64((prm[P_MCH] ? cm3_g : pv3) << 12 | PK2(PK2(PK2(Wc, Nc), NW), NE)) >> 48) & 65535) : (int)((hash64(pv3 * 64 + Wc * 8 + Nc) >> 52) & 4095);
    int self3 = (int)((hash64((uint64_t)(y >> 4) << 20 | (uint64_t)(x >> 5) << 12 | (uint64_t)c1 << 4 | (uint64_t)(Wc == Nc)) >> 52) & 4095);
    int self5 = (int)((hash64((uint64_t)(y >> 4) << 40 | (uint64_t)(x >> 5) << 32 | (uint64_t)ot << 16 | (uint64_t)os << 8 | (uint64_t)Wc) >> 52) & 4095);
    int self8 = (int)((hash64((uint64_t)(prm[P_MC] && y ? xo_c[x] : 19) << 32 | (uint64_t)(prm[P_MC2] && prm[P_MC] && y ? xo_c2[x] : 19) << 24 | (uint64_t)ld_sel << 8 | (uint64_t)Wc) >> 52) & 4095);
    int self9 = (int)((hash64((uint64_t)(y >> 3) << 20 | (uint64_t)(x >> 3) << 4 | (uint64_t)(Wc == Nc)) >> 52) & 4095);
    int self7 = (int)((hash64((uint64_t)D << 32 | (uint64_t)e1 << 24 | (uint64_t)vrun << 16 | (uint64_t)(Wc == Nc) << 8 | (uint64_t)c1) >> 52) & 4095);
    int self6 = (int)((hash64(cm3_g << 12 | (uint64_t)c1 << 4 | (uint64_t)Wc) >> 52) & 4095);
    int self4 = (int)((hash64(pv3 << 8 | (uint64_t)c1 << 4 | (uint64_t)(Wc == c1) << 1 | (uint64_t)(Nc == c1)) >> 52) & 4095);
    int self2 = (int)((hash64((uint64_t)ot << 16 | PK2(PK2(Wc, Nc), NE) | (uint64_t)os << 24) >> 54) & 1023);
    int apm4c = apm4c_g, apm5c = apm5c_g;
    int apm7c = (int)((hash64(cm3_g << 8 | (uint64_t)Wc << 4 | (uint64_t)Nc) >> 51) & 8191);
    int apm8c = (int)((hash64((uint64_t)(y >> 3) << 24 | (uint64_t)(x >> 4) << 12 | (uint64_t)c1 << 4 | (uint64_t)(Wc == Nc) << 1 | (uint64_t)(Wc == c1)) >> 51) & 8191);
    int apm6c = (int)((hash64((uint64_t)os << 32 | (uint64_t)(y ? xa_s[x] : 0xfff) << 12 | (uint64_t)(Wc == Nc) << 8 | (uint64_t)c1 << 4 | (uint64_t)Wc) >> 51) & 8191);
    int apm3c = (((ot * 2 + (qt / 6 == Wc)) * 20 + os) * 2 + (Wc == Nc)) * 5 + (c1 < 5 ? c1 : 0);
    int apm2c = prm[P_APMC] == 2 ? ((y >> 3) * 32 + (x >> 4)) * 5 + (c1 < 5 ? c1 : 0) : prm[P_APMC] ? (((ot * 2 + (qt / 6 == Wc)) * 20 + os) * 2 + (Wc == Nc)) * 5 + (c1 < 5 ? c1 : 0) : ((c1 * 6 + (Wc == OOB ? 5 : Wc)) * 6 + (Nc == OOB ? 5 : Nc)) * 10 * 2 + e1 * 2 + (PL == c1);

    Bucket *g[NCTXDEF];                 /* compact list of the active contexts (disabled ones are skipped) */
    const int na = m->na;
    for (int k = 0; k < na; k++) g[k] = ht_lookup(&m->ht[m->act[k]], cx[m->act[k]]);

    int r = 0, want0 = 0;
    if (!c->dec) { want0 = tok == c1; if (!want0 && tok != c2) { for (r = 0; r < 3; r++) if (others[r] == tok) break; } }
    int in[MAXIN];
    int node = 0, res = -1;
    for (;;) {
        int a9_used = 0, a10_used = 0;
        for (int k = 0; k < na; k++) in[k] = g[k]->n[node] ? stretch(g[k]->p[node]) : 0;
        in[na] = 256;
        int s1 = mixer_dot(&m->m1, node * (NSET1 / 4) + sel1, in);
        uint32_t p1 = squash(s1), pf = p1, p2m = 0, p3m = 0, pfm = 0;
        int s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0, s9 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0, s14 = 0, s15 = 0, s16 = 0, s17 = 0, s18 = 0, s19 = 0, s20 = 0, s21 = 0, s22 = 0, s23 = 0, sf = 0, sf2 = 0, sf3 = 0, sf4 = 0, sf5 = 0, sf6 = 0, sf7 = 0, sf8 = 0, sf9 = 0, sfx = 0, fx[7], fx2[9], l2[31]; uint32_t pf2m = 0, pf3m = 0, pf4m = 0, pf5m = 0, pf6m = 0, pf7m = 0, pf8m = 0, pf9m = 0, pfxm = 0; uint32_t p4m = 0, p5m = 0, p6m = 0, p7m = 0, p8m = 0, p9m = 0, p10m = 0, p11m = 0, p12m = 0, p13m = 0, p14m = 0, p15m = 0, p16m = 0, p17m = 0, p18m = 0, p19m = 0, p20m = 0, p21m = 0, p22m = 0, p23m = 0, p24m = 0, p25m = 0, p26m = 0, p27m = 0, p28m = 0, p29m = 0, p30m = 0; int sel13 = 0;
        if (st >= 3) {
            s2 = mixer_dot(&m->m2, node * (NSET2 / 4) + sel2, in); p2m = squash(s2);
            s3 = mixer_dot(&m->m3, node * (NSET3 / 4) + sel3, in); p3m = squash(s3);
            s4 = mixer_dot(&m->m4, node * (NSET4 / 4) + sel4, in); p4m = squash(s4);
            if (prm[P_M5]) { s5 = mixer_dot(&m->m5, node * (NSET5 / 4) + sel5, in); p5m = squash(s5); }
            if (prm[P_M6]) { s6 = mixer_dot(&m->m6, node * (NSET6 / 4) + sel6, in); p6m = squash(s6); }
            if (prm[P_M7]) { s7 = mixer_dot(&m->m7, node * (NSET7 / 4) + sel7, in); p7m = squash(s7); }
            if (prm[P_M8]) { s8 = mixer_dot(&m->m8, node * (NSET8 / 4) + sel8, in); p8m = squash(s8); }
            if (prm[P_M9]) { s9 = mixer_dot(&m->m9, node * (NSET9 / 4) + sel9, in); p9m = squash(s9); }
            if (prm[P_M10]) { s10 = mixer_dot(&m->m10, node * (NSET10 / 4) + sel10, in); p10m = squash(s10); }
            if (prm[P_M11]) { s11 = mixer_dot(&m->m11, node * (NSET11 / 4) + sel11, in); p11m = squash(s11); }
            if (prm[P_M12]) { s12 = mixer_dot(&m->m12, node * (NSET12 / 4) + sel12, in); p12m = squash(s12); }
            if (prm[P_M13]) {   /* confidence-selected: hit-count buckets of three high-order contexts */
#define NB7(k) ((k) < 0 ? 0 : g[k]->n[node] == 0 ? 0 : g[k]->n[node] < 2 ? 1 : g[k]->n[node] < 4 ? 2 : g[k]->n[node] < 16 ? 3 : g[k]->n[node] < 64 ? 4 : g[k]->n[node] < 256 ? 5 : 6)
                sel13 = (NB7(m->kx12) * 7 + NB7(m->kx5)) * 7 + NB7(m->kx8);
                s13 = mixer_dot(&m->m13, node * (NSET13 / 4) + sel13, in); p13m = squash(s13); }
            l2[13] = s13;
            if (prm[P_M14]) { s14 = mixer_dot(&m->m14, node * (NSET14 / 4) + sel14, in); p14m = squash(s14); }
            if (prm[P_M15]) { s15 = mixer_dot(&m->m15, node * (NSET15 / 4) + sel15, in); p15m = squash(s15); }
            if (prm[P_M16]) { s16 = mixer_dot(&m->m16, node * (NSET16 / 4) + sel16, in); p16m = squash(s16); }
            if (prm[P_M17]) { s17 = mixer_dot(&m->m17, node * (NSET17 / 4) + sel17, in); p17m = squash(s17); }
            if (prm[P_M18]) { s18 = mixer_dot(&m->m18, node * (NSET18 / 4) + sel18, in); p18m = squash(s18); }
            if (prm[P_M19]) { s19 = mixer_dot(&m->m19, node * (NSET19 / 4) + sel19, in); p19m = squash(s19); }
            if (prm[P_M20]) { s20 = mixer_dot(&m->m20, node * (NSET20 / 4) + sel20, in); p20m = squash(s20); }
            if (prm[P_M21]) { s21 = mixer_dot(&m->m21, node * (NSET21 / 4) + sel21, in); p21m = squash(s21); }
            if (prm[P_M22]) { s22 = mixer_dot(&m->m22, node * (NSET22 / 4) + sel22, in); p22m = squash(s22); }
            if (prm[P_M23]) { s23 = mixer_dot(&m->m23, node * (NSET23 / 4) + sel23, in); p23m = squash(s23); }
            l2[14] = s14; l2[15] = s15; l2[16] = s16; l2[17] = s17; l2[18] = s18; l2[19] = s19; l2[20] = s20; l2[21] = s21; l2[22] = s22; l2[23] = s23;
            { int s30 = 0; if (prm[P_M30]) { s30 = mixer_dot(&m->m30, node * (NSET30 / 4) + ld_sel2, in); p30m = squash(s30); } l2[30] = s30; }
            { int s29 = 0; if (prm[P_LD] >= 3) { s29 = mixer_dot(&m->m29, node * (NSET29 / 4) + ld_sel, in); p29m = squash(s29); } l2[29] = s29; }
            { int s28 = 0; if (prm[P_M28]) { s28 = mixer_dot(&m->m28, node * (NSET28 / 4) + mc_sel4, in); p28m = squash(s28); } l2[28] = s28; }
            { int s27 = 0; if (prm[P_M27]) { s27 = mixer_dot(&m->m27, node * (NSET27 / 4) + mc_sel3, in); p27m = squash(s27); } l2[27] = s27; }
            { int s26 = 0; if (prm[P_M26]) { s26 = mixer_dot(&m->m26, node * (NSET26 / 4) + mc_sel2, in); p26m = squash(s26); } l2[26] = s26; }
            { int s25 = 0; if (prm[P_MC] == 2) { s25 = mixer_dot(&m->m25, node * (NSET25 / 4) + mc_sel, in); p25m = squash(s25); } l2[25] = s25; }
            { int s24 = 0; if (prm[P_M24]) { s24 = mixer_dot(&m->m24, node * (NSET24 / 4) + sel24, in); p24m = squash(s24); } l2[24] = s24; }
            l2[0] = s1; l2[1] = s2; l2[2] = s3; l2[3] = s4; l2[4] = 256; l2[5] = s5; l2[6] = s6; l2[7] = s7; l2[8] = s8; l2[9] = s9; l2[10] = s10; l2[11] = s11; l2[12] = s12;
            sf = mixer_dot(&m->mf, node * (NSETF / 4) + self, l2); pfm = squash(sf); pf = pfm;
            if (prm[P_F2] && !prm[P_F3]) { sf2 = mixer_dot(&m->mf2, node * (NSETF2 / 4) + self2, l2); pf2m = squash(sf2); sf = (sf + sf2) / 2; pf = squash(sf); }
            if (prm[P_F3]) { sf2 = mixer_dot(&m->mf2, node * (NSETF2 / 4) + self2, l2); pf2m = squash(sf2);
                             sf3 = mixer_dot(&m->mf3, node * (NSETF3 / 4) + self3, l2); pf3m = squash(sf3);
                             if (prm[P_F4]) { sf4 = mixer_dot(&m->mf4, node * (NSETF4 / 4) + self4, l2); pf4m = squash(sf4); }
                             int nf = 3 + (prm[P_F4] != 0), tot = sf + sf2 + sf3 + (prm[P_F4] ? sf4 : 0);
                             if (prm[P_F5]) { sf5 = mixer_dot(&m->mf5, node * (NSETF5 / 4) + self5, l2); pf5m = squash(sf5); tot += sf5; nf++; }
                             if (prm[P_F5] >= 2) { sf6 = mixer_dot(&m->mf6, node * (NSETF5 / 4) + self6, l2); pf6m = squash(sf6); tot += sf6; nf++; }
                             if (prm[P_F5] >= 3) { sf7 = mixer_dot(&m->mf7, node * (NSETF5 / 4) + self7, l2); pf7m = squash(sf7); tot += sf7; nf++; }
                             if (prm[P_F5] >= 4) { sf8 = mixer_dot(&m->mf8, node * (NSETF5 / 4) + self8, l2); pf8m = squash(sf8); tot += sf8; nf++; }
                             if (prm[P_F5] >= 5) { sf9 = mixer_dot(&m->mf9, node * (NSETF5 / 4) + self9, l2); pf9m = squash(sf9); tot += sf9; nf++; }
                             if (prm[P_FMIX] == 1) {   /* 3rd layer: mix the final mixers instead of averaging them */
                                 fx[0] = sf; fx[1] = sf2; fx[2] = sf3; fx[3] = sf4; fx[4] = sf5; fx[5] = sf6; fx[6] = 256;
                                 sfx = mixer_dot(&m->mfx, node * 6 + self % 6, fx); pfxm = squash(sfx); sf = sfx;
                             } else if (prm[P_FMIX] == 2) {   /* 3rd layer starting AS the average: input 0 = average (w=1), others learn deviations (w=0) */
                                 fx2[0] = tot / nf; fx2[1] = sf; fx2[2] = sf2; fx2[3] = sf3; fx2[4] = sf4; fx2[5] = sf5; fx2[6] = sf6; fx2[7] = sf7; fx2[8] = 256;
                                 sfx = mixer_dot(&m->mfy, node * 6 + self % 6, fx2); pfxm = squash(sfx); sf = sfx;
                             } else sf = tot / nf;
                             pf = squash(sf); }
        }
        int sfin = st >= 3 ? sf : s1;
        if (st >= 5 && prm[P_APMW]) {
            uint32_t q1 = apm_p(&m->a1, node * (prm[P_A1B] ? 65536 : 4096) + apm1c, sfin);
            if (prm[P_APMW] == 1) pf = (uint32_t)(((uint64_t)pf + q1) >> 1);
            else if (prm[P_APMW] == 2) {
                uint32_t q2 = apm_p(&m->a2, node * (NAPM2 / 4) + apm2c, sfin);
                pf = (uint32_t)(((uint64_t)pf * 2 + q1 + q2) >> 2);
            } else {   /* apmw=3: third APM on the edge-transfer features */
                uint32_t q2 = apm_p(&m->a2, node * (NAPM2 / 4) + apm2c, sfin), q3 = apm_p(&m->a3, node * (NAPM2 / 4) + apm3c, sfin);
                if (prm[P_APM4]) {   /* 4th APM keyed on the reprojection-variant agreement (mcv=2) */
                    uint32_t q4 = apm_p(&m->a4, node * (NAPM2 / 4) + apm4c, sfin);
                    if (prm[P_APM5]) {   /* 5th APM keyed on the lane-dash features (ld>=2) */
                        uint32_t q5 = apm_p(&m->a5, node * (NAPM2 / 4) + apm5c, sfin);
                        pf = (uint32_t)(((uint64_t)pf * 4 + (uint64_t)q1 * 4 + (uint64_t)q2 * 3 + (uint64_t)q3 * 2 + (uint64_t)q4 * 2 + (uint64_t)q5) >> 4);
                    } else
                    pf = (uint32_t)(((uint64_t)pf * 2 + (uint64_t)q1 * 2 + (uint64_t)q2 * 2 + q3 + q4) >> 3);
                } else pf = (uint32_t)(((uint64_t)pf + q1 + q2 + q3) >> 2);
            }
        }
        if (st >= 5 && prm[P_APM6]) {   /* APM chain: refine the final probability on (spatial chain-code edge, W==N, c1) */
            uint32_t pc = pf < 4096 ? 4096 : pf > 0xffffffffu - 4096 ? 0xffffffffu - 4096 : pf;
            uint32_t q6 = apm_p(&m->a6, node * 8192 + apm6c, stretch(pc));
            pf = prm[P_APM6] == 1 ? (uint32_t)(((uint64_t)pf * 3 + q6) >> 2) : prm[P_APM6] == 3 ? (uint32_t)(((uint64_t)pf * 7 + q6) >> 3) : (uint32_t)(((uint64_t)pf + q6) >> 1);
        }
        if (st >= 5 && prm[P_APM9] && m->kx12 >= 0 && m->kmc1 >= 0) {   /* 2-D SSE on the quantised predictions of two strong contexts */
            int q1v = (in[m->kx12] + 4096) >> 8, q2v = (in[m->kmc1] + 4096) >> 8;   /* 0..32 each */
            uint32_t pc = pf < 4096 ? 4096 : pf > 0xffffffffu - 4096 ? 0xffffffffu - 4096 : pf;
            uint32_t q9 = apm_p(&m->a9, node * 1089 + q1v * 33 + q2v, stretch(pc));
            pf = prm[P_APM9] == 1 ? (uint32_t)(((uint64_t)pf * 3 + q9) >> 2) : (uint32_t)(((uint64_t)pf * 7 + q9) >> 3);
            a9_used = 1;
        }
        if (st >= 5 && prm[P_APM10] && m->kx7 >= 0 && m->kmc1 >= 0) {   /* 2-D SSE on (position/edge context X7, reprojected 3x3 MC1) */
            int q1v = (in[m->kx7] + 4096) >> 8, q2v = (in[m->kmc1] + 4096) >> 8;
            uint32_t pc = pf < 4096 ? 4096 : pf > 0xffffffffu - 4096 ? 0xffffffffu - 4096 : pf;
            uint32_t q10 = apm_p(&m->a10, node * 1089 + q1v * 33 + q2v, stretch(pc));
            pf = (uint32_t)(((uint64_t)pf * 7 + q10) >> 3);
            a10_used = 1;
        }
        if (st >= 5 && prm[P_APM7]) {   /* further chain stages: reprojected 3x3 pattern; position */
            uint32_t pc = pf < 4096 ? 4096 : pf > 0xffffffffu - 4096 ? 0xffffffffu - 4096 : pf;
            uint32_t q7 = apm_p(&m->a7, node * 8192 + apm7c, stretch(pc));
            pf = (uint32_t)(((uint64_t)pf * 3 + q7) >> 2);
            if (prm[P_APM7] >= 2) {
                pc = pf < 4096 ? 4096 : pf > 0xffffffffu - 4096 ? 0xffffffffu - 4096 : pf;
                uint32_t q8 = apm_p(&m->a8, node * 8192 + apm8c, stretch(pc));
                pf = (uint32_t)(((uint64_t)pf * 3 + q8) >> 2);
            }
        }
        if (pf < 4096) pf = 4096;
        if (pf > 0xffffffffu - 4096) pf = 0xffffffffu - 4096;
        int want = 0;
        if (!c->dec) want = node == 0 ? want0 : node == 1 ? (tok == c2) : node == 2 ? (r == 0) : (r == 1);
        int bit = code_bit(c, want, pf);
#ifdef STATS
        { int cat = (Wc != Nc) * 4 + (ot == 19 || ot >= 17 ? 3 : abs(ot - 8) <= 1 ? 0 : abs(ot - 8) <= 3 ? 1 : 2); double pp = pf / 4294967296.0, cb = -log2(bit ? pp : 1 - pp);
          st_cost[cat][node] += cb; st_cnt[cat][node]++; if (node == 0 && !bit) { st_ev[cat] += cb; st_nev[cat]++; } }
#endif
        mixer_update(&m->m1, node * (NSET1 / 4) + sel1, in, bit, p1);
        if (st >= 3) {
            mixer_update(&m->m2, node * (NSET2 / 4) + sel2, in, bit, p2m);
            mixer_update(&m->m3, node * (NSET3 / 4) + sel3, in, bit, p3m);
            mixer_update(&m->m4, node * (NSET4 / 4) + sel4, in, bit, p4m);
            if (prm[P_M5]) mixer_update(&m->m5, node * (NSET5 / 4) + sel5, in, bit, p5m);
            if (prm[P_M6]) mixer_update(&m->m6, node * (NSET6 / 4) + sel6, in, bit, p6m);
            if (prm[P_M7]) mixer_update(&m->m7, node * (NSET7 / 4) + sel7, in, bit, p7m);
            if (prm[P_M8]) mixer_update(&m->m8, node * (NSET8 / 4) + sel8, in, bit, p8m);
            if (prm[P_M9]) mixer_update(&m->m9, node * (NSET9 / 4) + sel9, in, bit, p9m);
            if (prm[P_M10]) mixer_update(&m->m10, node * (NSET10 / 4) + sel10, in, bit, p10m);
            if (prm[P_M11]) mixer_update(&m->m11, node * (NSET11 / 4) + sel11, in, bit, p11m);
            if (prm[P_M12]) mixer_update(&m->m12, node * (NSET12 / 4) + sel12, in, bit, p12m);
            if (prm[P_M13]) mixer_update(&m->m13, node * (NSET13 / 4) + sel13, in, bit, p13m);
            if (prm[P_M14]) mixer_update(&m->m14, node * (NSET14 / 4) + sel14, in, bit, p14m);
            if (prm[P_M15]) mixer_update(&m->m15, node * (NSET15 / 4) + sel15, in, bit, p15m);
            if (prm[P_M16]) mixer_update(&m->m16, node * (NSET16 / 4) + sel16, in, bit, p16m);
            if (prm[P_M17]) mixer_update(&m->m17, node * (NSET17 / 4) + sel17, in, bit, p17m);
            if (prm[P_M18]) mixer_update(&m->m18, node * (NSET18 / 4) + sel18, in, bit, p18m);
            if (prm[P_M19]) mixer_update(&m->m19, node * (NSET19 / 4) + sel19, in, bit, p19m);
            if (prm[P_M20]) mixer_update(&m->m20, node * (NSET20 / 4) + sel20, in, bit, p20m);
            if (prm[P_M21]) mixer_update(&m->m21, node * (NSET21 / 4) + sel21, in, bit, p21m);
            if (prm[P_M22]) mixer_update(&m->m22, node * (NSET22 / 4) + sel22, in, bit, p22m);
            if (prm[P_M23]) mixer_update(&m->m23, node * (NSET23 / 4) + sel23, in, bit, p23m);
            if (prm[P_M24]) mixer_update(&m->m24, node * (NSET24 / 4) + sel24, in, bit, p24m);
            if (prm[P_MC] == 2) mixer_update(&m->m25, node * (NSET25 / 4) + mc_sel, in, bit, p25m);
            if (prm[P_M30]) mixer_update(&m->m30, node * (NSET30 / 4) + ld_sel2, in, bit, p30m);
            if (prm[P_LD] >= 3) mixer_update(&m->m29, node * (NSET29 / 4) + ld_sel, in, bit, p29m);
            if (prm[P_M28]) mixer_update(&m->m28, node * (NSET28 / 4) + mc_sel4, in, bit, p28m);
            if (prm[P_M27]) mixer_update(&m->m27, node * (NSET27 / 4) + mc_sel3, in, bit, p27m);
            if (prm[P_M26]) mixer_update(&m->m26, node * (NSET26 / 4) + mc_sel2, in, bit, p26m);
            mixer_update(&m->mf, node * (NSETF / 4) + self, l2, bit, pfm);
            if (prm[P_F2] || prm[P_F3]) mixer_update(&m->mf2, node * (NSETF2 / 4) + self2, l2, bit, pf2m);
            if (prm[P_F3]) mixer_update(&m->mf3, node * (NSETF3 / 4) + self3, l2, bit, pf3m);
            if (prm[P_F3] && prm[P_F4]) mixer_update(&m->mf4, node * (NSETF4 / 4) + self4, l2, bit, pf4m);
            if (prm[P_F3] && prm[P_F5]) mixer_update(&m->mf5, node * (NSETF5 / 4) + self5, l2, bit, pf5m);
            if (prm[P_F3] && prm[P_F5] >= 2) mixer_update(&m->mf6, node * (NSETF5 / 4) + self6, l2, bit, pf6m);
            if (prm[P_F3] && prm[P_F5] >= 3) mixer_update(&m->mf7, node * (NSETF5 / 4) + self7, l2, bit, pf7m);
            if (prm[P_F3] && prm[P_F5] >= 4) mixer_update(&m->mf8, node * (NSETF5 / 4) + self8, l2, bit, pf8m);
            if (prm[P_F3] && prm[P_F5] >= 5) mixer_update(&m->mf9, node * (NSETF5 / 4) + self9, l2, bit, pf9m);
            if (prm[P_F3] && prm[P_FMIX] == 1) mixer_update(&m->mfx, node * 6 + self % 6, fx, bit, pfxm);
            if (prm[P_F3] && prm[P_FMIX] == 2) mixer_update(&m->mfy, node * 6 + self % 6, fx2, bit, pfxm);
        }
        if (st >= 5 && prm[P_APMW]) { apm_update(&m->a1, bit); if (prm[P_APMW] != 1) apm_update(&m->a2, bit); if (prm[P_APMW] == 3) apm_update(&m->a3, bit); if (prm[P_APMW] == 3 && prm[P_APM4]) apm_update(&m->a4, bit); if (prm[P_APMW] == 3 && prm[P_APM4] && prm[P_APM5]) apm_update(&m->a5, bit); }
        if (st >= 5 && prm[P_APM6]) apm_update(&m->a6, bit);
        if (a9_used) apm_update(&m->a9, bit);
        if (a10_used) apm_update(&m->a10, bit);
        if (st >= 5 && prm[P_APM7]) { apm_update(&m->a7, bit); if (prm[P_APM7] >= 2) apm_update(&m->a8, bit); }
        for (int k = 0; k < na; k++) ctr_update(&g[k]->p[node], &g[k]->n[node], bit);
        if (PINIT) for (int k = 0; k < na; k++) {
            uint32_t *pr = &m->ht[m->act[k]].prior[node];
            if (bit) *pr += (~*pr) >> PINIT_R; else *pr -= *pr >> PINIT_R;
            if (*pr < 65536) *pr = 65536; if (*pr > 0xffffffffu - 65536) *pr = 0xffffffffu - 65536;
        }
        if (node == 0) { if (bit) { res = c1; break; } node = 1; }
        else if (node == 1) { if (bit) { res = c2; break; } node = 2; }
        else if (node == 2) { if (bit) { res = others[0]; break; } node = 3; }
        else { res = bit ? others[1] : others[2]; break; }
    }
    return res;
}

/* ------------------------------------------------------------------ file io / driver */
static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*len ? *len : 1); if (fread(b, 1, *len, f) != *len) { perror("read"); exit(1); }
    fclose(f); return b;
}
static void write_file(const char *path, const uint8_t *b, size_t len) {
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(1); }
    fwrite(b, 1, len, f); fclose(f);
}

static void run(Coder *c, uint8_t *maps, int nmaps) {
    Model m; model_init(&m);
    uint8_t *pad[3];                     /* rolling padded buffers: cur, prev, prev2 */
    for (int i = 0; i < 3; i++) { pad[i] = malloc(PW * PH); memset(pad[i], OOB, PW * PH); }
    uint8_t *pd = malloc(H * W); memset(pd, 4, H * W);
    gdx = gdy = gprev_dx = gprev_dy = 0;
    mcp[0] = 44; mcp[1] = 348; mcp[2] = 261; mcp[3] = 0; mcp[4] = 0; mcp[5] = 0; mcp[6] = 0; mcs_reset();
    NQ = prm[P_MLR] ? 7 : 5; NQF = prm[P_MLR] == 1 ? 7 : 5;   /* mlr=2,3: corrections only */
    /* mlr=2: right-half offsets only as encoder-coded corrections; reset to 0 for each map's decoder-side fit */
    for (int k = 0; k < nmaps; k++) {
        uint8_t *cur = pad[k % 3], *pv = pad[(k + 2) % 3], *p2 = pad[(k + 1) % 3];
        uint8_t *raw = maps + (size_t)k * H * W;
        /* pv / p2 already hold maps k-1 / k-2 (or all-OOB) */
        memset(cur, OOB, PW * PH);
        if (k >= 1 && prm[P_STAGE] >= 1) boundary_buckets(pv, pd);
        if (prm[P_STAGE] >= 2) col_trans_build(pv);
        if (prm[P_MC]) {
            if (!cmap) { cmap = malloc(PW * PH); }
            memset(cmap, OOB, PW * PH);
            if (k >= 2) { if (prm[P_MLR] >= 2) { mcp[5] = 0; mcp[6] = 0; } mc_fit(p2, pv);
                if (prm[P_MCS]) {
                    int q[NMCP] = {0, 0, 0, 0, 0, 0, 0};
                    if (!c->dec) {
                        static uint8_t *cf = NULL; if (!cf) cf = malloc(PW * PH);
                        memset(cf, OOB, PW * PH);
                        for (int yy = 0; yy < H; yy++) memcpy(cf + (yy + B) * PW + B, raw + yy * W, W);
                        mcs_refine(pv, cf, q);
                    }
                    for (int j = 0; j < NQ; j++) mcp[j] += mcs_code(c, j, c->dec ? 0 : q[j] - mcp[j]);
                }
                composite(pv, cmap, mcp, 0, NULL, NULL);
                if (prm[P_MC2]) { if (!cmap2) cmap2 = malloc(PW * PH); memset(cmap2, OOB, PW * PH); warp_full(pv, cmap2, mcp); }
                if (prm[P_MCW] && prm[P_MC2]) {   /* full-warp variants tau -/+ mcd */
                    for (int vi = 0; vi < 2; vi++) {
                        if (!cmapw[vi]) cmapw[vi] = malloc(PW * PH);
                        memset(cmapw[vi], OOB, PW * PH);
                        int q[NMCP]; memcpy(q, mcp, sizeof q);
                        if (vi == 0) q[0] = q[0] > prm[P_MCD] ? q[0] - prm[P_MCD] : 0; else q[0] += prm[P_MCD];
                        warp_full(pv, cmapw[vi], q);
                    }
                }
                if (prm[P_MCV]) {   /* variant reprojections: tau -/+ mcd, dx -/+ 1 (decoder-side, zero side info) */
                    for (int vi = 0; vi < (prm[P_MCV] >= 5 ? 16 : prm[P_MCV] >= 4 ? 12 : prm[P_MCV] >= 2 ? 8 : 4); vi++) {
                        if (!cmapv[vi]) cmapv[vi] = malloc(PW * PH);
                        memset(cmapv[vi], OOB, PW * PH);
                        int q[NMCP]; memcpy(q, mcp, sizeof q);
                        if (vi == 0) q[0] = q[0] > prm[P_MCD] ? q[0] - prm[P_MCD] : 0; else if (vi == 1) q[0] += prm[P_MCD]; else if (vi == 2) q[3] -= 1; else if (vi == 3) q[3] += 1;
                        else if (vi == 4) q[1] -= 4; else if (vi == 5) q[1] += 4; else if (vi == 6) q[2] -= 4; else if (vi == 7) q[2] += 4;
                        else if (vi == 8) q[4] -= 1; else if (vi == 9) q[4] += 1; else if (vi == 10) q[0] = q[0] > 2 * prm[P_MCD] ? q[0] - 2 * prm[P_MCD] : 0; else if (vi == 11) q[0] += 2 * prm[P_MCD];
                        else if (vi == 12) q[0] = q[0] > 3 * prm[P_MCD] ? q[0] - 3 * prm[P_MCD] : 0; else if (vi == 13) q[0] += 3 * prm[P_MCD]; else if (vi == 14) q[1] -= 8; else q[1] += 8;
                        composite(pv, cmapv[vi], q, 0, NULL, NULL);
                    }
                }
                if (prm[P_MC3]) {   /* t-2 reprojected twice (2-step motion) */
                    static uint8_t *tmp3 = NULL; if (!tmp3) tmp3 = malloc(PW * PH); if (!cmap3) cmap3 = malloc(PW * PH);
                    memset(tmp3, OOB, PW * PH); memset(cmap3, OOB, PW * PH);
                    composite(p2, tmp3, mcp, 0, NULL, NULL); composite(tmp3, cmap3, mcp, 0, NULL, NULL);
                }
                if (getenv("TCM_MCDBG")) fprintf(stderr, "map %d mc %d %d %d %d %d\n", k, mcp[0], mcp[1], mcp[2], mcp[3], mcp[4]); }   /* diagnostic only */
            else { if (prm[P_MCW]) for (int vi = 0; vi < 2; vi++) { if (!cmapw[vi]) cmapw[vi] = malloc(PW * PH); memcpy(cmapw[vi], pv, PW * PH); }
                   if (prm[P_MCV]) for (int vi = 0; vi < 16; vi++) { if (!cmapv[vi]) cmapv[vi] = malloc(PW * PH); memcpy(cmapv[vi], pv, PW * PH); }
                   if (prm[P_MC3]) { if (!cmap3) cmap3 = malloc(PW * PH); memcpy(cmap3, pv, PW * PH); }
                   memcpy(cmap, pv, PW * PH); if (prm[P_MC2]) { if (!cmap2) cmap2 = malloc(PW * PH); memcpy(cmap2, pv, PW * PH); } }
        }
        if (prm[P_BD] && prm[P_MC]) { if (!pdc) pdc = malloc(H * W); boundary_buckets(cmap, pdc); }   /* distance to the reprojected map's boundaries */
        if (prm[P_OBJ]) {
            if (!omap) omap = malloc(PW * PH);
            const uint8_t *base = prm[P_MC] ? cmap : pv;
            OBJM = prm[P_OBJM];
            if (k >= 2) obj_build(pv, p2, base, prm[P_OBJC]); else memcpy(omap, base, PW * PH);
        }
        memset(gcnt, 0, sizeof gcnt); gdx = gprev_dx; gdy = gprev_dy;
        for (int y = 0; y < H; y++) {
            if (y >= 1 && prm[P_STAGE] >= 2) {
                if (prm[P_MM] == 0) motion_row(cur, pv, y, prm[P_MV]);
                gshift_row(cur, pv, y, prm[P_GS]);
                const uint8_t *tp = (prm[P_MC] && prm[P_MCX]) ? cmap : pv;   /* mcx=1: edge transfer from the reprojected map */
                edge_pred_row(cur, tp, y, 0, xo_t, xp_t, 1);
                edge_pred_row(cur, tp, y, -1, xo_v[0], xp_v[0], 0);
                edge_pred_row(cur, tp, y, 1, xo_v[1], xp_v[1], 0);
                if (ctx_on[C_X16]) edge_pred_row(cur, p2, y, 0, xo_u, xp_u, 0);
                if (prm[P_MC]) edge_pred_row(cur, cmap, y, 0, xo_c, xp_c, 0);
                if (prm[P_MC] && prm[P_MC2]) edge_pred_row(cur, cmap2, y, 0, xo_c2, xp_c2, 0);
                if (prm[P_OBJ]) edge_pred_row(cur, omap, y, 0, xo_o, xp_o, 0);
            }
            if (prm[P_MC] && prm[P_MCR] && k >= 1 && y > 0) {
                int nr = prm[P_MCR] + 1;
                for (int r = 1; r < nr; r++) if (y == H * r / nr) mc_refit(pv, cur, y);
            }
            if (prm[P_STAGE] >= 2) inrow_reset(pv, y);
            const int irow = prm[P_STAGE] >= 2;
            if (prm[P_STAGE] >= 2 && prm[P_MM] == 1) motion_pix_row(cur, pv, y, prm[P_MV], prm[P_MDY]);
            const int mm = prm[P_STAGE] >= 2 && prm[P_MM] == 1, mw = prm[P_MW];
            uint8_t *row = cur + (y + B) * PW + B;
            if (!c->dec) {      /* encoder: future pixels of this row stay OOB until coded (mirror decoder) */
                for (int x = 0; x < W; x++) { if (mm) motion_pix(cur, pv, y, x, mw); code_pixel(c, &m, cur, pv, p2, pd, y, x, raw[y * W + x]); row[x] = raw[y * W + x]; if (irow) inrow_update(row, x);
                    if (y && row[x] != row[x - PW]) { clt[x] = (int16_t)y; clq[x] = (uint8_t)(row[x - PW] * 6 + row[x]); } }
            } else {
                for (int x = 0; x < W; x++) { if (mm) motion_pix(cur, pv, y, x, mw); int t = code_pixel(c, &m, cur, pv, p2, pd, y, x, 0); row[x] = (uint8_t)t; raw[y * W + x] = (uint8_t)t; if (irow) inrow_update(row, x);
                    if (y && row[x] != row[x - PW]) { clt[x] = (int16_t)y; clq[x] = (uint8_t)(row[x - PW] * 6 + row[x]); } }
            }
        }
        gprev_dx = gdx; gprev_dy = gdy;
    }
#ifdef STATS
    for (int k = 0; k < m.na; k++) {   /* diagnostic: hash-table occupancy per active context */
        HashTab *h = &m.ht[m.act[k]]; long used = 0, n = 1L << h->bits;
        for (long i = 0; i < n; i++) used += h->t[i].chk != 0;
        fprintf(stderr, "  ctx %2d occupancy %ld / %ld\n", m.act[k], used, n);
    }
#endif
}

/* header: 'T' + FMT_VER | nmaps u16 | context-off mask 6 B | NPARAM_USED bytes (limit stored as (limit-3)/4).
 * FMT_VER is bumped for every frozen decoder format; the decoder rejects any other version.
 * ('2' = snap_best .. snap_T104894; '3' = first versioned snapshot) */
#define FMT_VER 'I'   /* working copy */
#define MAGIC_OK(b) ((b)[0] == 'T' && (b)[1] == FMT_VER)
#define HDR (2 + 2 + 6 + NPARAM_USED)
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s e|d in out [key=val ...]\n", argv[0]); return 2; }
    Coder c; memset(&c, 0, sizeof c); c.x2 = ~(uint64_t)0;
    prm_default();
    if (argv[1][0] == 'e') {
        for (int i = 4; i < argc; i++) {          /* each argv may hold several whitespace-separated key=val tokens */
            char buf[1024]; strncpy(buf, argv[i], sizeof buf - 1); buf[sizeof buf - 1] = 0;
            for (char *tok = strtok(buf, " \t"); tok; tok = strtok(NULL, " \t")) {
                char *eq = strchr(tok, '='), *end; int ok = 0;
                if (!eq) { fprintf(stderr, "bad param %s\n", tok); return 2; }
                *eq = 0;
                if (!strcmp(tok, "off")) { mask_off = strtoull(eq + 1, &end, 0); ok = *end == 0; }
                else for (int j = 0; j < NPARAM_USED; j++) if (!strcmp(tok, pname[j])) { prm[j] = (int)strtol(eq + 1, &end, 0); ok = *end == 0; }
                if (!ok) { fprintf(stderr, "bad param %s=%s\n", tok, eq + 1); return 2; }
            }
        }
        size_t len; uint8_t *maps = read_file(argv[2], &len);
        if (len % (H * W)) { fprintf(stderr, "input size %zu not a multiple of %d\n", len, H * W); return 1; }
        int nmaps = (int)(len / (H * W));
        for (size_t i = 0; i < len; i++) if (maps[i] > 4) { fprintf(stderr, "symbol > 4 at %zu\n", i); return 1; }
        if ((prm[P_LIMIT] & 3) != 3 || prm[P_LIMIT] > 1023) { fprintf(stderr, "limit must be 4k+3 <= 1023\n"); return 2; }
        for (int j = 0; j < NPARAM_USED; j++) if (j != P_LIMIT && (prm[j] < 0 || prm[j] > 255)) { fprintf(stderr, "param %s out of 0..255\n", pname[j]); return 2; }
        if (mask_off >> 48) { fprintf(stderr, "mask too wide\n"); return 2; }
        LIMIT = prm[P_LIMIT] << prm[P_LIMX]; if (LIMIT > 4095) LIMIT = 4095;
        put_byte(&c, 'T'); put_byte(&c, FMT_VER);
        put_byte(&c, (uint8_t)nmaps); put_byte(&c, (uint8_t)(nmaps >> 8));
        for (int i = 0; i < 6; i++) put_byte(&c, (uint8_t)(mask_off >> (8 * i)));
        for (int j = 0; j < NPARAM_USED; j++) put_byte(&c, (uint8_t)(j == P_LIMIT ? (prm[j] - 3) / 4 : prm[j]));
        run(&c, maps, nmaps);
        coder_flush(&c);
        write_file(argv[3], c.buf, c.pos);
        fprintf(stderr, "encoded %d maps -> %zu bytes (hdr %d) |", nmaps, c.pos, HDR);
        for (int j = 0; j < NPARAM_USED; j++) fprintf(stderr, " %s=%d", pname[j], prm[j]);
        fprintf(stderr, " off=0x%llx\n", (unsigned long long)mask_off);
#ifdef STATS
        double tot = 0;
        for (int a = 0; a < 8; a++) {
            double ct = 0; for (int n = 0; n < 4; n++) ct += st_cost[a][n] / 8;
            fprintf(stderr, "  W!=N:%d otcat(0:|o|<=1,1:<=3,2:<=8,3:far/none):%d %d n=%9ld  node0 %6.0f B (events n=%6ld %6.0f B)  node1 %5.0f  node2+3 %4.0f  all %6.0f\n", a >> 2, (a >> 1) & 1, a & 1,
                    st_cnt[a][0], st_cost[a][0] / 8, st_nev[a], st_ev[a] / 8, st_cost[a][1] / 8, (st_cost[a][2] + st_cost[a][3]) / 8, ct);
            tot += ct; }
        fprintf(stderr, "  total model bytes %.0f\n", tot);
#endif
    } else {
        size_t len; uint8_t *in = read_file(argv[2], &len);
        if (len < HDR || !MAGIC_OK(in)) {
            fprintf(stderr, "bad header: stream format '%c%c', this decoder only reads 'T%c'\n", len ? in[0] : '?', len > 1 ? in[1] : '?', FMT_VER); return 1; }
        int nmaps = in[2] | (in[3] << 8);
        mask_off = 0; for (int i = 0; i < 6; i++) mask_off |= (uint64_t)in[4 + i] << (8 * i);
        for (int j = 0; j < NPARAM_USED; j++) prm[j] = j == P_LIMIT ? in[10 + j] * 4 + 3 : in[10 + j];
        LIMIT = prm[P_LIMIT] << prm[P_LIMX]; if (LIMIT > 4095) LIMIT = 4095;
        c.dec = 1; c.in = in; c.in_len = len; c.in_pos = HDR;
        for (int i = 0; i < 8; i++) c.x = (c.x << 8) | get_byte(&c);
        uint8_t *maps = calloc((size_t)nmaps * H * W, 1);
        run(&c, maps, nmaps);
        write_file(argv[3], maps, (size_t)nmaps * H * W);
        fprintf(stderr, "decoded %d maps\n", nmaps);
    }
    return 0;
}
