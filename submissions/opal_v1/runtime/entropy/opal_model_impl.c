#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    H = 384, W = 512, HW = H * W,
    CELL_H = 64, CELL_W = 64, CELL_ROWS = 6, CELL_COLS = 8,
    CELLS = 48, GROUPS = 190, FRAME_FAMILIES = 44,
    SYNC_FAMILIES = 11, FAMILIES = 55, CBINS = 101
};

typedef struct {
    uint32_t size;
    uint64_t offset;
    float *gradient;
    float *curvature;
    double meta_gradient;
    double meta_curvature;
    double multiplier;
} family;

typedef struct { void *mapping; size_t bytes; void *data; } mapped_npy;

static mapped_npy map_npy(const char *path) {
    mapped_npy out = {0};
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); exit(2); }
    out.bytes = (size_t)st.st_size;
    out.mapping = mmap(NULL, out.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (out.mapping == MAP_FAILED) { perror("mmap"); exit(2); }
    const uint8_t *p = (const uint8_t *)out.mapping;
    if (out.bytes < 12 || memcmp(p, "\x93NUMPY", 6)) {
        fprintf(stderr, "%s is not an npy file\n", path); exit(2);
    }
    size_t header;
    if (p[6] == 1) {
        header = 10u + p[8] + ((size_t)p[9] << 8);
    } else {
        header = 12u + p[8] + ((size_t)p[9] << 8) +
                 ((size_t)p[10] << 16) + ((size_t)p[11] << 24);
    }
    if (header >= out.bytes) { fprintf(stderr, "%s has invalid npy header\n", path); exit(2); }
    out.data = (uint8_t *)out.mapping + header;
    return out;
}

static inline double sigmoid(double z) {
    if (z >= 0.0) { double e = exp(-z); return 1.0 / (1.0 + e); }
    { double e = exp(z); return e / (1.0 + e); }
}

static inline double clamp(double value, double lo, double hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static inline double loss_bits(unsigned outcome, double q) {
    q = clamp(q, 1e-15, 1.0 - 1e-15);
    return outcome ? -log2(q) : -log2(1.0 - q);
}

static inline int cell_of(int y, int x) { return (y >> 6) * CELL_COLS + (x >> 6); }
static inline int group_of(int y, int x) { return (x & 63) + 2 * (y & 63); }

static inline unsigned defect_at(const uint8_t *d, int frame, int y, int x) {
    if (frame < 0 || y < 0 || y >= H || x < 0 || x >= W) return 0u;
    return d[(size_t)frame * HW + (size_t)y * W + x] != 0;
}

static inline unsigned same_cell_at(
    const uint8_t *d, int frame, int y, int x, int cy, int cx
) {
    if (y < 0 || y >= H || x < 0 || x >= W || cell_of(y, x) != cell_of(cy, cx)) return 0u;
    return defect_at(d, frame, y, x);
}

static inline unsigned causal_at(
    const uint8_t *d, int frame, int y, int x, int group, int position
) {
    if (y < 0 || y >= H || x < 0 || x >= W) return 0u;
    int neighbour_group = group_of(y, x);
    int neighbour_position = y * W + x;
    if (neighbour_group > group ||
        (neighbour_group == group && neighbour_position >= position)) return 0u;
    return defect_at(d, frame, y, x);
}

static inline unsigned run_bucket(uint32_t run) { return run < 15u ? run : 15u; }

static uint8_t spatial_d4[256], spatial_c4[256], word_reverse[256], word_necklace[256];

static unsigned reverse8(unsigned word) {
    unsigned out = 0u;
    int bit;
    for (bit = 0; bit < 8; ++bit) out |= ((word >> bit) & 1u) << (7 - bit);
    return out;
}

static unsigned rotate8(unsigned word, int shift) {
    return ((word << shift) | (word >> (8 - shift))) & 255u;
}

static int neighbour_index(int y, int x) {
    static const int ys[8] = {-1,-1,-1,0,0,1,1,1};
    static const int xs[8] = {-1,0,1,-1,1,-1,0,1};
    int i;
    for (i = 0; i < 8; ++i) if (ys[i] == y && xs[i] == x) return i;
    return -1;
}

static void init_word_orbits(void) {
    static const int ys[8] = {-1,-1,-1,0,0,1,1,1};
    static const int xs[8] = {-1,0,1,-1,1,-1,0,1};
    int word;
    for (word = 0; word < 256; ++word) {
        unsigned best4 = 255u, best8 = 255u, best_necklace = 255u;
        int reflect, rotation;
        word_reverse[word] = (uint8_t)reverse8((unsigned)word);
        for (reflect = 0; reflect < 2; ++reflect) {
            for (rotation = 0; rotation < 4; ++rotation) {
                unsigned transformed = 0u;
                int bit;
                for (bit = 0; bit < 8; ++bit) if (((unsigned)word >> bit) & 1u) {
                    int y = ys[bit], x = xs[bit], step;
                    if (reflect) x = -x;
                    for (step = 0; step < rotation; ++step) {
                        int old_y = y; y = x; x = -old_y;
                    }
                    int mapped = neighbour_index(y, x);
                    if (mapped < 0) abort();
                    transformed |= 1u << mapped;
                }
                if (transformed < best8) best8 = transformed;
                if (!reflect && transformed < best4) best4 = transformed;
            }
        }
        for (reflect = 0; reflect < 2; ++reflect) {
            unsigned seed = reflect ? reverse8((unsigned)word) : (unsigned)word;
            for (rotation = 0; rotation < 8; ++rotation) {
                unsigned transformed = rotation ? rotate8(seed, rotation) : seed;
                if (transformed < best_necklace) best_necklace = transformed;
            }
        }
        spatial_c4[word] = (uint8_t)best4;
        spatial_d4[word] = (uint8_t)best8;
        word_necklace[word] = (uint8_t)best_necklace;
    }
}

static unsigned word_feature(unsigned word, int mode, int spatial) {
    if (mode == 0) return word;
    if (mode == 1) return spatial ? spatial_d4[word] : (word < word_reverse[word] ? word : word_reverse[word]);
    if (mode == 2) return spatial ? spatial_c4[word] : word_necklace[word];
    if (mode == 3) return word_necklace[word];
    if (mode == 4) {
        unsigned transitions = 0u, bit;
        for (bit = 1; bit < 8; ++bit) transitions += ((word >> bit) ^ (word >> (bit - 1))) & 1u;
        return (unsigned)__builtin_popcount(word) * 18u + ((word >> 7) & 1u) * 9u + transitions;
    }
    return (unsigned)__builtin_popcount(word);
}

static uint64_t init_catalogue(family fs[FAMILIES]) {
    int j = 0;
#define S(value) fs[j++].size = (uint32_t)(value)
    S(CBINS); S(GROUPS); S(CELLS); S(CBINS * GROUPS); S(CBINS * CELLS);
    S(CELLS * GROUPS); S(CBINS * CELLS * GROUPS);
    S(2); S(CBINS * 2); S(GROUPS * 2); S(CELLS * GROUPS * 2);
    S(CBINS * CELLS * GROUPS * 2);
    S(10); S(CBINS * 10); S(GROUPS * 10); S(CELLS * GROUPS * 10);
    S(256); S(CBINS * 256); S(GROUPS * 256); S(CELLS * 256);
    S(9); S(CBINS * 9); S(GROUPS * 9); S(CELLS * GROUPS * 9);
    S(256); S(CBINS * 256); S(GROUPS * 256); S(CELLS * 256);
    S(4); S(CBINS * 4); S(GROUPS * 4); S(CELLS * GROUPS * 4);
    S(8); S(CBINS * 8); S(GROUPS * 8); S(CELLS * GROUPS * 8);
    S(9); S(CBINS * 9); S(GROUPS * 9); S(CELLS * GROUPS * 9);
    S(256); S(CBINS * 256); S(GROUPS * 256); S(CELLS * 256);
    S(256); S(CBINS * 256); S(GROUPS * 256); S(CELLS * 256);
    S(CELLS * GROUPS * 256); S(9); S(CBINS * GROUPS * 9);
    S(GROUPS * 16); S(CELLS * GROUPS * 8); S(GROUPS * 16); S(GROUPS * 256);
#undef S
    if (j != FAMILIES) { fprintf(stderr, "family count mismatch %d\n", j); exit(3); }
    uint64_t total = 0u;
    for (j = 0; j < FAMILIES; ++j) {
        fs[j].offset = total;
        total += fs[j].size;
        fs[j].gradient = calloc(fs[j].size, sizeof(float));
        fs[j].curvature = calloc(fs[j].size, sizeof(float));
        fs[j].multiplier = 1.0;
        if (!fs[j].gradient || !fs[j].curvature) { fprintf(stderr, "allocation failed\n"); exit(3); }
    }
    if (total != 6175440u) { fprintf(stderr, "sector mismatch %llu\n", (unsigned long long)total); exit(3); }
    return total;
}

static void cross_neighbourhood(
    const uint8_t *d, int frame, int y, int x, int mode,
    unsigned *word, unsigned *count
) {
    static const int dy[8] = {-1,-1,-1,0,0,1,1,1};
    static const int dx[8] = {-1,0,1,-1,1,-1,0,1};
    int z;
    *word = 0u; *count = 0u;
    for (z = 0; z < 8; ++z) {
        int ny, nx;
        unsigned value = 0u;
        if (mode == 0) {
            ny = y + dy[z]; nx = x + dx[z];
            if (ny >= 0 && ny < H && nx >= 0 && nx < W && cell_of(ny, nx) != cell_of(y, x))
                value = defect_at(d, frame - 1, ny, nx);
        } else {
            ny = y + 64 * dy[z]; nx = x + 64 * dx[z];
            value = defect_at(d, frame - 1, ny, nx);
        }
        *word |= value << z;
        *count += value;
    }
}

int main(int argc, char **argv) {
    if (argc < 8 || argc > 25) {
        fprintf(stderr,
                "usage: %s defects.npy logodds.npy frames cross_mode variant_mask causal_mode "
                "confidence_mode [update_mode [scale [local_lambda [local_cap "
                "[meta_lambda [meta_cap [confidence_multiplier [local_word_mode "
                "[cross_word_mode [temporal_word_mode [causal_word_mode "
                "[meta_mode [family_mask [sync_mode [geometry_mode [class_mode "
                "[argmax.npy]]]]]]]]]]]]]]]]\n",
                argv[0]);
        return 2;
    }
    mapped_npy dm = map_npy(argv[1]);
    mapped_npy bm = map_npy(argv[2]);
    const uint8_t *d = (const uint8_t *)dm.data;
    const float *base = (const float *)bm.data;
    int frames = atoi(argv[3]);
    int cross_mode = atoi(argv[4]);
    unsigned variant_mask = (unsigned)strtoul(argv[5], NULL, 0);
    int causal_mode = atoi(argv[6]);
    int confidence_mode = atoi(argv[7]);
    int update_mode = argc > 8 ? atoi(argv[8]) : 0;
    double scale = argc > 9 ? atof(argv[9]) : 0.15;
    double local_lambda = argc > 10 ? atof(argv[10]) : 1.0;
    double local_cap = argc > 11 ? atof(argv[11]) : 4.0;
    double meta_lambda = argc > 12 ? atof(argv[12]) : 4.0;
    double meta_cap = argc > 13 ? atof(argv[13]) : 4.0;
    double confidence_multiplier = argc > 14 ? atof(argv[14]) : 800.0;
    int local_word_mode = argc > 15 ? atoi(argv[15]) : 0;
    int cross_word_mode = argc > 16 ? atoi(argv[16]) : 0;
    int temporal_word_mode = argc > 17 ? atoi(argv[17]) : 0;
    int causal_word_mode = argc > 18 ? atoi(argv[18]) : 0;
    int meta_mode = argc > 19 ? atoi(argv[19]) : 0;
    uint64_t family_mask = argc > 20 ? strtoull(argv[20], NULL, 0) : ((UINT64_C(1) << FAMILIES) - 1u);
    int sync_mode = argc > 21 ? atoi(argv[21]) : 0;
    int geometry_mode = argc > 22 ? atoi(argv[22]) : 0;
    int class_mode = argc > 23 ? atoi(argv[23]) : 0;
    if (frames < 1 || frames > 600 || cross_mode < 0 || cross_mode > 1 ||
        variant_mask > 0x1ffu || causal_mode < 0 || causal_mode > 9 ||
        confidence_mode < 0 || confidence_mode > 12 || update_mode < 0 || update_mode > 4 ||
        scale < 0.0 || local_lambda <= 0.0 || local_cap <= 0.0 ||
        meta_lambda <= 0.0 || meta_cap <= 0.0 || confidence_multiplier <= 0.0 ||
        local_word_mode < 0 || local_word_mode > 5 || cross_word_mode < 0 || cross_word_mode > 5 ||
        temporal_word_mode < 0 || temporal_word_mode > 5 ||
        causal_word_mode < 0 || causal_word_mode > 5 || meta_mode < 0 || meta_mode > 4 ||
        family_mask >= (UINT64_C(1) << FAMILIES) || sync_mode < 0 || sync_mode > 11 ||
        geometry_mode < 0 || geometry_mode > 7 || class_mode < 0 || class_mode > 4 ||
        (class_mode > 0 && argc < 25)) return 2;

    mapped_npy am = {0};
    const uint8_t *argmax = NULL;
    if (class_mode > 0) {
        am = map_npy(argv[24]);
        argmax = (const uint8_t *)am.data;
    }

    init_word_orbits();

    family fs[FAMILIES] = {{0}};
    uint64_t sectors = init_catalogue(fs);
    int starts[GROUPS + 1] = {0};
    int *ordered = malloc((size_t)HW * sizeof(int));
    int cursor[GROUPS];
    int y, x, j;
    if (!ordered) return 3;
    for (y = 0; y < H; ++y) for (x = 0; x < W; ++x) starts[1 + group_of(y, x)]++;
    for (j = 1; j <= GROUPS; ++j) starts[j] += starts[j - 1];
    memcpy(cursor, starts, sizeof(cursor));
    for (y = 0; y < H; ++y) for (x = 0; x < W; ++x) {
        int g = group_of(y, x); ordered[cursor[g]++] = y * W + x;
    }

    double baseline_bits = 0.0, modeled_bits = 0.0;
    uint64_t defects = 0u, symbols = 0u;
    int frame, group;
    static const int neighbour_dy[8] = {-1,-1,-1,0,0,1,1,1};
    static const int neighbour_dx[8] = {-1,0,1,-1,1,-1,0,1};
    static const int causal_dy[8] = {0,-1,-1,0,-1,1,-1,-1};
    static const int causal_dx[8] = {-1,2,-1,-2,-2,-4,0,1};

    for (frame = 0; frame < frames; ++frame) {
        for (group = 0; group < GROUPS; ++group) {
            uint32_t prefix = 0u, run = UINT32_MAX;
            uint8_t sequence = 0u, cell_prefix[CELLS] = {0};
            int oi;
            for (oi = starts[group]; oi < starts[group + 1]; ++oi) {
                int position = ordered[oi];
                y = position / W; x = position % W;
                int cell = cell_of(y, x);
                double b = (double)base[(size_t)frame * HW + position];
                unsigned maximal_class = class_mode > 0
                    ? argmax[(size_t)frame * HW + position] : 0u;
                double q0 = sigmoid(b);
                int conf;
                if (confidence_mode == 0) conf = (int)floor((25.0 + b) / 0.25);
                else if (confidence_mode == 1) conf = (int)floor(100.0 * q0 + 0.5);
                else if (confidence_mode == 2) conf = (int)floor(100.0 * q0);
                else if (confidence_mode == 3) conf = (int)floor(200.0 * q0 + 0.5);
                else if (confidence_mode == 4) conf = (int)floor(400.0 * q0 + 0.5);
                else if (confidence_mode == 5) conf = (int)floor(800.0 * q0 + 0.5);
                else if (confidence_mode == 6) conf = (int)floor(50.0 * q0 + 0.5);
                else if (confidence_mode == 7) conf = (int)floor(100.0 * sqrt(q0) + 0.5);
                else if (confidence_mode == 8) conf = (int)floor(-10.0 * log10(q0) + 0.5);
                else if (confidence_mode == 9) conf = (int)floor(1600.0 * q0 + 0.5);
                else if (confidence_mode == 10) conf = (int)floor(3200.0 * q0 + 0.5);
                else if (confidence_mode == 11) conf = (int)floor(1000.0 * q0 + 0.5);
                else conf = (int)floor(confidence_multiplier * q0 + 0.5);
                if (conf < 0) conf = 0; else if (conf >= CBINS) conf = CBINS - 1;
                unsigned outcome = defect_at(d, frame, y, x);
                unsigned prev = defect_at(d, frame - 1, y, x);
                unsigned local_word = 0u, local_count = prev;
                unsigned cross_word = 0u, cross_count = 0u;
                unsigned temporal_word = 0u, temporal_count = 0u;
                unsigned causal_word = 0u, causal_count = 0u;
                unsigned causal_prev_word = 0u;
                int z;
                for (z = 0; z < 8; ++z) {
                    unsigned value = same_cell_at(
                        d, frame - 1, y + neighbour_dy[z], x + neighbour_dx[z], y, x
                    );
                    local_word |= value << z; local_count += value;
                }
                cross_neighbourhood(d, frame, y, x, cross_mode, &cross_word, &cross_count);
                unsigned lag12 = defect_at(d, frame - 1, y, x) |
                                 (defect_at(d, frame - 2, y, x) << 1);
                unsigned lag124 = lag12 | (defect_at(d, frame - 4, y, x) << 2);
                for (z = 1; z <= 8; ++z) {
                    unsigned value = defect_at(d, frame - z, y, x);
                    temporal_word |= value << (z - 1); temporal_count += value;
                }
                for (z = 0; z < 8; ++z) {
                    int cy, cx;
                    if (causal_mode == 0) {
                        cy = y + causal_dy[z]; cx = x + causal_dx[z];
                    } else if (causal_mode == 1) {
                        static const int tile_dy[8] = {0,0,-1,-1,-1,-2,-2,-2};
                        static const int tile_dx[8] = {-1,-2,-1,0,1,-1,0,1};
                        cy = y + 64 * tile_dy[z]; cx = x + 64 * tile_dx[z];
                    } else if (causal_mode == 2) {
                        static const int hybrid_dy[8] = {0,-1,-1,0,0,-1,-1,-1};
                        static const int hybrid_dx[8] = {-1,2,-1,-2,-64,-64,0,64};
                        cy = y + hybrid_dy[z]; cx = x + hybrid_dx[z];
                    } else if (causal_mode == 3) {
                        static const int near_dy[8] = {-1,-1,-1,0,0,1,1,1};
                        static const int near_dx[8] = {-1,0,1,-1,1,-1,0,1};
                        cy = y + near_dy[z]; cx = x + near_dx[z];
                    } else {
                        static const int candidate_dy[6][8] = {
                            {0,1,-1,1,-1,-1,0,-3},
                            {0,-1,-1,0,-1,1,-1,1},
                            {0,-1,-1,0,1,1,-1,-3},
                            {0,-1,-1,0,-1,1,-1,-3},
                            {0,1,-1,1,-1,-1,-3,1},
                            {0,1,-1,1,-1,-1,0,1}
                        };
                        static const int candidate_dx[6][8] = {
                            {-1,-4,2,-5,1,-1,-2,4},
                            {-1,2,-1,-2,-2,-4,1,-5},
                            {-1,2,-1,-2,-5,-4,1,4},
                            {-1,2,-1,-2,-2,-4,1,4},
                            {-1,-4,2,-5,1,-1,4,-3},
                            {-1,-4,2,-5,1,-1,-2,-3}
                        };
                        int candidate = causal_mode - 4;
                        cy = y + candidate_dy[candidate][z];
                        cx = x + candidate_dx[candidate][z];
                    }
                    unsigned value = causal_at(d, frame, cy, cx, group, position);
                    unsigned previous_value = causal_at(d, frame - 1, cy, cx, group, position);
                    causal_word |= value << z; causal_count += value;
                    causal_prev_word |= previous_value << z;
                }
                local_word = word_feature(local_word, local_word_mode, 1);
                cross_word = word_feature(cross_word, cross_word_mode, 1);
                temporal_word = word_feature(temporal_word, temporal_word_mode, 0);
                causal_word = word_feature(causal_word, causal_word_mode, 0);
                causal_prev_word = word_feature(causal_prev_word, causal_word_mode, 0);
                unsigned causal_holonomy_word = causal_word ^ causal_prev_word;
                unsigned causal_holonomy_count = (unsigned)__builtin_popcount(causal_holonomy_word);
                unsigned local8 = ((unsigned)(y & 63) >> 3) * 8u + ((unsigned)(x & 63) >> 3);
                unsigned local4 = ((unsigned)(y & 63) >> 2) * 16u + ((unsigned)(x & 63) >> 2);
                unsigned local16 = ((unsigned)(y & 63) >> 4) * 4u + ((unsigned)(x & 63) >> 4);
                uint32_t id[FAMILIES]; int k = 0;
#define ID(value) id[k++] = (uint32_t)(value)
                ID(conf); ID(group); ID(cell); ID(conf * GROUPS + group);
                ID(conf * CELLS + cell); ID(cell * GROUPS + group);
                ID((variant_mask & 1u) ? (uint32_t)((conf * CELLS + cell) * 64) + local8
                                       : (uint32_t)((conf * CELLS + cell) * GROUPS + group));
                ID(prev); ID(conf * 2 + prev); ID(group * 2 + prev);
                ID((cell * GROUPS + group) * 2 + prev);
                ID((variant_mask & 2u) ? ((conf * 2 + prev) * CELLS + cell) * 64 + local8
                                       : ((conf * CELLS + cell) * GROUPS + group) * 2 + prev);
                ID(local_count); ID(conf * 10 + local_count); ID(group * 10 + local_count);
                ID((cell * GROUPS + group) * 10 + local_count);
                ID(local_word); ID(conf * 256 + local_word); ID(group * 256 + local_word);
                ID((variant_mask & 4u) ? cell * 64 + local8 : cell * 256 + local_word);
                ID(cross_count); ID(conf * 9 + cross_count); ID(group * 9 + cross_count);
                ID((variant_mask & 8u) ? (conf * CELLS + cell) * 16 + local16
                                       : (cell * GROUPS + group) * 9 + cross_count);
                ID(cross_word); ID(conf * 256 + cross_word); ID(group * 256 + cross_word);
                ID((variant_mask & 16u) ? cell * 256 + local4 : cell * 256 + cross_word);
                ID(lag12); ID(conf * 4 + lag12); ID(group * 4 + lag12);
                ID((variant_mask & 32u) ? (temporal_count * CELLS + cell) * 64 + local8
                                       : (cell * GROUPS + group) * 4 + lag12);
                ID(lag124); ID((variant_mask & 64u) ? cell * 16 + local16 : conf * 8 + lag124);
                ID(group * 8 + lag124); ID((cell * GROUPS + group) * 8 + lag124);
                ID(temporal_count); ID(conf * 9 + temporal_count); ID(group * 9 + temporal_count);
                ID((cell * GROUPS + group) * 9 + temporal_count);
                ID(temporal_word);
                ID((variant_mask & 128u) ? (lag124 * CELLS + cell) * 64 + local8
                                       : conf * 256 + temporal_word);
                ID(group * 256 + temporal_word);
                ID((variant_mask & 256u) ? (lag12 * CELLS + cell) * 64 + local8
                                       : cell * 256 + temporal_word);
                if (geometry_mode == 1 || geometry_mode == 3 || geometry_mode == 4) {
                    id[1] = local8;
                    id[3] = (uint32_t)(conf * 64 + local8);
                    id[4] = (uint32_t)(cell * 64 + local8);
                    id[5] = (uint32_t)(group * 16 + local16);
                }
                if (geometry_mode == 1 || geometry_mode == 3 || geometry_mode == 5) {
                    id[16] = local4;
                    id[17] = (uint32_t)(conf * 256 + local4);
                    id[18] = (uint32_t)(group * 256 + local4);
                }
                if (geometry_mode == 2 || geometry_mode == 3 || geometry_mode == 6) {
                    id[25] = (uint32_t)((conf * 4 + lag12) * 64 + local8);
                    id[26] = (uint32_t)((group * 4 + lag12) * 64 + local8);
                }
                if (geometry_mode == 2 || geometry_mode == 3 || geometry_mode == 7) {
                    id[30] = (uint32_t)(cell * 4 + lag12);
                    id[34] = (uint32_t)(group * 4 + lag12);
                    id[38] = (uint32_t)(conf * 16 + local16);
                    id[39] = (uint32_t)((cell * 8 + lag124) * 64 + local8);
                    id[40] = (uint32_t)(lag12 * 64 + local8);
                    id[41] = (uint32_t)((conf * 8 + lag124) * 16 + local16);
                    id[42] = (uint32_t)((group * 8 + lag124) * 16 + local16);
                }
                if (class_mode == 1 || class_mode == 4) {
                    id[1] = maximal_class;
                    id[3] = (uint32_t)(conf * 5 + maximal_class);
                    id[4] = (uint32_t)(cell * 5 + maximal_class);
                    id[5] = (uint32_t)(group * 5 + maximal_class);
                }
                if (class_mode == 2 || class_mode == 4) {
                    id[16] = (uint32_t)(maximal_class * 16 + local16);
                    id[17] = (uint32_t)((conf * 5 + maximal_class) * 16 + local16);
                    id[18] = (uint32_t)((group * 5 + maximal_class) * 16 + local16);
                }
                if (class_mode == 3 || class_mode == 4) {
                    id[25] = (uint32_t)((conf * 5 + maximal_class) * 4 + lag12);
                    id[26] = (uint32_t)((group * 5 + maximal_class) * 4 + lag12);
                    id[30] = (uint32_t)(cell * 5 + maximal_class);
                    id[34] = (uint32_t)(group * 5 + maximal_class);
                    id[38] = (uint32_t)(conf * 5 + maximal_class);
                    id[39] = (uint32_t)((conf * 5 + maximal_class) * CELLS + cell);
                    id[40] = (uint32_t)(cell * 5 + maximal_class);
                    id[41] = (uint32_t)((conf * 5 + maximal_class) * 8 + lag124);
                    id[42] = (uint32_t)((group * 5 + maximal_class) * 8 + lag124);
                }
                uint32_t sync_id[11] = {
                    causal_word,
                    (uint32_t)(conf * 256 + causal_word),
                    (uint32_t)(group * 256 + causal_word),
                    (uint32_t)(cell * 256 + causal_word),
                    (uint32_t)((cell * GROUPS + group) * 256 + causal_word),
                    causal_count,
                    (uint32_t)((conf * GROUPS + group) * 9 + causal_count),
                    (uint32_t)(group * 16 + (prefix < 15u ? prefix : 15u)),
                    (uint32_t)((cell * GROUPS + group) * 8 +
                               (cell_prefix[cell] < 7u ? cell_prefix[cell] : 7u)),
                    (uint32_t)(group * 16 + run_bucket(run)),
                    (uint32_t)(group * 256 + sequence)
                };
                if (sync_mode >= 1) {
                    sync_id[4] = (uint32_t)((conf * CELLS + cell) * 256 + causal_word);
                }
                if (sync_mode == 2) {
                    sync_id[2] = (uint32_t)(conf * 256 + causal_holonomy_word);
                    sync_id[6] = (uint32_t)((conf * GROUPS + group) * 9 + causal_holonomy_count);
                    sync_id[7] = (uint32_t)(group * 9 + causal_holonomy_count);
                    sync_id[8] = (uint32_t)((cell * GROUPS + group) * 8 +
                                            (causal_holonomy_count < 7u ? causal_holonomy_count : 7u));
                    sync_id[9] = (uint32_t)(lag124 * 256 + causal_holonomy_word);
                    sync_id[10] = (uint32_t)(group * 256 + causal_holonomy_word);
                } else if (sync_mode == 3 || sync_mode == 4 || sync_mode == 5) {
                    sync_id[2] = sync_mode == 3
                        ? (uint32_t)(conf * 256 + causal_holonomy_word)
                        : (sync_mode == 4
                            ? (uint32_t)(local8 * 256 + causal_word)
                            : (uint32_t)(((conf < 95 ? conf : 94) * 2 + prev) * 256 + causal_word));
                    sync_id[5] = causal_holonomy_count;
                    sync_id[6] = (uint32_t)((conf * CELLS + cell) * 9 + causal_count);
                    sync_id[7] = (uint32_t)(conf * 9 + causal_holonomy_count);
                    sync_id[8] = (uint32_t)((conf * CELLS + cell) * 9 + causal_holonomy_count);
                    sync_id[9] = (uint32_t)(lag124 * 256 + causal_holonomy_word);
                    sync_id[10] = (uint32_t)(local8 * 256 +
                                             (sync_mode == 4 ? causal_holonomy_word : causal_word));
                } else if (sync_mode >= 6) {
                    sync_id[2] = (uint32_t)(conf * 256 + causal_holonomy_word);
                    sync_id[5] = causal_holonomy_count;
                    sync_id[6] = (uint32_t)((conf * 64 + local8) * 9 + causal_count);
                    sync_id[7] = (uint32_t)(conf * 9 + causal_holonomy_count);
                    sync_id[8] = (uint32_t)((cell * 64 + local8) * 9 + causal_holonomy_count);
                    sync_id[9] = (uint32_t)(lag124 * 256 + causal_holonomy_word);
                    sync_id[10] = (uint32_t)(local8 * 256 + causal_word);
                    if (sync_mode >= 7) {
                        sync_id[7] = (uint32_t)(lag124 * 256 + causal_word);
                        if (sync_mode == 7) {
                            sync_id[6] = (uint32_t)((conf * 4 + lag12) * 256 + causal_word);
                        } else if (sync_mode == 8) {
                            sync_id[6] = (uint32_t)((conf * 4 + lag12) * 256 + causal_holonomy_word);
                            sync_id[8] = (uint32_t)((conf * 64 + local8) * 9 + causal_holonomy_count);
                        } else if (sync_mode == 9) {
                            sync_id[6] = (uint32_t)((cell * 8 + lag124) * 256 + causal_word);
                        } else if (sync_mode == 10) {
                            sync_id[6] = (uint32_t)((local8 * 4 + lag12) * 256 + causal_word);
                        } else {
                            sync_id[6] = (uint32_t)((conf * 2 + prev) * 256 + causal_word);
                            sync_id[8] = (uint32_t)((conf * 4 + lag12) * 9 + causal_holonomy_count);
                        }
                    }
                }
                for (z = 0; z < 11; ++z) ID(sync_id[z]);
#undef ID
                if (k != FAMILIES) return 3;
                double weight[FAMILIES], multiplier[FAMILIES], sum = 0.0;
                for (j = 0; j < FAMILIES; ++j) {
                    if (id[j] >= fs[j].size) {
                        fprintf(stderr, "id overflow family %d id %u size %u\n", j, id[j], fs[j].size);
                        return 3;
                    }
                    if (!(family_mask & (UINT64_C(1) << j))) {
                        weight[j] = 0.0;
                        multiplier[j] = 0.0;
                        continue;
                    }
                    weight[j] = -(double)fs[j].gradient[id[j]] /
                                ((double)fs[j].curvature[id[j]] + local_lambda);
                    weight[j] = clamp(weight[j], -local_cap, local_cap);
                    if (meta_mode == 0) {
                        multiplier[j] = clamp(
                            1.0 - fs[j].meta_gradient / (fs[j].meta_curvature + meta_lambda),
                            -meta_cap, meta_cap
                        );
                    } else if (meta_mode == 1) {
                        multiplier[j] = 1.0;
                    } else if (meta_mode == 2) {
                        multiplier[j] = clamp(
                            -fs[j].meta_gradient / (fs[j].meta_curvature + meta_lambda),
                            -meta_cap, meta_cap
                        );
                    } else {
                        if (meta_mode == 4 && symbols == 0u) fs[j].multiplier = 0.0;
                        multiplier[j] = fs[j].multiplier;
                    }
                    sum += multiplier[j] * weight[j];
                }
                double q = sigmoid(b + scale * sum);
                baseline_bits += loss_bits(outcome, q0);
                modeled_bits += loss_bits(outcome, q);
                for (j = 0; j < FAMILIES; ++j) {
                    if (!(family_mask & (UINT64_C(1) << j))) continue;
                    double local_q;
                    if (update_mode == 0) local_q = q;
                    else if (update_mode == 1) local_q = q0;
                    else if (update_mode == 2) local_q = sigmoid(b + weight[j]);
                    else if (update_mode == 3) local_q = sigmoid(b + scale * weight[j]);
                    else local_q = sigmoid(b + scale * multiplier[j] * weight[j]);
                    double feature = scale * weight[j];
                    fs[j].gradient[id[j]] += (float)(local_q - outcome);
                    fs[j].curvature[id[j]] += (float)(local_q * (1.0 - local_q));
                    double meta_gradient = (q - outcome) * feature;
                    double meta_curvature = q * (1.0 - q) * feature * feature;
                    fs[j].meta_gradient += meta_gradient;
                    fs[j].meta_curvature += meta_curvature;
                    if (meta_mode == 3 || meta_mode == 4) {
                        fs[j].multiplier = clamp(
                            fs[j].multiplier - meta_gradient /
                                (fs[j].meta_curvature + meta_lambda),
                            -meta_cap, meta_cap
                        );
                    }
                }
                if (outcome) {
                    prefix++;
                    if (cell_prefix[cell] < 255u) cell_prefix[cell]++;
                    run = 0u; defects++;
                } else if (run != UINT32_MAX) run++;
                sequence = (uint8_t)((sequence << 1) | outcome);
                symbols++;
            }
        }
        if (frame == 0 || (frame + 1) % 50 == 0)
            fprintf(stderr, "frame %d gain %.6f bits\n", frame + 1, baseline_bits - modeled_bits);
    }
    printf("{\n");
    printf("  \"frames\": %d,\n", frames);
    printf("  \"cross_mode\": %d,\n", cross_mode);
    printf("  \"variant_mask\": %u,\n", variant_mask);
    printf("  \"causal_mode\": %d,\n", causal_mode);
    printf("  \"confidence_mode\": %d,\n", confidence_mode);
    printf("  \"update_mode\": %d,\n", update_mode);
    printf("  \"scale\": %.12g,\n", scale);
    printf("  \"local_lambda\": %.12g,\n", local_lambda);
    printf("  \"local_cap\": %.12g,\n", local_cap);
    printf("  \"meta_lambda\": %.12g,\n", meta_lambda);
    printf("  \"meta_cap\": %.12g,\n", meta_cap);
    printf("  \"confidence_multiplier\": %.12g,\n", confidence_multiplier);
    printf("  \"local_word_mode\": %d,\n", local_word_mode);
    printf("  \"cross_word_mode\": %d,\n", cross_word_mode);
    printf("  \"temporal_word_mode\": %d,\n", temporal_word_mode);
    printf("  \"causal_word_mode\": %d,\n", causal_word_mode);
    printf("  \"meta_mode\": %d,\n", meta_mode);
    printf("  \"family_mask\": \"0x%014llx\",\n", (unsigned long long)family_mask);
    printf("  \"sync_mode\": %d,\n", sync_mode);
    printf("  \"geometry_mode\": %d,\n", geometry_mode);
    printf("  \"class_mode\": %d,\n", class_mode);
    printf("  \"symbols\": %llu,\n", (unsigned long long)symbols);
    printf("  \"defects\": %llu,\n", (unsigned long long)defects);
    printf("  \"sectors\": %llu,\n", (unsigned long long)sectors);
    printf("  \"adaptive_state_bytes\": %llu,\n", (unsigned long long)(sectors * 8u));
    printf("  \"baseline_bits\": %.12f,\n", baseline_bits);
    printf("  \"modeled_bits\": %.12f,\n", modeled_bits);
    printf("  \"gain_bits\": %.12f\n", baseline_bits - modeled_bits);
    printf("}\n");
    return 0;
}
