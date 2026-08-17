#define main opal_offline_evaluator_main
#include "opal_model_impl.c"
#undef main

/*
 * Production OPAL arithmetic backend.
 *
 * The inherited five-class law is split by its rank-one maximal projector.
 * The online state changes only the total mass transported between that
 * projector and its complement.  Relative mass inside the complement is
 * preserved exactly.
 */

#define RC64_ALPHABET 5u
#define RC64_TOTAL ((uint64_t)1u << 31)
#define RC64_TOP (((uint64_t)1u << 63) - 1u)
#define RC64_FIRST_QTR ((uint64_t)1u << 61)
#define RC64_HALF ((uint64_t)1u << 62)
#define RC64_THIRD_QTR (RC64_FIRST_QTR * 3u)
#define OPAL_HISTORY 9

typedef struct {
    family fs[FAMILIES];
    uint8_t *defects;
    int starts[GROUPS + 1];
    int *ordered;
    int frame;
    int group;
} opal_model;

typedef struct {
    uint64_t low, high, code;
    const uint8_t *data;
    size_t size, bit_position;
    int error;
    opal_model model;
} rc64_decoder;

typedef struct {
    uint64_t low, high, pending;
    uint8_t *data;
    size_t capacity, bit_position;
    int error, finished;
    opal_model model;
} rc64_encoder;

static unsigned opal_defect_at(const opal_model *model, int frame, int y, int x) {
    if (frame < 0 || frame > model->frame || y < 0 || y >= H || x < 0 || x >= W) return 0u;
    return model->defects[((size_t)(frame % OPAL_HISTORY) * HW) + (size_t)y * W + x] != 0u;
}

static unsigned opal_same_cell_at(
    const opal_model *model, int frame, int y, int x, int cy, int cx
) {
    if (y < 0 || y >= H || x < 0 || x >= W || cell_of(y, x) != cell_of(cy, cx)) return 0u;
    return opal_defect_at(model, frame, y, x);
}

static unsigned opal_causal_at(
    const opal_model *model, int frame, int y, int x, int group, int position
) {
    int neighbour_group, neighbour_position;
    if (frame < 0 || y < 0 || y >= H || x < 0 || x >= W) return 0u;
    neighbour_group = group_of(y, x);
    neighbour_position = y * W + x;
    if (neighbour_group > group ||
        (neighbour_group == group && neighbour_position >= position)) return 0u;
    return opal_defect_at(model, frame, y, x);
}

static int opal_model_init(opal_model *model) {
    int cursor[GROUPS], y, x, j;
    memset(model, 0, sizeof(*model));
    init_word_orbits();
    if (init_catalogue(model->fs) != 6175440u) return -1;
    model->defects = (uint8_t *)calloc((size_t)OPAL_HISTORY * HW, 1u);
    model->ordered = (int *)malloc((size_t)HW * sizeof(int));
    if (!model->defects || !model->ordered) return -1;
    for (y = 0; y < H; ++y) for (x = 0; x < W; ++x)
        model->starts[1 + group_of(y, x)]++;
    for (j = 1; j <= GROUPS; ++j) model->starts[j] += model->starts[j - 1];
    memcpy(cursor, model->starts, sizeof(cursor));
    for (y = 0; y < H; ++y) for (x = 0; x < W; ++x) {
        int group = group_of(y, x);
        model->ordered[cursor[group]++] = y * W + x;
    }
    return 0;
}

static void opal_model_destroy(opal_model *model) {
    int j;
    for (j = 0; j < FAMILIES; ++j) {
        free(model->fs[j].gradient);
        free(model->fs[j].curvature);
    }
    free(model->defects);
    free(model->ordered);
    memset(model, 0, sizeof(*model));
}

static void opal_cross_neighbourhood(
    const opal_model *model, int frame, int y, int x,
    unsigned *word, unsigned *count
) {
    static const int dy[8] = {-1,-1,-1,0,0,1,1,1};
    static const int dx[8] = {-1,0,1,-1,1,-1,0,1};
    int z;
    *word = 0u; *count = 0u;
    for (z = 0; z < 8; ++z) {
        int ny = y + dy[z], nx = x + dx[z];
        unsigned value = 0u;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && cell_of(ny, nx) != cell_of(y, x))
            value = opal_defect_at(model, frame - 1, ny, nx);
        *word |= value << z;
        *count += value;
    }
}

static double opal_prepare(
    opal_model *model, const float row[RC64_ALPHABET], int position,
    uint32_t id[FAMILIES], unsigned *maximal
) {
    static const int neighbour_dy[8] = {-1,-1,-1,0,0,1,1,1};
    static const int neighbour_dx[8] = {-1,0,1,-1,1,-1,0,1};
    static const int causal_dy[8] = {0,-1,-1,0,-1,1,-1,-1};
    static const int causal_dx[8] = {-1,2,-1,-2,-2,-4,0,1};
    const unsigned variant_mask = 251u;
    int frame = model->frame, group = model->group;
    int y = position / W, x = position % W, cell = cell_of(y, x);
    unsigned local_word = 0u, local_count;
    unsigned cross_word = 0u, cross_count = 0u;
    unsigned temporal_word = 0u, temporal_count = 0u;
    unsigned causal_word = 0u, causal_count = 0u, causal_prev_word = 0u;
    unsigned maximal_class = 0u, symbol, z;
    double wrong_mass = 0.0, b, q0, sum = 0.0;
    int conf, k = 0, j;

    for (symbol = 1u; symbol < RC64_ALPHABET; ++symbol)
        if (row[symbol] > row[maximal_class]) maximal_class = symbol;
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol)
        if (symbol != maximal_class) wrong_mass += (double)row[symbol];
    wrong_mass = clamp(wrong_mass, exp(-30.0), 1.0 - exp(-30.0));
    b = (double)(float)log(wrong_mass / (1.0 - wrong_mass));
    q0 = sigmoid(b);
    conf = (int)floor(875.0 * q0 + 0.5);
    if (conf < 0) conf = 0; else if (conf >= CBINS) conf = CBINS - 1;
    *maximal = maximal_class;

    local_count = opal_defect_at(model, frame - 1, y, x);
    for (z = 0u; z < 8u; ++z) {
        unsigned value = opal_same_cell_at(
            model, frame - 1, y + neighbour_dy[z], x + neighbour_dx[z], y, x
        );
        local_word |= value << z;
        local_count += value;
    }
    opal_cross_neighbourhood(model, frame, y, x, &cross_word, &cross_count);
    unsigned lag12 = opal_defect_at(model, frame - 1, y, x) |
                     (opal_defect_at(model, frame - 2, y, x) << 1);
    unsigned lag124 = lag12 | (opal_defect_at(model, frame - 4, y, x) << 2);
    for (z = 1u; z <= 8u; ++z) {
        unsigned value = opal_defect_at(model, frame - (int)z, y, x);
        temporal_word |= value << (z - 1u);
        temporal_count += value;
    }
    for (z = 0u; z < 8u; ++z) {
        int cy = y + causal_dy[z], cx = x + causal_dx[z];
        unsigned value = opal_causal_at(model, frame, cy, cx, group, position);
        unsigned previous_value = opal_causal_at(model, frame - 1, cy, cx, group, position);
        causal_word |= value << z;
        causal_count += value;
        causal_prev_word |= previous_value << z;
    }
    temporal_word = word_feature(temporal_word, 2, 0);
    unsigned causal_holonomy_word = causal_word ^ causal_prev_word;
    unsigned causal_holonomy_count = (unsigned)__builtin_popcount(causal_holonomy_word);
    unsigned local8 = ((unsigned)(y & 63) >> 3) * 8u + ((unsigned)(x & 63) >> 3);
    unsigned local4 = ((unsigned)(y & 63) >> 2) * 16u + ((unsigned)(x & 63) >> 2);
    unsigned local16 = ((unsigned)(y & 63) >> 4) * 4u + ((unsigned)(x & 63) >> 4);

#define ID(value) id[k++] = (uint32_t)(value)
    ID(conf); ID(group); ID(cell); ID(conf * GROUPS + group);
    ID(conf * CELLS + cell); ID(cell * GROUPS + group);
    ID((variant_mask & 1u) ? (uint32_t)((conf * CELLS + cell) * 64) + local8
                           : (uint32_t)((conf * CELLS + cell) * GROUPS + group));
    ID(opal_defect_at(model, frame - 1, y, x));
    unsigned prev = id[7];
    ID(conf * 2 + prev); ID(group * 2 + prev);
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

    /* Maximal-projector identity refines weak central sectors. */
    id[1] = maximal_class;
    id[3] = (uint32_t)(conf * 5 + maximal_class);
    id[4] = (uint32_t)(cell * 5 + maximal_class);
    id[5] = (uint32_t)(group * 5 + maximal_class);
    id[16] = (uint32_t)(maximal_class * 16 + local16);
    id[17] = (uint32_t)((conf * 5 + maximal_class) * 16 + local16);
    id[18] = (uint32_t)((group * 5 + maximal_class) * 16 + local16);
    id[25] = (uint32_t)((conf * 5 + maximal_class) * 4 + lag12);
    id[26] = (uint32_t)((group * 5 + maximal_class) * 4 + lag12);
    id[30] = (uint32_t)(cell * 5 + maximal_class);
    id[34] = (uint32_t)(group * 5 + maximal_class);
    id[38] = (uint32_t)(conf * 5 + maximal_class);
    id[39] = (uint32_t)((conf * 5 + maximal_class) * CELLS + cell);
    id[40] = (uint32_t)(cell * 5 + maximal_class);
    id[41] = (uint32_t)((conf * 5 + maximal_class) * 8 + lag124);
    id[42] = (uint32_t)((group * 5 + maximal_class) * 8 + lag124);

    /* Symbol-synchronous causal word and spacetime holonomy sectors. */
    ID(causal_word);
    ID(conf * 256 + causal_word);
    ID(conf * 256 + causal_holonomy_word);
    ID(cell * 256 + causal_word);
    ID((conf * CELLS + cell) * 256 + causal_word);
    ID(causal_holonomy_count);
    ID((cell * 8 + lag124) * 256 + causal_word);
    ID(lag124 * 256 + causal_word);
    ID((cell * 64 + local8) * 9 + causal_holonomy_count);
    ID(lag124 * 256 + causal_holonomy_word);
    ID(local8 * 256 + causal_word);
#undef ID
    if (k != FAMILIES) abort();

    for (j = 0; j < FAMILIES; ++j) {
        double weight = -(double)model->fs[j].gradient[id[j]] /
                        ((double)model->fs[j].curvature[id[j]] + 2.5);
        weight = clamp(weight, -4.0, 4.0);
        sum += model->fs[j].multiplier * weight;
    }
    return sigmoid(b + 0.22 * sum);
}

static void opal_update(
    opal_model *model, const uint32_t id[FAMILIES], double q,
    unsigned outcome, int position
) {
    int j;
    for (j = 0; j < FAMILIES; ++j) {
        family *f = model->fs + j;
        double weight = -(double)f->gradient[id[j]] /
                        ((double)f->curvature[id[j]] + 2.5);
        double feature, gradient, curvature;
        weight = clamp(weight, -4.0, 4.0);
        feature = 0.22 * weight;
        f->gradient[id[j]] += (float)(q - outcome);
        f->curvature[id[j]] += (float)(q * (1.0 - q));
        gradient = (q - outcome) * feature;
        curvature = q * (1.0 - q) * feature * feature;
        f->meta_gradient += gradient;
        f->meta_curvature += curvature;
        f->multiplier = clamp(
            f->multiplier - gradient / (f->meta_curvature + 0.75), -8.0, 8.0
        );
    }
    model->defects[(size_t)(model->frame % OPAL_HISTORY) * HW + position] = (uint8_t)outcome;
}

static int opal_adjust_frequencies(
    const float row[RC64_ALPHABET], double q, unsigned maximal,
    uint32_t frequencies[RC64_ALPHABET]
) {
    double wrong_mass = 0.0, adjusted[RC64_ALPHABET], probability_sum = 0.0;
    uint64_t frequency_sum = 0u;
    unsigned symbol, winner = 0u;
    int64_t balance, balanced;
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol)
        if (symbol != maximal) wrong_mass += (double)row[symbol];
    if (!(wrong_mass > 0.0 && wrong_mass < 1.0)) return -1;
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
        double value = symbol == maximal ? 1.0 - q : q * (double)row[symbol] / wrong_mass;
        uint64_t frequency;
        if (!isfinite(value) || value <= 0.0) return -2;
        adjusted[symbol] = value;
        probability_sum += value;
        if (adjusted[symbol] > adjusted[winner]) winner = symbol;
        frequency = (uint64_t)(value * (double)RC64_TOTAL);
        if (frequency < 1u) frequency = 1u;
        frequencies[symbol] = (uint32_t)frequency;
        frequency_sum += frequency;
    }
    if (probability_sum < 0.99998 || probability_sum > 1.00002) return -3;
    balance = (int64_t)RC64_TOTAL - (int64_t)frequency_sum;
    balanced = (int64_t)frequencies[winner] + balance;
    if (balanced <= 0 || balanced >= (int64_t)RC64_TOTAL) return -4;
    frequencies[winner] = (uint32_t)balanced;
    return 0;
}

static void opal_advance_group(opal_model *model) {
    model->group++;
    if (model->group == GROUPS) {
        model->group = 0;
        model->frame++;
        memset(model->defects + (size_t)(model->frame % OPAL_HISTORY) * HW, 0, HW);
    }
}

static unsigned rc64_read_bit(rc64_decoder *decoder) {
    size_t byte_index = decoder->bit_position >> 3u;
    unsigned bit_index = (unsigned)(decoder->bit_position & 7u), bit = 0u;
    if (byte_index < decoder->size)
        bit = (unsigned)((decoder->data[byte_index] >> (7u - bit_index)) & 1u);
    decoder->bit_position++;
    return bit;
}

static int rc64_decode_row(rc64_decoder *decoder, const uint32_t frequencies[5], int32_t *output) {
    uint64_t width = decoder->high - decoder->low + 1u;
    uint64_t scaled = (uint64_t)((((__uint128_t)(decoder->code - decoder->low + 1u) *
                                  RC64_TOTAL) - 1u) / width);
    uint64_t cumulative_low = 0u, cumulative_high = 0u, lower_offset, upper_offset;
    unsigned symbol;
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
        cumulative_high += frequencies[symbol];
        if (scaled < cumulative_high) break;
        cumulative_low = cumulative_high;
    }
    if (symbol == RC64_ALPHABET) return -1;
    lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
    upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
    if (upper_offset <= lower_offset) return -2;
    decoder->high = decoder->low + upper_offset - 1u;
    decoder->low += lower_offset;
    for (;;) {
        if (decoder->high < RC64_HALF) {
        } else if (decoder->low >= RC64_HALF) {
            decoder->code -= RC64_HALF; decoder->low -= RC64_HALF; decoder->high -= RC64_HALF;
        } else if (decoder->low >= RC64_FIRST_QTR && decoder->high < RC64_THIRD_QTR) {
            decoder->code -= RC64_FIRST_QTR;
            decoder->low -= RC64_FIRST_QTR; decoder->high -= RC64_FIRST_QTR;
        } else break;
        decoder->low <<= 1u;
        decoder->high = (decoder->high << 1u) | 1u;
        decoder->code = (decoder->code << 1u) | rc64_read_bit(decoder);
    }
    *output = (int32_t)symbol;
    return 0;
}

void *rc64_decoder_create(const uint8_t *data, size_t size) {
    rc64_decoder *decoder;
    unsigned bit;
    if (!data || !size) return NULL;
    decoder = (rc64_decoder *)calloc(1u, sizeof(*decoder));
    if (!decoder || opal_model_init(&decoder->model)) { free(decoder); return NULL; }
    decoder->data = data; decoder->size = size; decoder->high = RC64_TOP;
    for (bit = 0u; bit < 63u; ++bit)
        decoder->code = (decoder->code << 1u) | rc64_read_bit(decoder);
    return decoder;
}

void rc64_decoder_destroy(void *opaque) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    if (decoder) opal_model_destroy(&decoder->model);
    free(decoder);
}

int rc64_decoder_decode_probabilities(
    void *opaque, const float *probabilities, size_t count, int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    opal_model *model;
    size_t index;
    if (!decoder || (!symbols && count) || (!probabilities && count)) return -1;
    if (decoder->error) return -2;
    model = &decoder->model;
    if (model->frame >= 600 || count != (size_t)(model->starts[model->group + 1] - model->starts[model->group]))
        return -3;
    for (index = 0u; index < count; ++index) {
        int position = model->ordered[model->starts[model->group] + (int)index];
        uint32_t id[FAMILIES], frequencies[RC64_ALPHABET];
        unsigned maximal, outcome;
        double q = opal_prepare(model, probabilities + index * 5u, position, id, &maximal);
        if (opal_adjust_frequencies(probabilities + index * 5u, q, maximal, frequencies) ||
            rc64_decode_row(decoder, frequencies, symbols + index)) {
            decoder->error = 1; return -4;
        }
        outcome = (unsigned)symbols[index] != maximal;
        opal_update(model, id, q, outcome, position);
    }
    opal_advance_group(model);
    return 0;
}

size_t rc64_decoder_bit_position(const void *opaque) {
    const rc64_decoder *decoder = (const rc64_decoder *)opaque;
    return decoder ? decoder->bit_position : 0u;
}

uint64_t rc64_total_frequency(void) { return RC64_TOTAL; }

static int rc64_write_bit(rc64_encoder *encoder, unsigned bit) {
    size_t byte_index = encoder->bit_position >> 3u;
    if (byte_index >= encoder->capacity) {
        size_t capacity = encoder->capacity ? encoder->capacity * 2u : 16384u;
        uint8_t *data = (uint8_t *)realloc(encoder->data, capacity);
        if (!data) return -1;
        memset(data + encoder->capacity, 0, capacity - encoder->capacity);
        encoder->data = data; encoder->capacity = capacity;
    }
    if (bit) encoder->data[byte_index] |= (uint8_t)(1u << (7u - (encoder->bit_position & 7u)));
    encoder->bit_position++;
    return 0;
}

static int rc64_write_decision(rc64_encoder *encoder, unsigned bit) {
    if (rc64_write_bit(encoder, bit)) return -1;
    while (encoder->pending) {
        if (rc64_write_bit(encoder, bit ^ 1u)) return -1;
        encoder->pending--;
    }
    return 0;
}

static int rc64_encode_row(rc64_encoder *encoder, const uint32_t frequencies[5], unsigned symbol) {
    uint64_t width = encoder->high - encoder->low + 1u;
    uint64_t cumulative_low = 0u, cumulative_high = 0u;
    unsigned index;
    for (index = 0u; index < symbol; ++index) cumulative_low += frequencies[index];
    cumulative_high = cumulative_low + frequencies[symbol];
    uint64_t lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
    uint64_t upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
    if (upper_offset <= lower_offset) return -1;
    encoder->high = encoder->low + upper_offset - 1u;
    encoder->low += lower_offset;
    for (;;) {
        if (encoder->high < RC64_HALF) {
            if (rc64_write_decision(encoder, 0u)) return -1;
        } else if (encoder->low >= RC64_HALF) {
            if (rc64_write_decision(encoder, 1u)) return -1;
            encoder->low -= RC64_HALF; encoder->high -= RC64_HALF;
        } else if (encoder->low >= RC64_FIRST_QTR && encoder->high < RC64_THIRD_QTR) {
            encoder->pending++;
            encoder->low -= RC64_FIRST_QTR; encoder->high -= RC64_FIRST_QTR;
        } else break;
        encoder->low <<= 1u;
        encoder->high = (encoder->high << 1u) | 1u;
    }
    return 0;
}

void *rc64_encoder_create(void) {
    rc64_encoder *encoder = (rc64_encoder *)calloc(1u, sizeof(*encoder));
    if (!encoder || opal_model_init(&encoder->model)) { free(encoder); return NULL; }
    encoder->high = RC64_TOP;
    return encoder;
}

void rc64_encoder_destroy(void *opaque) {
    rc64_encoder *encoder = (rc64_encoder *)opaque;
    if (encoder) opal_model_destroy(&encoder->model);
    if (encoder) free(encoder->data);
    free(encoder);
}

int rc64_encoder_encode_probabilities(
    void *opaque, const float *probabilities, const int32_t *symbols, size_t count
) {
    rc64_encoder *encoder = (rc64_encoder *)opaque;
    opal_model *model;
    size_t index;
    if (!encoder || encoder->finished || encoder->error) return -1;
    model = &encoder->model;
    if (model->frame >= 600 || count != (size_t)(model->starts[model->group + 1] - model->starts[model->group]))
        return -2;
    for (index = 0u; index < count; ++index) {
        int position = model->ordered[model->starts[model->group] + (int)index];
        uint32_t id[FAMILIES], frequencies[RC64_ALPHABET];
        unsigned maximal, symbol = (unsigned)symbols[index], outcome;
        double q;
        if (symbol >= RC64_ALPHABET) return -3;
        q = opal_prepare(model, probabilities + index * 5u, position, id, &maximal);
        if (opal_adjust_frequencies(probabilities + index * 5u, q, maximal, frequencies) ||
            rc64_encode_row(encoder, frequencies, symbol)) {
            encoder->error = 1; return -4;
        }
        outcome = symbol != maximal;
        opal_update(model, id, q, outcome, position);
    }
    opal_advance_group(model);
    return 0;
}

int rc64_encoder_finish(void *opaque) {
    rc64_encoder *encoder = (rc64_encoder *)opaque;
    if (!encoder || encoder->error) return -1;
    if (encoder->finished) return 0;
    encoder->pending++;
    if (encoder->low < RC64_FIRST_QTR) {
        if (rc64_write_decision(encoder, 0u)) return -1;
    } else if (rc64_write_decision(encoder, 1u)) return -1;
    while (encoder->bit_position & 7u) if (rc64_write_bit(encoder, 0u)) return -1;
    encoder->finished = 1;
    return 0;
}

const uint8_t *rc64_encoder_data(const void *opaque) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    return encoder ? encoder->data : NULL;
}

size_t rc64_encoder_size(const void *opaque) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    return encoder ? (encoder->bit_position + 7u) / 8u : 0u;
}

size_t rc64_encoder_bit_position(const void *opaque) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    return encoder ? encoder->bit_position : 0u;
}
