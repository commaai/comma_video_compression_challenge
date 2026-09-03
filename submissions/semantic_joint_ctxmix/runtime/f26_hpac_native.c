#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__ARM_NEON) && !defined(F26_FORCE_SCALAR)
#include <arm_neon.h>
#elif defined(__AVX2__) && !defined(F26_FORCE_SCALAR)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Generic native lowering of F26's integer sparse-HPAC -> probability -> RC64
 * receiver.  Learned and video-derived values are supplied by the archive via
 * the Python binding; this source contains only the fixed decoder algorithm.
 */

#define F26_ALPHABET 5
#define F26_TOTAL ((uint64_t)1u << 31)
#define F26_TOP (((uint64_t)1u << 63) - 1u)
#define F26_FIRST_QTR ((uint64_t)1u << 61)
#define F26_HALF ((uint64_t)1u << 62)
#define F26_THIRD_QTR (F26_FIRST_QTR * 3u)

static double f26_last_timing[5];

static double f26_now_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec stamp;
    timespec_get(&stamp, TIME_UTC);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1.0e-9;
#endif
}

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    const uint8_t *data;
    size_t size;
    size_t bit_position;
    int error;
    int16_t *hidden_workspace;
    int16_t *b1_workspace;
    int16_t *logits_workspace;
    int16_t *shift_workspace;
    int16_t *past_workspace;
    int16_t *scale_workspace;
    int16_t *spm_workspace;
    int16_t *spm_dw_workspace;
    int32_t *pool_workspace;
    int32_t *conv_state_workspace;
    uint8_t *decoded_workspace;
    size_t hidden_capacity;
    size_t b1_capacity;
    size_t logits_capacity;
    size_t shift_capacity;
    size_t past_capacity;
    size_t scale_capacity;
    size_t spm_capacity;
    size_t spm_dw_capacity;
    size_t pool_capacity;
    size_t conv_state_capacity;
    size_t decoded_capacity;
} f26_rc64_decoder;

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    uint64_t bit_position;
    int32_t error;
} f26_rc64_state;

typedef struct {
    int32_t height;
    int32_t width;
    int32_t patch;
    int32_t patch_rows;
    int32_t patch_cols;
    int32_t patch_count;
    int32_t channels;
    int32_t groups;
    int32_t a_taps;
    int32_t b1_taps;
    int32_t b2_taps;
    int32_t logit_precision;
    int32_t num_frames;
    int32_t frame_dim;
    int32_t has_frame_scale;
    int32_t has_spm;

    /* conv_a_weight is [7, a_taps, channels], channel-contiguous. */
    const int8_t *conv_a_weight;
    /* Exact [5, a_taps, channels] class-minus-class-zero deltas. */
    const int16_t *conv_a_delta;
    /* Exact class-zero plus coordinate accumulator per [patch position, channel]. */
    const int32_t *conv_a_initial;
    const int16_t *conv_a_bias;
    const int8_t *conv_a_exponent;
    const int8_t *b1_weight;
    const int16_t *b1_bias;
    const int8_t *b1_exponent;
    const int8_t *b2_weight;
    const int16_t *b2_bias;
    const int8_t *b2_exponent;
    const int8_t *head_weight;
    const int16_t *head_bias;
    const int8_t *head_exponent;
    const float *residual_table;

    const int8_t *frame_codes;
    const int8_t *frame_shift_weight;
    const int16_t *frame_shift_bias;
    const int8_t *frame_shift_exponent;
    const int8_t *frame_scale_weight;
    const int16_t *frame_scale_bias;
    const int8_t *frame_scale_exponent;
    /* conv_past_weight is [3, 3, 5, channels], channel-contiguous. */
    const int8_t *conv_past_weight;
    const int16_t *conv_past_bias;
    const int8_t *conv_past_exponent;
    /* spm_dw_weight is [3, 3, channels], channel-contiguous. */
    const int8_t *spm_dw_weight;
    const int16_t *spm_dw_bias;
    const int8_t *spm_dw_exponent;
    /* spm_pw_weight is [channels, channels], output-major. */
    const int8_t *spm_pw_weight;
    const int16_t *spm_pw_bias;
    const int8_t *spm_pw_exponent;

    const int16_t *a_offsets;
    const int32_t *group_h_offsets;
    const int32_t *group_b1_offsets;
    const int32_t *group_target_offsets;
    const int32_t *group_output_offsets;
    const int16_t *h_positions;
    const int16_t *b1_gather;
    const int16_t *b2_gather;
    const int16_t *target_positions;
    const int32_t *output_order;
    const int32_t *flat_positions;
} f26_hpac_model;

static unsigned f26_read_bit(f26_rc64_decoder *decoder) {
    size_t byte_index = decoder->bit_position >> 3u;
    unsigned bit_index = (unsigned)(decoder->bit_position & 7u);
    unsigned bit = 0u;
    if (byte_index < decoder->size) {
        bit = (unsigned)((decoder->data[byte_index] >> (7u - bit_index)) & 1u);
    }
    decoder->bit_position++;
    return bit;
}

void *f26_rc64_create(const uint8_t *data, size_t size) {
    f26_rc64_decoder *decoder;
    unsigned bit;
    if (!data || !size) return NULL;
    decoder = (f26_rc64_decoder *)calloc(1u, sizeof(f26_rc64_decoder));
    if (!decoder) return NULL;
    decoder->data = data;
    decoder->size = size;
    decoder->high = F26_TOP;
    for (bit = 0u; bit < 63u; ++bit) {
        decoder->code = (decoder->code << 1u) | f26_read_bit(decoder);
    }
    return decoder;
}

void f26_rc64_destroy(void *opaque) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque;
    if (!decoder) return;
    free(decoder->hidden_workspace);
    free(decoder->b1_workspace);
    free(decoder->logits_workspace);
    free(decoder->shift_workspace);
    free(decoder->past_workspace);
    free(decoder->scale_workspace);
    free(decoder->spm_workspace);
    free(decoder->spm_dw_workspace);
    free(decoder->pool_workspace);
    free(decoder->conv_state_workspace);
    free(decoder->decoded_workspace);
    free(decoder);
}

int f26_rc64_get_state(const void *opaque, f26_rc64_state *state) {
    const f26_rc64_decoder *decoder = (const f26_rc64_decoder *)opaque;
    if (!decoder || !state) return -1;
    state->low = decoder->low;
    state->high = decoder->high;
    state->code = decoder->code;
    state->bit_position = (uint64_t)decoder->bit_position;
    state->error = decoder->error;
    return 0;
}

int f26_rc64_set_state(void *opaque, const f26_rc64_state *state) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque;
    if (!decoder || !state || state->bit_position < 63u) return -1;
    decoder->low = state->low;
    decoder->high = state->high;
    decoder->code = state->code;
    decoder->bit_position = (size_t)state->bit_position;
    decoder->error = state->error;
    return 0;
}

size_t f26_rc64_bit_position(const void *opaque) {
    const f26_rc64_decoder *decoder = (const f26_rc64_decoder *)opaque;
    return decoder ? decoder->bit_position : 0u;
}

static int f26_decode_row(
    f26_rc64_decoder *decoder,
    const uint32_t *frequencies,
    uint8_t *output
) {
    uint64_t total = 0u;
    uint64_t width = decoder->high - decoder->low + 1u;
    uint64_t scaled;
    uint64_t cumulative_low = 0u;
    uint64_t cumulative_high = 0u;
    uint64_t lower_offset;
    uint64_t upper_offset;
    unsigned symbol;

    for (symbol = 0u; symbol < F26_ALPHABET; ++symbol) {
        if (!frequencies[symbol]) return -1;
        total += frequencies[symbol];
    }
    if (
        total != F26_TOTAL || decoder->code < decoder->low || decoder->code > decoder->high
    ) return -2;
    scaled = (uint64_t)(
        (((__uint128_t)(decoder->code - decoder->low + 1u) * F26_TOTAL) - 1u) / width
    );
    for (symbol = 0u; symbol < F26_ALPHABET; ++symbol) {
        cumulative_high += frequencies[symbol];
        if (scaled < cumulative_high) break;
        cumulative_low = cumulative_high;
    }
    if (symbol == F26_ALPHABET) return -3;
    lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
    upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
    if (upper_offset <= lower_offset) return -4;
    decoder->high = decoder->low + upper_offset - 1u;
    decoder->low += lower_offset;
    for (;;) {
        if (decoder->high < F26_HALF) {
            /* No offset. */
        } else if (decoder->low >= F26_HALF) {
            decoder->code -= F26_HALF;
            decoder->low -= F26_HALF;
            decoder->high -= F26_HALF;
        } else if (decoder->low >= F26_FIRST_QTR && decoder->high < F26_THIRD_QTR) {
            decoder->code -= F26_FIRST_QTR;
            decoder->low -= F26_FIRST_QTR;
            decoder->high -= F26_FIRST_QTR;
        } else {
            break;
        }
        decoder->low <<= 1u;
        decoder->high = (decoder->high << 1u) | 1u;
        decoder->code = (decoder->code << 1u) | f26_read_bit(decoder);
    }
    *output = (uint8_t)symbol;
    return 0;
}

static int32_t clamp_i32(int64_t value, int32_t low, int32_t high) {
    if (value < low) return low;
    if (value > high) return high;
    return (int32_t)value;
}

/* Round signed numerator / 2^shift to nearest, with ties to even. */
static int64_t round_div_pow2_even(int64_t numerator, unsigned shift) {
    uint64_t magnitude;
    uint64_t quotient;
    uint64_t remainder;
    uint64_t half;
    int negative;
    if (!shift) return numerator;
    negative = numerator < 0;
    magnitude = negative ? (uint64_t)(-numerator) : (uint64_t)numerator;
    quotient = magnitude >> shift;
    remainder = magnitude & (((uint64_t)1u << shift) - 1u);
    half = (uint64_t)1u << (shift - 1u);
    if (remainder > half || (remainder == half && (quotient & 1u))) quotient++;
    return negative ? -(int64_t)quotient : (int64_t)quotient;
}

static int32_t requantize_affine(
    int64_t sum,
    int16_t bias,
    int8_t exponent,
    unsigned shift,
    int32_t low,
    int32_t high
) {
    unsigned exponent_shift;
    int64_t numerator;
    if (exponent > 0 || exponent < -20) return low - 1;
    exponent_shift = (unsigned)(-exponent);
    numerator = sum + ((int64_t)bias << exponent_shift);
    return clamp_i32(
        round_div_pow2_even(numerator, shift + exponent_shift), low, high
    );
}

static void add_i8x64_to_i32(int32_t destination[64], const int8_t source[64]) {
    int channel;
#if defined(__ARM_NEON) && !defined(F26_FORCE_SCALAR)
    for (channel = 0; channel < 64; channel += 16) {
        int8x16_t packed = vld1q_s8(source + channel);
        int16x8_t low16 = vmovl_s8(vget_low_s8(packed));
        int16x8_t high16 = vmovl_s8(vget_high_s8(packed));
        int32x4_t values[4] = {
            vmovl_s16(vget_low_s16(low16)),
            vmovl_s16(vget_high_s16(low16)),
            vmovl_s16(vget_low_s16(high16)),
            vmovl_s16(vget_high_s16(high16)),
        };
        int lane;
        for (lane = 0; lane < 4; ++lane) {
            int32_t *target = destination + channel + lane * 4;
            vst1q_s32(target, vaddq_s32(vld1q_s32(target), values[lane]));
        }
    }
#elif defined(__AVX2__) && !defined(F26_FORCE_SCALAR)
    for (channel = 0; channel < 64; channel += 8) {
        __m128i packed = _mm_loadl_epi64((const __m128i *)(source + channel));
        __m256i values = _mm256_cvtepi8_epi32(packed);
        __m256i prior = _mm256_loadu_si256((const __m256i *)(destination + channel));
        _mm256_storeu_si256((__m256i *)(destination + channel), _mm256_add_epi32(prior, values));
    }
#else
    for (channel = 0; channel < 64; ++channel) {
        destination[channel] += (int32_t)source[channel];
    }
#endif
}

static int f26_prepare_frame_context(
    f26_rc64_decoder *decoder,
    const f26_hpac_model *model,
    int32_t frame_index,
    const uint8_t *previous
) {
    int16_t *shift = decoder->shift_workspace;
    int16_t *past = decoder->past_workspace;
    int16_t *scale = decoder->scale_workspace;
    int16_t *spm = decoder->spm_workspace;
    int16_t *spm_dw = decoder->spm_dw_workspace;
    int32_t *pooled = decoder->pool_workspace;
    int32_t patch_index;

    if (
        frame_index < 0 || frame_index >= model->num_frames || model->frame_dim <= 0 ||
        model->patch != 64 || model->channels != 64 || !previous || !model->frame_codes ||
        !model->frame_shift_weight || !model->frame_shift_bias ||
        !model->frame_shift_exponent || !model->conv_past_weight ||
        !model->conv_past_bias || !model->conv_past_exponent
    ) return -1;
    if (
        model->has_frame_scale &&
        (!model->frame_scale_weight || !model->frame_scale_bias || !model->frame_scale_exponent)
    ) return -2;
    if (
        model->has_spm &&
        (!model->spm_dw_weight || !model->spm_dw_bias || !model->spm_dw_exponent ||
         !model->spm_pw_weight || !model->spm_pw_bias || !model->spm_pw_exponent)
    ) return -3;

    {
        const int8_t *embedding = model->frame_codes +
            (size_t)frame_index * (size_t)model->frame_dim;
        int16_t frame_shift[64];
        int16_t frame_scale[64];
        int32_t channel;
        for (channel = 0; channel < model->channels; ++channel) {
            int64_t shift_sum = 0;
            int64_t scale_sum = 0;
            int32_t feature;
            for (feature = 0; feature < model->frame_dim; ++feature) {
                shift_sum += (int64_t)embedding[feature] *
                    model->frame_shift_weight[channel * model->frame_dim + feature];
                if (model->has_frame_scale) {
                    scale_sum += (int64_t)embedding[feature] *
                        model->frame_scale_weight[channel * model->frame_dim + feature];
                }
            }
            frame_shift[channel] = (int16_t)requantize_affine(
                shift_sum, model->frame_shift_bias[channel],
                model->frame_shift_exponent[channel], 1u, -127, 127
            );
            frame_scale[channel] = model->has_frame_scale
                ? (int16_t)requantize_affine(
                    scale_sum, model->frame_scale_bias[channel],
                    model->frame_scale_exponent[channel], 4u, -8, 8
                )
                : 0;
        }
        for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
            memcpy(
                shift + (size_t)patch_index * (size_t)model->channels,
                frame_shift,
                (size_t)model->channels * sizeof(int16_t)
            );
            memcpy(
                scale + (size_t)patch_index * (size_t)model->channels,
                frame_scale,
                (size_t)model->channels * sizeof(int16_t)
            );
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
    for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
        int32_t patch_row = patch_index / model->patch_cols;
        int32_t patch_col = patch_index % model->patch_cols;
        int32_t pool_sum[64] = {0};
        int32_t local_row;
        for (local_row = 0; local_row < model->patch; ++local_row) {
            int32_t local_col;
            int32_t global_row = patch_row * model->patch + local_row;
            for (local_col = 0; local_col < model->patch; ++local_col) {
                int32_t accum[64] = {0};
                int32_t global_col = patch_col * model->patch + local_col;
                int32_t kernel_row;
                size_t destination_offset =
                    (((size_t)patch_index * (size_t)model->patch + (size_t)local_row) *
                     (size_t)model->patch + (size_t)local_col) * (size_t)model->channels;
                for (kernel_row = 0; kernel_row < 3; ++kernel_row) {
                    int32_t source_row = global_row + kernel_row - 1;
                    int32_t kernel_col;
                    if (source_row < 0 || source_row >= model->height) continue;
                    for (kernel_col = 0; kernel_col < 3; ++kernel_col) {
                        int32_t source_col = global_col + kernel_col - 1;
                        uint8_t class_id;
                        const int8_t *weights;
                        if (source_col < 0 || source_col >= model->width) continue;
                        class_id = previous[
                            (size_t)source_row * (size_t)model->width + (size_t)source_col
                        ];
                        if (class_id >= F26_ALPHABET) continue;
                        weights = model->conv_past_weight +
                            ((((size_t)kernel_row * 3u + (size_t)kernel_col) * F26_ALPHABET +
                              (size_t)class_id) * (size_t)model->channels);
                        add_i8x64_to_i32(accum, weights);
                    }
                }
                {
                    int32_t channel;
                    for (channel = 0; channel < model->channels; ++channel) {
                        int32_t value = requantize_affine(
                            accum[channel], model->conv_past_bias[channel],
                            model->conv_past_exponent[channel], 0u, -127, 127
                        );
                        past[destination_offset + (size_t)channel] = (int16_t)value;
                        pool_sum[channel] += value;
                    }
                }
            }
        }
        {
            int32_t channel;
            for (channel = 0; channel < model->channels; ++channel) {
                pooled[(size_t)patch_index * (size_t)model->channels + (size_t)channel] =
                    (int32_t)round_div_pow2_even(pool_sum[channel], 12u);
            }
        }
    }

    if (!model->has_spm) {
        memset(spm, 0, (size_t)model->patch_count * (size_t)model->channels * sizeof(int16_t));
        return 0;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
    for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
        int32_t patch_row = patch_index / model->patch_cols;
        int32_t patch_col = patch_index % model->patch_cols;
        int32_t channel;
        for (channel = 0; channel < model->channels; ++channel) {
            int64_t sum = 0;
            int32_t kernel_row;
            for (kernel_row = 0; kernel_row < 3; ++kernel_row) {
                int32_t source_row = patch_row + kernel_row - 1;
                int32_t kernel_col;
                if (source_row < 0 || source_row >= model->patch_rows) continue;
                for (kernel_col = 0; kernel_col < 3; ++kernel_col) {
                    int32_t source_col = patch_col + kernel_col - 1;
                    int32_t source_patch;
                    if (source_col < 0 || source_col >= model->patch_cols) continue;
                    source_patch = source_row * model->patch_cols + source_col;
                    sum += (int64_t)pooled[
                        (size_t)source_patch * (size_t)model->channels + (size_t)channel
                    ] * model->spm_dw_weight[
                        ((size_t)kernel_row * 3u + (size_t)kernel_col) *
                        (size_t)model->channels + (size_t)channel
                    ];
                }
            }
            {
                int32_t value = requantize_affine(
                    sum, model->spm_dw_bias[channel], model->spm_dw_exponent[channel],
                    3u, -127, 127
                );
                if (value < 0) value = 0;
                spm_dw[(size_t)patch_index * (size_t)model->channels + (size_t)channel] =
                    (int16_t)value;
            }
        }
        for (channel = 0; channel < model->channels; ++channel) {
            int64_t sum = 0;
            int32_t input_channel;
            const int8_t *weights = model->spm_pw_weight +
                (size_t)channel * (size_t)model->channels;
            for (input_channel = 0; input_channel < model->channels; ++input_channel) {
                sum += (int64_t)spm_dw[
                    (size_t)patch_index * (size_t)model->channels + (size_t)input_channel
                ] * weights[input_channel];
            }
            spm[(size_t)patch_index * (size_t)model->channels + (size_t)channel] =
                (int16_t)requantize_affine(
                    sum, model->spm_pw_bias[channel], model->spm_pw_exponent[channel],
                    4u, -127, 127
                );
        }
    }
    return 0;
}

static int probability_and_frequencies(
    const float corrected[F26_ALPHABET],
    int precision,
    float probability[F26_ALPHABET],
    uint32_t frequency[F26_ALPHABET]
) {
    double values[F26_ALPHABET];
    double maximum;
    double sum = 0.0;
    uint64_t frequency_sum = 0u;
    int winner = 0;
    int symbol;
    int64_t balance;
    int64_t adjusted;

    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        float scaled = corrected[symbol] * (float)precision;
        long quantized = lrintf(scaled);
        if (quantized < -32768) quantized = -32768;
        if (quantized > 32767) quantized = 32767;
        values[symbol] = (double)((float)quantized / (float)precision);
    }
    maximum = values[0];
    for (symbol = 1; symbol < F26_ALPHABET; ++symbol) {
        if (values[symbol] > maximum) maximum = values[symbol];
    }
    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        values[symbol] = exp(values[symbol] - maximum);
        sum += values[symbol];
    }
    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        double value;
        uint64_t item;
        probability[symbol] = (float)(values[symbol] / sum);
        value = (double)probability[symbol];
        if (!isfinite(value) || value <= 0.0 || value > 1.00002) return -1;
        if (probability[symbol] > probability[winner]) winner = symbol;
        item = (uint64_t)(value * (double)F26_TOTAL);
        if (item < 1u) item = 1u;
        frequency[symbol] = (uint32_t)item;
        frequency_sum += item;
    }
    balance = (int64_t)F26_TOTAL - (int64_t)frequency_sum;
    adjusted = (int64_t)frequency[winner] + balance;
    if (adjusted <= 0 || adjusted >= (int64_t)F26_TOTAL) return -2;
    frequency[winner] = (uint32_t)adjusted;
    return 0;
}

int f26_hpac_decode_frame(
    void *opaque_decoder,
    const f26_hpac_model *model,
    const uint8_t *boundary,
    const uint8_t *previous,
    int32_t frame_index,
    uint8_t *current,
    float *corrected_trace,
    float *probability_trace
) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque_decoder;
    int16_t *hidden = NULL;
    int16_t *b1 = NULL;
    int16_t *logits = NULL;
    int16_t *shift = NULL;
    int16_t *past = NULL;
    int16_t *scale = NULL;
    int16_t *spm = NULL;
    int32_t *conv_state = NULL;
    uint8_t *decoded_patch_major = NULL;
    size_t hidden_capacity;
    size_t b1_capacity;
    size_t logits_capacity;
    size_t compact_context_capacity;
    size_t past_capacity;
    size_t conv_state_capacity;
    int32_t max_h = 0;
    int32_t max_b1 = 0;
    int32_t max_targets = 0;
    int32_t group;
    int status = 0;
    double timing_started;

    if (
        !decoder || !model || !boundary || !previous || !current || !corrected_trace ||
        !probability_trace
    ) return -1;
    if (
        model->channels != 64 || model->patch_count != model->patch_rows * model->patch_cols ||
        model->height != model->patch_rows * model->patch ||
        model->width != model->patch_cols * model->patch || model->groups <= 0 ||
        !model->conv_a_delta
    ) return -2;
    if (fesetround(FE_TONEAREST)) return -3;

    for (group = 0; group < model->groups; ++group) {
        int32_t h_count = model->group_h_offsets[group + 1] - model->group_h_offsets[group];
        int32_t b1_count = model->group_b1_offsets[group + 1] - model->group_b1_offsets[group];
        int32_t target_count = model->group_target_offsets[group + 1] - model->group_target_offsets[group];
        if (h_count > max_h) max_h = h_count;
        if (b1_count > max_b1) max_b1 = b1_count;
        if (target_count > max_targets) max_targets = target_count;
    }
    hidden_capacity = (size_t)model->patch_count * (size_t)(max_h + 1) * (size_t)model->channels;
    b1_capacity = (size_t)model->patch_count * (size_t)(max_b1 + 1) * (size_t)model->channels;
    logits_capacity = (size_t)model->patch_count * (size_t)max_targets * F26_ALPHABET;
    compact_context_capacity = (size_t)model->patch_count * (size_t)model->channels;
    past_capacity = (size_t)model->patch_count * (size_t)model->patch *
        (size_t)model->patch * (size_t)model->channels;
    conv_state_capacity = (size_t)model->patch_count * (size_t)model->patch *
        (size_t)model->patch * (size_t)model->channels;
    if (!decoder->hidden_workspace) {
        decoder->hidden_workspace = (int16_t *)malloc(hidden_capacity * sizeof(int16_t));
        decoder->b1_workspace = (int16_t *)malloc(b1_capacity * sizeof(int16_t));
        decoder->logits_workspace = (int16_t *)malloc(logits_capacity * sizeof(int16_t));
        decoder->shift_workspace = (int16_t *)malloc(compact_context_capacity * sizeof(int16_t));
        decoder->past_workspace = (int16_t *)malloc(past_capacity * sizeof(int16_t));
        decoder->scale_workspace = (int16_t *)malloc(compact_context_capacity * sizeof(int16_t));
        decoder->spm_workspace = (int16_t *)malloc(compact_context_capacity * sizeof(int16_t));
        decoder->spm_dw_workspace = (int16_t *)malloc(
            compact_context_capacity * sizeof(int16_t)
        );
        decoder->pool_workspace = (int32_t *)malloc(compact_context_capacity * sizeof(int32_t));
        decoder->conv_state_workspace = (int32_t *)malloc(conv_state_capacity * sizeof(int32_t));
        decoder->decoded_workspace = (uint8_t *)malloc(
            (size_t)model->patch_count * (size_t)max_targets
        );
        decoder->hidden_capacity = hidden_capacity;
        decoder->b1_capacity = b1_capacity;
        decoder->logits_capacity = logits_capacity;
        decoder->shift_capacity = compact_context_capacity;
        decoder->past_capacity = past_capacity;
        decoder->scale_capacity = compact_context_capacity;
        decoder->spm_capacity = compact_context_capacity;
        decoder->spm_dw_capacity = compact_context_capacity;
        decoder->pool_capacity = compact_context_capacity;
        decoder->conv_state_capacity = conv_state_capacity;
        decoder->decoded_capacity = (size_t)model->patch_count * (size_t)max_targets;
    }
    if (
        !decoder->hidden_workspace || !decoder->b1_workspace || !decoder->logits_workspace ||
        !decoder->shift_workspace || !decoder->past_workspace || !decoder->scale_workspace ||
        !decoder->spm_workspace || !decoder->spm_dw_workspace || !decoder->pool_workspace ||
        !decoder->conv_state_workspace || !decoder->decoded_workspace ||
        decoder->hidden_capacity < hidden_capacity || decoder->b1_capacity < b1_capacity ||
        decoder->logits_capacity < logits_capacity ||
        decoder->shift_capacity < compact_context_capacity ||
        decoder->past_capacity < past_capacity ||
        decoder->scale_capacity < compact_context_capacity ||
        decoder->spm_capacity < compact_context_capacity ||
        decoder->spm_dw_capacity < compact_context_capacity ||
        decoder->pool_capacity < compact_context_capacity ||
        decoder->conv_state_capacity < conv_state_capacity ||
        decoder->decoded_capacity < (size_t)model->patch_count * (size_t)max_targets
    ) {
        status = -4;
        goto cleanup;
    }
    hidden = decoder->hidden_workspace;
    b1 = decoder->b1_workspace;
    logits = decoder->logits_workspace;
    shift = decoder->shift_workspace;
    past = decoder->past_workspace;
    scale = decoder->scale_workspace;
    spm = decoder->spm_workspace;
    conv_state = decoder->conv_state_workspace;
    decoded_patch_major = decoder->decoded_workspace;

    memset(current, 0, (size_t)model->height * (size_t)model->width);
    memset(f26_last_timing, 0, sizeof(f26_last_timing));
    timing_started = f26_now_seconds();
    status = f26_prepare_frame_context(decoder, model, frame_index, previous);
    if (status) {
        status -= 40;
        goto cleanup;
    }
    f26_last_timing[0] = f26_now_seconds() - timing_started;
    timing_started = f26_now_seconds();
    for (group = 0; group < model->patch_count; ++group) {
        memcpy(
            conv_state + (size_t)group * (size_t)model->patch * (size_t)model->patch *
                (size_t)model->channels,
            model->conv_a_initial,
            (size_t)model->patch * (size_t)model->patch * (size_t)model->channels *
            sizeof(int32_t)
        );
    }
    f26_last_timing[1] = f26_now_seconds() - timing_started;
    for (group = 0; group < model->groups; ++group) {
        int32_t h_start = model->group_h_offsets[group];
        int32_t h_count = model->group_h_offsets[group + 1] - h_start;
        int32_t b1_start = model->group_b1_offsets[group];
        int32_t b1_count = model->group_b1_offsets[group + 1] - b1_start;
        int32_t target_start = model->group_target_offsets[group];
        int32_t target_count = model->group_target_offsets[group + 1] - target_start;
        int32_t output_start = model->group_output_offsets[group];
        int32_t output_count = model->group_output_offsets[group + 1] - output_start;
        int32_t patch_index;
        int32_t local_index;

        if (output_count != model->patch_count * target_count) {
            status = -5;
            goto cleanup;
        }
        timing_started = f26_now_seconds();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4) private(local_index)
#endif
        for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
            for (local_index = 0; local_index < h_count; ++local_index) {
                int32_t local_row = model->h_positions[(h_start + local_index) * 2];
                int32_t local_col = model->h_positions[(h_start + local_index) * 2 + 1];
                int32_t channel;
                const int32_t *accum = conv_state +
                    (((size_t)patch_index * (size_t)model->patch * (size_t)model->patch +
                      (size_t)local_row * (size_t)model->patch + (size_t)local_col) *
                     (size_t)model->channels);
                for (channel = 0; channel < model->channels; ++channel) {
                    int32_t value = requantize_affine(
                        accum[channel], model->conv_a_bias[channel],
                        model->conv_a_exponent[channel], 1u, -127, 127
                    );
                    int32_t context_offset =
                        (((patch_index * model->patch + local_row) * model->patch + local_col) *
                         model->channels + channel);
                    int32_t film_offset = patch_index * model->channels + channel;
                    int64_t scaled = (int64_t)value * (16 + (int32_t)scale[film_offset]);
                    value = clamp_i32(round_div_pow2_even(scaled, 4u), -127, 127);
                    {
                        int32_t past_value = (int32_t)past[context_offset];
                        past_value += (int32_t)spm[film_offset];
                        /* Match the reference's distinct past+SPM requantization. */
                        past_value = clamp_i32(past_value, -127, 127);
                        value += (int32_t)shift[film_offset];
                        value += past_value;
                    }
                    value = clamp_i32(value, -127, 127);
                    if (value < 0) value = 0;
                    hidden[
                        ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)local_index) *
                        (size_t)model->channels + (size_t)channel
                    ] = (int16_t)value;
                }
            }
            memset(
                hidden + ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)h_count) *
                    (size_t)model->channels,
                0,
                (size_t)model->channels * sizeof(int16_t)
            );

            for (local_index = 0; local_index < b1_count; ++local_index) {
                int64_t sums[64];
                int32_t channel;
                int32_t tap;
                for (channel = 0; channel < model->channels; ++channel) sums[channel] = 0;
                for (tap = 0; tap < model->b1_taps; ++tap) {
                    int32_t gather = model->b1_gather[
                        ((size_t)b1_start + (size_t)local_index) * (size_t)model->b1_taps +
                        (size_t)tap
                    ];
                    const int16_t *values = hidden +
                        ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)gather) *
                        (size_t)model->channels;
                    const int8_t *weights = model->b1_weight +
                        (size_t)tap * (size_t)model->channels;
                    for (channel = 0; channel < model->channels; ++channel) {
                        sums[channel] += (int64_t)values[channel] * weights[channel];
                    }
                }
                for (channel = 0; channel < model->channels; ++channel) {
                    {
                        int32_t value = requantize_affine(
                            sums[channel], model->b1_bias[channel], model->b1_exponent[channel],
                            3u, -127, 127
                        );
                        if (value < 0) value = 0;
                        b1[
                            ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)local_index) *
                            (size_t)model->channels + (size_t)channel
                        ] = (int16_t)value;
                    }
                }
            }
            memset(
                b1 + ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)b1_count) *
                    (size_t)model->channels,
                0,
                (size_t)model->channels * sizeof(int16_t)
            );

            for (local_index = 0; local_index < target_count; ++local_index) {
                int16_t b2_hidden[64];
                int64_t sums[64];
                int32_t channel;
                int32_t symbol;
                int32_t tap;
                for (channel = 0; channel < model->channels; ++channel) sums[channel] = 0;
                for (tap = 0; tap < model->b2_taps; ++tap) {
                    int32_t gather = model->b2_gather[
                        ((size_t)target_start + (size_t)local_index) * (size_t)model->b2_taps +
                        (size_t)tap
                    ];
                    const int16_t *values = b1 +
                        ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)gather) *
                        (size_t)model->channels;
                    const int8_t *weights = model->b2_weight +
                        (size_t)tap * (size_t)model->channels;
                    for (channel = 0; channel < model->channels; ++channel) {
                        sums[channel] += (int64_t)values[channel] * weights[channel];
                    }
                }
                for (channel = 0; channel < model->channels; ++channel) {
                    {
                        int32_t value = requantize_affine(
                            sums[channel], model->b2_bias[channel], model->b2_exponent[channel],
                            3u, -127, 127
                        );
                        if (value < 0) value = 0;
                        b2_hidden[channel] = (int16_t)value;
                    }
                }
                for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                    int64_t sum = 0;
                    for (channel = 0; channel < model->channels; ++channel) {
                        sum += (int64_t)b2_hidden[channel] *
                            model->head_weight[symbol * model->channels + channel];
                    }
                    logits[
                        ((size_t)patch_index * (size_t)target_count + (size_t)local_index) *
                        F26_ALPHABET + (size_t)symbol
                    ] = (int16_t)requantize_affine(
                        sum, model->head_bias[symbol], model->head_exponent[symbol],
                        3u, -32768, 32767
                    );
                }
            }
        }
        if (status) goto cleanup;
        f26_last_timing[2] += f26_now_seconds() - timing_started;

        timing_started = f26_now_seconds();
        for (local_index = 0; local_index < output_count; ++local_index) {
            int32_t patch_major = model->output_order[output_start + local_index];
            int32_t flat_position = model->flat_positions[output_start + local_index];
            const int16_t *row = logits + (size_t)patch_major * F26_ALPHABET;
            float corrected[F26_ALPHABET];
            float probability[F26_ALPHABET];
            uint32_t frequency[F26_ALPHABET];
            int32_t predicted = 0;
            int32_t symbol;
            uint8_t decoded;
            int32_t feature;
            for (symbol = 1; symbol < F26_ALPHABET; ++symbol) {
                if (row[symbol] > row[predicted]) predicted = symbol;
            }
            feature = (int32_t)boundary[flat_position] * F26_ALPHABET + predicted;
            if (feature < 0 || feature >= 25) {
                status = -7;
                goto cleanup;
            }
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                corrected[symbol] = (float)row[symbol] / 8.0f +
                    model->residual_table[feature * F26_ALPHABET + symbol];
                corrected_trace[((size_t)output_start + (size_t)local_index) * F26_ALPHABET + symbol] =
                    corrected[symbol];
            }
            status = probability_and_frequencies(
                corrected, model->logit_precision, probability, frequency
            );
            if (status) {
                status = -20 + status;
                goto cleanup;
            }
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                probability_trace[((size_t)output_start + (size_t)local_index) * F26_ALPHABET + symbol] =
                    probability[symbol];
            }
            status = f26_decode_row(decoder, frequency, &decoded);
            if (status) {
                status -= 30;
                goto cleanup;
            }
            current[flat_position] = decoded;
            decoded_patch_major[patch_major] = decoded;
        }
        f26_last_timing[3] += f26_now_seconds() - timing_started;
        /* Apply this group's decoded symbols to the incremental conv-A state. */
        timing_started = f26_now_seconds();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4) private(local_index)
#endif
        for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
            for (local_index = 0; local_index < target_count; ++local_index) {
                uint8_t class_id = decoded_patch_major[patch_index * target_count + local_index];
                int32_t source_row;
                int32_t source_col;
                int32_t tap;
                if (!class_id) continue;
                if (class_id >= F26_ALPHABET) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                    status = -8;
                    continue;
                }
                source_row = model->target_positions[(target_start + local_index) * 2];
                source_col = model->target_positions[(target_start + local_index) * 2 + 1];
                for (tap = 0; tap < model->a_taps; ++tap) {
                    int32_t hidden_row = source_row - model->a_offsets[tap * 2];
                    int32_t hidden_col = source_col - model->a_offsets[tap * 2 + 1];
                    int32_t channel;
                    int32_t *destination;
                    const int16_t *class_delta;
                    if (
                        hidden_row < 0 || hidden_row >= model->patch ||
                        hidden_col < 0 || hidden_col >= model->patch
                    ) continue;
                    destination = conv_state +
                        (((size_t)patch_index * (size_t)model->patch * (size_t)model->patch +
                          (size_t)hidden_row * (size_t)model->patch + (size_t)hidden_col) *
                         (size_t)model->channels);
                    class_delta = model->conv_a_delta +
                        ((size_t)class_id * (size_t)model->a_taps + (size_t)tap) *
                        (size_t)model->channels;
                    for (channel = 0; channel < model->channels; ++channel) {
                        destination[channel] += (int32_t)class_delta[channel];
                    }
                }
            }
        }
        if (status) goto cleanup;
        f26_last_timing[4] += f26_now_seconds() - timing_started;
    }

cleanup:
    return status;
}

uint64_t f26_total_frequency(void) {
    return F26_TOTAL;
}

void f26_hpac_last_timing(double output[5]) {
    if (output) memcpy(output, f26_last_timing, sizeof(f26_last_timing));
}
