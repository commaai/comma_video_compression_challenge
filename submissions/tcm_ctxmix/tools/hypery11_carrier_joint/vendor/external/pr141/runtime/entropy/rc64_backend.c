#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* Fixed five-symbol, 63-bit arithmetic decoder used by F26. */

#define RC64_ALPHABET 5u
#define RC64_TOTAL ((uint64_t)1u << 31)
#define RC64_TOP (((uint64_t)1u << 63) - 1u)
#define RC64_FIRST_QTR ((uint64_t)1u << 61)
#define RC64_HALF ((uint64_t)1u << 62)
#define RC64_THIRD_QTR (RC64_FIRST_QTR * 3u)

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    const uint8_t *data;
    size_t size;
    size_t bit_position;
    int error;
} rc64_decoder;

static unsigned rc64_read_bit(rc64_decoder *decoder) {
    size_t byte_index = decoder->bit_position >> 3u;
    unsigned bit_index = (unsigned)(decoder->bit_position & 7u);
    unsigned bit = 0u;
    if (byte_index < decoder->size) {
        bit = (unsigned)((decoder->data[byte_index] >> (7u - bit_index)) & 1u);
    }
    decoder->bit_position++;
    return bit;
}

void *rc64_decoder_create(const uint8_t *data, size_t size) {
    rc64_decoder *decoder;
    unsigned bit;
    if (!data || !size) return NULL;
    decoder = (rc64_decoder *)calloc(1u, sizeof(rc64_decoder));
    if (!decoder) return NULL;
    decoder->data = data;
    decoder->size = size;
    decoder->high = RC64_TOP;
    for (bit = 0u; bit < 63u; ++bit) {
        decoder->code = (decoder->code << 1u) | rc64_read_bit(decoder);
    }
    return decoder;
}

void rc64_decoder_destroy(void *opaque) {
    free(opaque);
}

static int rc64_decode_row(
    rc64_decoder *decoder,
    const uint32_t *frequencies,
    int32_t *output
) {
    uint64_t total = 0u;
    uint64_t width = decoder->high - decoder->low + 1u;
    uint64_t scaled;
    uint64_t cumulative_low = 0u;
    uint64_t cumulative_high = 0u;
    uint64_t lower_offset;
    uint64_t upper_offset;
    unsigned symbol;

    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
        if (!frequencies[symbol]) return -1;
        total += frequencies[symbol];
    }
    if (
        total != RC64_TOTAL ||
        decoder->code < decoder->low ||
        decoder->code > decoder->high
    ) return -2;

    scaled = (uint64_t)(
        (((__uint128_t)(decoder->code - decoder->low + 1u) * RC64_TOTAL) - 1u) /
        width
    );
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
        cumulative_high += frequencies[symbol];
        if (scaled < cumulative_high) break;
        cumulative_low = cumulative_high;
    }
    if (symbol == RC64_ALPHABET) return -3;

    lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
    upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
    if (upper_offset <= lower_offset) return -4;
    decoder->high = decoder->low + upper_offset - 1u;
    decoder->low += lower_offset;

    for (;;) {
        if (decoder->high < RC64_HALF) {
            /* No offset. */
        } else if (decoder->low >= RC64_HALF) {
            decoder->code -= RC64_HALF;
            decoder->low -= RC64_HALF;
            decoder->high -= RC64_HALF;
        } else if (
            decoder->low >= RC64_FIRST_QTR &&
            decoder->high < RC64_THIRD_QTR
        ) {
            decoder->code -= RC64_FIRST_QTR;
            decoder->low -= RC64_FIRST_QTR;
            decoder->high -= RC64_FIRST_QTR;
        } else {
            break;
        }
        decoder->low <<= 1u;
        decoder->high = (decoder->high << 1u) | 1u;
        decoder->code = (decoder->code << 1u) | rc64_read_bit(decoder);
    }
    *output = (int32_t)symbol;
    return 0;
}

int rc64_decoder_decode_probabilities(
    void *opaque,
    const float *probabilities,
    size_t count,
    int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    size_t index;
    if (!decoder || (!symbols && count) || (!probabilities && count)) return -1;
    if (decoder->error) return -2;

    for (index = 0u; index < count; ++index) {
        const float *row = probabilities + index * RC64_ALPHABET;
        uint32_t frequencies[RC64_ALPHABET];
        uint64_t frequency_sum = 0u;
        double probability_sum = 0.0;
        unsigned winner = 0u;
        unsigned symbol;
        int64_t balance;
        int64_t adjusted;
        int status;

        for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
            double value = (double)row[symbol];
            uint64_t frequency;
            if (!isfinite(value) || value <= 0.0) return -3;
            probability_sum += value;
            if (row[symbol] > row[winner]) winner = symbol;
            if (value > 1.00002) return -4;
            frequency = (uint64_t)(value * (double)RC64_TOTAL);
            if (frequency < 1u) frequency = 1u;
            frequencies[symbol] = (uint32_t)frequency;
            frequency_sum += frequency;
        }
        if (probability_sum < 0.99998 || probability_sum > 1.00002) return -5;
        balance = (int64_t)RC64_TOTAL - (int64_t)frequency_sum;
        adjusted = (int64_t)frequencies[winner] + balance;
        if (adjusted <= 0 || adjusted >= (int64_t)RC64_TOTAL) return -6;
        frequencies[winner] = (uint32_t)adjusted;
        for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
            if (!frequencies[symbol] || frequencies[symbol] >= RC64_TOTAL) {
                return -6;
            }
        }
        status = rc64_decode_row(decoder, frequencies, symbols + index);
        if (status) return status - 6;
    }
    return 0;
}

size_t rc64_decoder_bit_position(const void *opaque) {
    const rc64_decoder *decoder = (const rc64_decoder *)opaque;
    return decoder ? decoder->bit_position : 0u;
}

uint64_t rc64_total_frequency(void) {
    return RC64_TOTAL;
}
