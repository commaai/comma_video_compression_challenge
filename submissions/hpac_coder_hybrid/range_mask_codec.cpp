// Static arithmetic coder audit for exact qpose/Quantizr masks.
//
// This is not an inflate-ready submission codec yet. It verifies whether the
// p6 raster context entropy oracle can be approached by an actual coder.

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint8_t SENTINEL = 5;
constexpr int CTX_COUNT = 216;
constexpr int CTX5_COUNT = 7776;
constexpr int CTX6_COUNT = 46656;
constexpr int CTX7_COUNT = 279936;
constexpr int CTX8_COUNT = 1679616;
constexpr int CTX9_COUNT = 10077696;
constexpr int EVENT_SYMS = 4;
constexpr int CLASS_SYMS = 5;
constexpr uint32_t TOP = 0xFFFFFFFFu;
constexpr uint32_t HALF = 0x80000000u;
constexpr uint32_t FIRST_QTR = 0x40000000u;
constexpr uint32_t THIRD_QTR = 0xC0000000u;
constexpr uint32_t SCALE_TOTAL = 32767;

#ifndef UP_TRUE_INIT
#define UP_TRUE_INIT 3
#endif
#ifndef LEFT_TRUE_INIT
#define LEFT_TRUE_INIT 4
#endif
#ifndef PREV_TRUE_INIT
#define PREV_TRUE_INIT 3
#endif
#ifndef IMPOSSIBLE_FALSE_INIT
#define IMPOSSIBLE_FALSE_INIT 60000
#endif
#ifndef FALLBACK_OTHER_INIT
#define FALLBACK_OTHER_INIT 3
#endif

struct Model {
  std::array<std::array<uint16_t, EVENT_SYMS>, CTX_COUNT> event_freq{};
  std::array<std::array<uint16_t, CLASS_SYMS>, CTX_COUNT> class_freq{};
};

struct AdaptiveModel {
  std::array<std::array<uint16_t, EVENT_SYMS>, CTX5_COUNT> event_freq{};
  std::array<std::array<uint16_t, CLASS_SYMS>, CTX5_COUNT> class_freq{};

  AdaptiveModel() {
    for (auto& row : event_freq) row.fill(1);
    for (auto& row : class_freq) row.fill(1);
  }
};

struct AdaptiveModel6 {
  std::array<std::array<uint16_t, EVENT_SYMS>, CTX6_COUNT> event_freq{};
  std::array<std::array<uint16_t, CLASS_SYMS>, CTX6_COUNT> class_freq{};

  AdaptiveModel6() {
    for (auto& row : event_freq) row.fill(1);
    for (auto& row : class_freq) row.fill(1);
  }
};

struct AdaptiveModel7 {
  std::vector<std::array<uint16_t, EVENT_SYMS>> event_freq;
  std::vector<std::array<uint16_t, CLASS_SYMS>> class_freq;

  AdaptiveModel7() : event_freq(CTX7_COUNT), class_freq(CTX7_COUNT) {
    for (auto& row : event_freq) row.fill(1);
    for (auto& row : class_freq) row.fill(1);
  }
};

struct AdaptiveModel8 {
  std::vector<std::array<uint16_t, EVENT_SYMS>> event_freq;
  std::vector<std::array<uint16_t, CLASS_SYMS>> class_freq;

  AdaptiveModel8() : event_freq(CTX8_COUNT), class_freq(CTX8_COUNT) {
    for (auto& row : event_freq) row.fill(1);
    for (auto& row : class_freq) row.fill(1);
  }
};

struct AdaptiveModel9 {
  std::vector<std::array<uint16_t, EVENT_SYMS>> event_freq;
  std::vector<std::array<uint16_t, CLASS_SYMS>> class_freq;

  AdaptiveModel9() : event_freq(CTX9_COUNT), class_freq(CTX9_COUNT) {
    for (auto& row : event_freq) row.fill(1);
    for (auto& row : class_freq) row.fill(1);
  }
};

struct AdaptiveModel9Binary {
  std::vector<std::array<uint16_t, 2>> up_freq;
  std::vector<std::array<uint16_t, 2>> left_freq;
  std::vector<std::array<uint16_t, 2>> prev_freq;
  std::vector<std::array<uint16_t, 2>> ul_freq;
  std::vector<std::array<uint16_t, 2>> ur_freq;
  std::vector<std::array<uint16_t, 2>> pr_freq;
  std::vector<std::array<uint16_t, 2>> pd_freq;
  std::vector<std::array<uint16_t, 2>> up2_freq;
  std::vector<std::array<uint16_t, 2>> left2_freq;
  std::vector<std::array<uint16_t, 2>> pl_freq;
  std::vector<std::array<uint16_t, 2>> pu_freq;
  std::vector<std::array<uint16_t, CLASS_SYMS>> class_freq;

  AdaptiveModel9Binary()
      : up_freq(CTX9_COUNT), left_freq(CTX9_COUNT), prev_freq(CTX9_COUNT), ul_freq(CTX9_COUNT), ur_freq(CTX9_COUNT), pr_freq(CTX9_COUNT), pd_freq(CTX9_COUNT), up2_freq(CTX9_COUNT), left2_freq(CTX9_COUNT), pl_freq(CTX9_COUNT), pu_freq(CTX9_COUNT), class_freq(CTX9_COUNT) {
    for (int ctx = 0; ctx < CTX9_COUNT; ctx++) {
      int v = ctx;
      uint8_t left2 = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t up2 = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t pd = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t pr = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t ur = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t ul = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t up = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t left = static_cast<uint8_t>(v % 6); v /= 6;
      uint8_t prev = static_cast<uint8_t>(v % 6);
      up_freq[ctx] = {1, UP_TRUE_INIT};
      left_freq[ctx] = {1, LEFT_TRUE_INIT};
      prev_freq[ctx] = {1, PREV_TRUE_INIT};
      ul_freq[ctx] = {1, 3};
      ur_freq[ctx] = {1, 3};
      pr_freq[ctx] = {1, 3};
      pd_freq[ctx] = {1, 3};
      up2_freq[ctx] = {1, 3};
      left2_freq[ctx] = {1, 3};
      pl_freq[ctx] = {1, 3};
      pu_freq[ctx] = {1, 3};
      class_freq[ctx].fill(1);
      for (uint8_t cls = 0; cls < CLASS_SYMS; cls++) { if (cls != up && cls != left && cls != prev && cls != ul && cls != ur && cls != pr && cls != pd && cls != up2 && cls != left2) class_freq[ctx][cls] = FALLBACK_OTHER_INIT; }
    }
  }
};
struct BitWriter {
  std::vector<uint8_t> bytes;
  uint8_t cur = 0;
  int fill = 0;

  void bit(int b) {
    cur = static_cast<uint8_t>((cur << 1) | (b & 1));
    fill++;
    if (fill == 8) {
      bytes.push_back(cur);
      cur = 0;
      fill = 0;
    }
  }

  void finish() {
    if (fill) {
      cur <<= (8 - fill);
      bytes.push_back(cur);
      cur = 0;
      fill = 0;
    }
  }
};

struct BitReader {
  const std::vector<uint8_t>& bytes;
  size_t pos = 0;
  int left = 0;
  uint8_t cur = 0;

  explicit BitReader(const std::vector<uint8_t>& data) : bytes(data) {}

  int bit() {
    if (left == 0) {
      cur = pos < bytes.size() ? bytes[pos++] : 0;
      left = 8;
    }
    int b = (cur >> 7) & 1;
    cur <<= 1;
    left--;
    return b;
  }
};

struct ArithmeticEncoder {
  uint32_t low = 0;
  uint32_t high = TOP;
  uint32_t pending = 0;
  BitWriter writer;

  void output_bit_plus_pending(int b) {
    writer.bit(b);
    while (pending) {
      writer.bit(!b);
      pending--;
    }
  }

  void encode(uint32_t cum_low, uint32_t cum_high, uint32_t total) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1ull;
    high = static_cast<uint32_t>(low + (range * cum_high) / total - 1ull);
    low = static_cast<uint32_t>(low + (range * cum_low) / total);
    while (true) {
      if (high < HALF) {
        output_bit_plus_pending(0);
      } else if (low >= HALF) {
        output_bit_plus_pending(1);
        low -= HALF;
        high -= HALF;
      } else if (low >= FIRST_QTR && high < THIRD_QTR) {
        pending++;
        low -= FIRST_QTR;
        high -= FIRST_QTR;
      } else {
        break;
      }
      low <<= 1;
      high = (high << 1) | 1u;
    }
  }

  std::vector<uint8_t> finish() {
    pending++;
    if (low < FIRST_QTR) {
      output_bit_plus_pending(0);
    } else {
      output_bit_plus_pending(1);
    }
    writer.finish();
    return std::move(writer.bytes);
  }
};

struct ArithmeticDecoder {
  uint32_t low = 0;
  uint32_t high = TOP;
  uint32_t value = 0;
  BitReader reader;

  explicit ArithmeticDecoder(const std::vector<uint8_t>& data) : reader(data) {
    for (int i = 0; i < 32; i++) value = (value << 1) | reader.bit();
  }

  uint32_t scaled(uint32_t total) const {
    uint64_t range = static_cast<uint64_t>(high) - low + 1ull;
    return static_cast<uint32_t>((((static_cast<uint64_t>(value) - low + 1ull) * total) - 1ull) / range);
  }

  void update(uint32_t cum_low, uint32_t cum_high, uint32_t total) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1ull;
    high = static_cast<uint32_t>(low + (range * cum_high) / total - 1ull);
    low = static_cast<uint32_t>(low + (range * cum_low) / total);
    while (true) {
      if (high < HALF) {
        // no-op
      } else if (low >= HALF) {
        value -= HALF;
        low -= HALF;
        high -= HALF;
      } else if (low >= FIRST_QTR && high < THIRD_QTR) {
        value -= FIRST_QTR;
        low -= FIRST_QTR;
        high -= FIRST_QTR;
      } else {
        break;
      }
      low <<= 1;
      high = (high << 1) | 1u;
      value = (value << 1) | static_cast<uint32_t>(reader.bit());
    }
  }
};

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("failed to open input: " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("failed to open output: " + path);
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

int ctx_id(uint8_t prev, uint8_t left, uint8_t up) {
  return static_cast<int>(prev) * 36 + static_cast<int>(left) * 6 + static_cast<int>(up);
}

uint8_t get_prev(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int w, int xcoord) {
  if (t == 0) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y) * w + xcoord];
}

uint8_t get_left(const std::vector<uint8_t>& decoded, size_t base, int xcoord) {
  if (xcoord == 0) return SENTINEL;
  return decoded[base + static_cast<size_t>(xcoord - 1)];
}

uint8_t get_up(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y == 0) return SENTINEL;
  return decoded[base - static_cast<size_t>(w) + xcoord];
}

uint8_t get_up_left(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y == 0 || xcoord == 0) return SENTINEL;
  return decoded[base - static_cast<size_t>(w) + xcoord - 1];
}

uint8_t get_up_right(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y == 0 || xcoord + 1 >= w) return SENTINEL;
  return decoded[base - static_cast<size_t>(w) + xcoord + 1];
}

uint8_t get_left2(const std::vector<uint8_t>& decoded, size_t base, int xcoord) {
  if (xcoord < 2) return SENTINEL;
  return decoded[base + static_cast<size_t>(xcoord - 2)];
}

uint8_t get_up2(const std::vector<uint8_t>& decoded, size_t base, int y, int w, int xcoord) {
  if (y < 2) return SENTINEL;
  return decoded[base - static_cast<size_t>(2 * w) + xcoord];
}

uint8_t get_prev_right(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int w, int xcoord) {
  if (t == 0 || xcoord + 1 >= w) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y) * w + xcoord + 1];
}

uint8_t get_prev_left(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int w, int xcoord) {
  if (t == 0 || xcoord == 0) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y) * w + xcoord - 1];
}

uint8_t get_prev_down(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int h, int w, int xcoord) {
  if (t == 0 || y + 1 >= h) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y + 1) * w + xcoord];
}

uint8_t get_prev_up(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int w, int xcoord) {
  if (t == 0 || y == 0) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y - 1) * w + xcoord];
}

uint8_t get_prev_down_left(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int h, int w, int xcoord) {
  if (t == 0 || y + 1 >= h || xcoord == 0) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y + 1) * w + xcoord - 1];
}

uint8_t get_prev_down_right(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int h, int w, int xcoord) {
  if (t == 0 || y + 1 >= h || xcoord + 1 >= w) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y + 1) * w + xcoord + 1];
}

uint8_t event_for(uint8_t cls, uint8_t prev, uint8_t left, uint8_t up) {
  if (cls == prev) return 0;
  if (cls == left) return 1;
  if (cls == up) return 2;
  return 3;
}

int ctx5_id(uint8_t prev, uint8_t left, uint8_t up, uint8_t up_left, uint8_t up_right) {
  return static_cast<int>(prev) * 1296 + static_cast<int>(left) * 216 + static_cast<int>(up) * 36 + static_cast<int>(up_left) * 6 + static_cast<int>(up_right);
}

int ctx6_id(uint8_t prev, uint8_t left, uint8_t up, uint8_t up_left, uint8_t up_right, uint8_t prev_right) {
  return static_cast<int>(prev) * 7776 + static_cast<int>(left) * 1296 + static_cast<int>(up) * 216 + static_cast<int>(up_left) * 36 + static_cast<int>(up_right) * 6 + static_cast<int>(prev_right);
}

int ctx7_id(uint8_t prev, uint8_t left, uint8_t up, uint8_t up_left, uint8_t up_right, uint8_t prev_right, uint8_t prev_down) {
  return static_cast<int>(prev) * 46656 + static_cast<int>(left) * 7776 + static_cast<int>(up) * 1296 + static_cast<int>(up_left) * 216 + static_cast<int>(up_right) * 36 + static_cast<int>(prev_right) * 6 + static_cast<int>(prev_down);
}

int ctx8_id(uint8_t prev, uint8_t left, uint8_t up, uint8_t up_left, uint8_t up_right, uint8_t prev_right, uint8_t prev_down, uint8_t prev_left) {
  return ctx7_id(prev, left, up, up_left, up_right, prev_right, prev_down) * 6 + static_cast<int>(prev_left);
}

int ctx9_id(uint8_t prev, uint8_t left, uint8_t up, uint8_t up_left, uint8_t up_right, uint8_t prev_right, uint8_t prev_down, uint8_t up2, uint8_t extra) {
  return ctx8_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2) * 6 + static_cast<int>(extra);
}

template <size_t N>
std::array<uint16_t, N> scale_counts(const std::array<uint64_t, N>& counts) {
  std::array<uint16_t, N> out{};
  uint64_t total = 0;
  for (auto c : counts) total += c;
  if (total == 0) {
    for (size_t i = 0; i < N; i++) out[i] = 1;
    return out;
  }
  uint32_t used = 0;
  for (size_t i = 0; i < N; i++) {
    uint32_t v = std::max<uint32_t>(1, static_cast<uint32_t>((counts[i] * SCALE_TOTAL) / total));
    out[i] = static_cast<uint16_t>(v);
    used += v;
  }
  while (used > SCALE_TOTAL) {
    size_t best = 0;
    for (size_t i = 1; i < N; i++) {
      if (out[i] > out[best]) best = i;
    }
    if (out[best] <= 1) break;
    out[best]--;
    used--;
  }
  while (used < SCALE_TOTAL) {
    size_t best = 0;
    for (size_t i = 1; i < N; i++) {
      if (counts[i] > counts[best]) best = i;
    }
    out[best]++;
    used++;
  }
  return out;
}

Model build_model(const std::vector<uint8_t>& x, int t_count, int h, int w) {
  std::array<std::array<uint64_t, EVENT_SYMS>, CTX_COUNT> event_counts{};
  std::array<std::array<uint64_t, CLASS_SYMS>, CTX_COUNT> class_counts{};
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        int ctx = ctx_id(prev, left, up);
        uint8_t ev = event_for(cls, prev, left, up);
        event_counts[ctx][ev]++;
        if (ev == 3) class_counts[ctx][cls]++;
        decoded[idx] = cls;
      }
    }
  }
  Model model;
  for (int ctx = 0; ctx < CTX_COUNT; ctx++) {
    model.event_freq[ctx] = scale_counts<EVENT_SYMS>(event_counts[ctx]);
    model.class_freq[ctx] = scale_counts<CLASS_SYMS>(class_counts[ctx]);
  }
  return model;
}

template <size_t N>
void encode_symbol(ArithmeticEncoder& enc, const std::array<uint16_t, N>& freq, uint32_t sym) {
  uint32_t cum = 0;
  for (uint32_t i = 0; i < sym; i++) cum += freq[i];
  uint32_t total = 0;
  for (uint32_t i = 0; i < N; i++) total += freq[i];
  enc.encode(cum, cum + freq[sym], total);
}

template <size_t N>
void update_adaptive(std::array<uint16_t, N>& freq, uint32_t sym) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < N; i++) total += freq[i];
  if (total >= SCALE_TOTAL) {
    total = 0;
    for (uint32_t i = 0; i < N; i++) {
      freq[i] = static_cast<uint16_t>(std::max<uint32_t>(1, (freq[i] + 1) >> 1));
      total += freq[i];
    }
  }
  (void)total;
  freq[sym] = static_cast<uint16_t>(
      std::min<uint32_t>(65535, static_cast<uint32_t>(freq[sym]) + 16));
}

template <size_t N>
uint32_t decode_symbol(ArithmeticDecoder& dec, const std::array<uint16_t, N>& freq) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < N; i++) total += freq[i];
  uint32_t v = dec.scaled(total);
  uint32_t cum = 0;
  for (uint32_t sym = 0; sym < N; sym++) {
    uint32_t next = cum + freq[sym];
    if (v < next) {
      dec.update(cum, next, total);
      return sym;
    }
    cum = next;
  }
  throw std::runtime_error("decode_symbol out of range");
}

std::vector<uint8_t> encode_payload(const std::vector<uint8_t>& x, int t_count, int h, int w, const Model& model) {
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        int ctx = ctx_id(prev, left, up);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        if (ev == 3) encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload(const std::vector<uint8_t>& bits, int t_count, int h, int w, const Model& model) {
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        int ctx = ctx_id(prev, left, up);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

std::vector<uint8_t> encode_payload_adaptive5(const std::vector<uint8_t>& x, int t_count, int h, int w) {
  AdaptiveModel model;
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        int ctx = ctx5_id(prev, left, up, ul, ur);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        if (ev == 3) {
          encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload_adaptive5(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  AdaptiveModel model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord);
        uint8_t ur = get_up_right(out, base, y, w, xcoord);
        int ctx = ctx5_id(prev, left, up, ul, ur);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else {
          cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

std::vector<uint8_t> encode_payload_adaptive6pr(const std::vector<uint8_t>& x, int t_count, int h, int w) {
  AdaptiveModel6 model;
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        uint8_t pr = get_prev_right(x, frame_size, t, y, w, xcoord);
        int ctx = ctx6_id(prev, left, up, ul, ur, pr);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        if (ev == 3) {
          encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload_adaptive6pr(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  AdaptiveModel6 model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord);
        uint8_t ur = get_up_right(out, base, y, w, xcoord);
        uint8_t pr = get_prev_right(out, frame_size, t, y, w, xcoord);
        int ctx = ctx6_id(prev, left, up, ul, ur, pr);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else {
          cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

std::vector<uint8_t> encode_payload_adaptive7prpd(const std::vector<uint8_t>& x, int t_count, int h, int w) {
  AdaptiveModel7 model;
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        uint8_t pr = get_prev_right(x, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(x, frame_size, t, y, h, w, xcoord);
        int ctx = ctx7_id(prev, left, up, ul, ur, pr, pd);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        if (ev == 3) {
          encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload_adaptive7prpd(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  AdaptiveModel7 model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord);
        uint8_t ur = get_up_right(out, base, y, w, xcoord);
        uint8_t pr = get_prev_right(out, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(out, frame_size, t, y, h, w, xcoord);
        int ctx = ctx7_id(prev, left, up, ul, ur, pr, pd);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else {
          cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

std::vector<uint8_t> encode_payload_adaptive8prpdpl(const std::vector<uint8_t>& x, int t_count, int h, int w) {
  AdaptiveModel8 model;
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        uint8_t pr = get_prev_right(x, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(x, frame_size, t, y, h, w, xcoord);
        uint8_t pl = get_prev_left(x, frame_size, t, y, w, xcoord);
        int ctx = ctx8_id(prev, left, up, ul, ur, pr, pd, pl);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        if (ev == 3) {
          encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload_adaptive8prpdpl(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  AdaptiveModel8 model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord);
        uint8_t ur = get_up_right(out, base, y, w, xcoord);
        uint8_t pr = get_prev_right(out, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(out, frame_size, t, y, h, w, xcoord);
        uint8_t pl = get_prev_left(out, frame_size, t, y, w, xcoord);
        int ctx = ctx8_id(prev, left, up, ul, ur, pr, pd, pl);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else {
          cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

uint8_t get_extra8_value(const std::vector<uint8_t>& src, const std::vector<uint8_t>& decoded, size_t frame_size, size_t base, int t, int y, int h, int w, int xcoord, int extra_kind) {
  switch (extra_kind) {
    case 0: return get_prev_left(src, frame_size, t, y, w, xcoord);
    case 1: return get_prev_up(src, frame_size, t, y, w, xcoord);
    case 2: return get_left2(decoded, base, xcoord);
    case 3: return get_up2(decoded, base, y, w, xcoord);
    case 4: return get_prev_down_left(src, frame_size, t, y, h, w, xcoord);
    case 5: return get_prev_down_right(src, frame_size, t, y, h, w, xcoord);
    default: return SENTINEL;
  }
}

std::vector<uint8_t> encode_payload_adaptive8x(const std::vector<uint8_t>& x, int t_count, int h, int w, int extra_kind) {
  AdaptiveModel8 model;
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        uint8_t pr = get_prev_right(x, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(x, frame_size, t, y, h, w, xcoord);
        uint8_t extra = get_extra8_value(x, decoded, frame_size, base, t, y, h, w, xcoord, extra_kind);
        int ctx = ctx8_id(prev, left, up, ul, ur, pr, pd, extra);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        if (ev == 3) {
          encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload_adaptive8x(const std::vector<uint8_t>& bits, int t_count, int h, int w, int extra_kind) {
  AdaptiveModel8 model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord);
        uint8_t ur = get_up_right(out, base, y, w, xcoord);
        uint8_t pr = get_prev_right(out, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(out, frame_size, t, y, h, w, xcoord);
        uint8_t extra = get_extra8_value(out, out, frame_size, base, t, y, h, w, xcoord, extra_kind);
        int ctx = ctx8_id(prev, left, up, ul, ur, pr, pd, extra);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else {
          cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

std::vector<uint8_t> encode_payload_adaptive9x(const std::vector<uint8_t>& x, int t_count, int h, int w, int extra_kind) {
  AdaptiveModel9 model;
  ArithmeticEncoder enc;
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord;
        uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(decoded, base, xcoord);
        uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord);
        uint8_t ur = get_up_right(decoded, base, y, w, xcoord);
        uint8_t pr = get_prev_right(x, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(x, frame_size, t, y, h, w, xcoord);
        uint8_t up2 = get_up2(decoded, base, y, w, xcoord);
        uint8_t extra = get_extra8_value(x, decoded, frame_size, base, t, y, h, w, xcoord, extra_kind);
        int ctx = ctx9_id(prev, left, up, ul, ur, pr, pd, up2, extra);
        uint8_t ev = event_for(cls, prev, left, up);
        encode_symbol<EVENT_SYMS>(enc, model.event_freq[ctx], ev);
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        if (ev == 3) {
          encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls);
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        decoded[idx] = cls;
      }
    }
  }
  return enc.finish();
}

std::vector<uint8_t> decode_payload_adaptive9x(const std::vector<uint8_t>& bits, int t_count, int h, int w, int extra_kind) {
  AdaptiveModel9 model;
  ArithmeticDecoder dec(bits);
  const size_t frame_size = static_cast<size_t>(h) * w;
  std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) {
    for (int y = 0; y < h; y++) {
      size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w;
      for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord);
        uint8_t left = get_left(out, base, xcoord);
        uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord);
        uint8_t ur = get_up_right(out, base, y, w, xcoord);
        uint8_t pr = get_prev_right(out, frame_size, t, y, w, xcoord);
        uint8_t pd = get_prev_down(out, frame_size, t, y, h, w, xcoord);
        uint8_t up2 = get_up2(out, base, y, w, xcoord);
        uint8_t extra = get_extra8_value(out, out, frame_size, base, t, y, h, w, xcoord, extra_kind);
        int ctx = ctx9_id(prev, left, up, ul, ur, pr, pd, up2, extra);
        uint8_t ev = static_cast<uint8_t>(decode_symbol<EVENT_SYMS>(dec, model.event_freq[ctx]));
        update_adaptive<EVENT_SYMS>(model.event_freq[ctx], ev);
        uint8_t cls = 0;
        if (ev == 0) cls = prev;
        else if (ev == 1) cls = left;
        else if (ev == 2) cls = up;
        else {
          cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
          update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

std::vector<uint8_t> encode_payload_adaptive9bin(const std::vector<uint8_t>& x, int t_count, int h, int w) {
  AdaptiveModel9Binary model; ArithmeticEncoder enc; const size_t frame_size = static_cast<size_t>(h) * w; std::vector<uint8_t> decoded(x.size(), 0);
  for (int t = 0; t < t_count; t++) { for (int y = 0; y < h; y++) { size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w; for (int xcoord = 0; xcoord < w; xcoord++) {
        size_t idx = base + xcoord; uint8_t cls = x[idx];
        uint8_t prev = get_prev(x, frame_size, t, y, w, xcoord); uint8_t left = get_left(decoded, base, xcoord); uint8_t up = get_up(decoded, base, y, w, xcoord);
        uint8_t ul = get_up_left(decoded, base, y, w, xcoord); uint8_t ur = get_up_right(decoded, base, y, w, xcoord); uint8_t pr = get_prev_right(x, frame_size, t, y, w, xcoord); uint8_t pd = get_prev_down(x, frame_size, t, y, h, w, xcoord);
        uint8_t up2 = get_up2(decoded, base, y, w, xcoord); uint8_t left2 = get_left2(decoded, base, xcoord);
        uint8_t pl = get_prev_left(x, frame_size, t, y, w, xcoord);
        uint8_t pu = get_prev_up(x, frame_size, t, y, w, xcoord);
        uint8_t pdl = get_prev_down_left(x, frame_size, t, y, h, w, xcoord);
        uint8_t pdr = get_prev_down_right(x, frame_size, t, y, h, w, xcoord);
        int ctx = ctx9_id(prev, left, up, ul, ur, pr, pd, up2, left2); bool matched = false;
        if (!matched && (up != SENTINEL)) {
          uint8_t b = cls == up;
          encode_symbol<2>(enc, model.up_freq[ctx], b);
          update_adaptive<2>(model.up_freq[ctx], b);
          matched = b;
        }
        if (!matched && (left != SENTINEL && left != up)) {
          uint8_t b = cls == left;
          encode_symbol<2>(enc, model.left_freq[ctx], b);
          update_adaptive<2>(model.left_freq[ctx], b);
          matched = b;
        }
        if (!matched && (prev != SENTINEL && prev != up && prev != left)) {
          uint8_t b = cls == prev;
          encode_symbol<2>(enc, model.prev_freq[ctx], b);
          update_adaptive<2>(model.prev_freq[ctx], b);
          matched = b;
        }
        if (!matched && (ul != SENTINEL && ul != up && ul != left && ul != prev)) {
          uint8_t b = cls == ul;
          encode_symbol<2>(enc, model.ul_freq[ctx], b);
          update_adaptive<2>(model.ul_freq[ctx], b);
          matched = b;
        }
        if (!matched && (ur != SENTINEL && ur != up && ur != left && ur != prev && ur != ul)) {
          uint8_t b = cls == ur;
          encode_symbol<2>(enc, model.ur_freq[ctx], b);
          update_adaptive<2>(model.ur_freq[ctx], b);
          matched = b;
        }
        if (!matched && (pr != SENTINEL && pr != up && pr != left && pr != prev && pr != ul && pr != ur)) {
          uint8_t b = cls == pr;
          encode_symbol<2>(enc, model.pr_freq[ctx], b);
          update_adaptive<2>(model.pr_freq[ctx], b);
          matched = b;
        }
        if (!matched && (pd != SENTINEL && pd != up && pd != left && pd != prev && pd != ul && pd != ur && pd != pr)) {
          uint8_t b = cls == pd;
          encode_symbol<2>(enc, model.pd_freq[ctx], b);
          update_adaptive<2>(model.pd_freq[ctx], b);
          matched = b;
        }
        if (!matched && (up2 != SENTINEL && up2 != up && up2 != left && up2 != prev && up2 != ul && up2 != ur && up2 != pr && up2 != pd)) {
          uint8_t b = cls == up2;
          encode_symbol<2>(enc, model.up2_freq[ctx], b);
          update_adaptive<2>(model.up2_freq[ctx], b);
          matched = b;
        }
        if (!matched && (left2 != SENTINEL && left2 != up && left2 != left && left2 != prev && left2 != ul && left2 != ur && left2 != pr && left2 != pd && left2 != up2)) {
          uint8_t b = cls == left2;
          encode_symbol<2>(enc, model.left2_freq[ctx], b);
          update_adaptive<2>(model.left2_freq[ctx], b);
          matched = b;
        }
        if (!matched && (pl != SENTINEL && pl != up && pl != left && pl != prev && pl != ul && pl != ur && pl != pr && pl != pd && pl != up2 && pl != left2)) {
          uint8_t b = cls == pl;
          encode_symbol<2>(enc, model.pl_freq[ctx], b);
          update_adaptive<2>(model.pl_freq[ctx], b);
          matched = b;
        }
        if (!matched && (pu != SENTINEL && pu != up && pu != left && pu != prev && pu != ul && pu != ur && pu != pr && pu != pd && pu != up2 && pu != left2 && pu != pl)) {
          uint8_t b = cls == pu;
          encode_symbol<2>(enc, model.pu_freq[ctx], b);
          update_adaptive<2>(model.pu_freq[ctx], b);
          matched = b;
        }
        if (!matched) { encode_symbol<CLASS_SYMS>(enc, model.class_freq[ctx], cls); update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls); } decoded[idx] = cls;
  } } } return enc.finish(); }

std::vector<uint8_t> decode_payload_adaptive9bin(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  AdaptiveModel9Binary model; ArithmeticDecoder dec(bits); const size_t frame_size = static_cast<size_t>(h) * w; std::vector<uint8_t> out(static_cast<size_t>(t_count) * frame_size, 0);
  for (int t = 0; t < t_count; t++) { for (int y = 0; y < h; y++) { size_t base = static_cast<size_t>(t) * frame_size + static_cast<size_t>(y) * w; for (int xcoord = 0; xcoord < w; xcoord++) {
        uint8_t prev = get_prev(out, frame_size, t, y, w, xcoord); uint8_t left = get_left(out, base, xcoord); uint8_t up = get_up(out, base, y, w, xcoord);
        uint8_t ul = get_up_left(out, base, y, w, xcoord); uint8_t ur = get_up_right(out, base, y, w, xcoord); uint8_t pr = get_prev_right(out, frame_size, t, y, w, xcoord); uint8_t pd = get_prev_down(out, frame_size, t, y, h, w, xcoord);
        uint8_t up2 = get_up2(out, base, y, w, xcoord); uint8_t left2 = get_left2(out, base, xcoord);
        uint8_t pl = get_prev_left(out, frame_size, t, y, w, xcoord);
        uint8_t pu = get_prev_up(out, frame_size, t, y, w, xcoord);
        uint8_t pdl = get_prev_down_left(out, frame_size, t, y, h, w, xcoord);
        uint8_t pdr = get_prev_down_right(out, frame_size, t, y, h, w, xcoord);
        int ctx = ctx9_id(prev, left, up, ul, ur, pr, pd, up2, left2); uint8_t cls = 0; bool matched = false;
        if (!matched && (up != SENTINEL)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.up_freq[ctx]));
          update_adaptive<2>(model.up_freq[ctx], b);
          if (b) { cls = up; matched = true; }
        }
        if (!matched && (left != SENTINEL && left != up)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.left_freq[ctx]));
          update_adaptive<2>(model.left_freq[ctx], b);
          if (b) { cls = left; matched = true; }
        }
        if (!matched && (prev != SENTINEL && prev != up && prev != left)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.prev_freq[ctx]));
          update_adaptive<2>(model.prev_freq[ctx], b);
          if (b) { cls = prev; matched = true; }
        }
        if (!matched && (ul != SENTINEL && ul != up && ul != left && ul != prev)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.ul_freq[ctx]));
          update_adaptive<2>(model.ul_freq[ctx], b);
          if (b) { cls = ul; matched = true; }
        }
        if (!matched && (ur != SENTINEL && ur != up && ur != left && ur != prev && ur != ul)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.ur_freq[ctx]));
          update_adaptive<2>(model.ur_freq[ctx], b);
          if (b) { cls = ur; matched = true; }
        }
        if (!matched && (pr != SENTINEL && pr != up && pr != left && pr != prev && pr != ul && pr != ur)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.pr_freq[ctx]));
          update_adaptive<2>(model.pr_freq[ctx], b);
          if (b) { cls = pr; matched = true; }
        }
        if (!matched && (pd != SENTINEL && pd != up && pd != left && pd != prev && pd != ul && pd != ur && pd != pr)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.pd_freq[ctx]));
          update_adaptive<2>(model.pd_freq[ctx], b);
          if (b) { cls = pd; matched = true; }
        }
        if (!matched && (up2 != SENTINEL && up2 != up && up2 != left && up2 != prev && up2 != ul && up2 != ur && up2 != pr && up2 != pd)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.up2_freq[ctx]));
          update_adaptive<2>(model.up2_freq[ctx], b);
          if (b) { cls = up2; matched = true; }
        }
        if (!matched && (left2 != SENTINEL && left2 != up && left2 != left && left2 != prev && left2 != ul && left2 != ur && left2 != pr && left2 != pd && left2 != up2)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.left2_freq[ctx]));
          update_adaptive<2>(model.left2_freq[ctx], b);
          if (b) { cls = left2; matched = true; }
        }
        if (!matched && (pl != SENTINEL && pl != up && pl != left && pl != prev && pl != ul && pl != ur && pl != pr && pl != pd && pl != up2 && pl != left2)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.pl_freq[ctx]));
          update_adaptive<2>(model.pl_freq[ctx], b);
          if (b) { cls = pl; matched = true; }
        }
        if (!matched && (pu != SENTINEL && pu != up && pu != left && pu != prev && pu != ul && pu != ur && pu != pr && pu != pd && pu != up2 && pu != left2 && pu != pl)) {
          uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.pu_freq[ctx]));
          update_adaptive<2>(model.pu_freq[ctx], b);
          if (b) { cls = pu; matched = true; }
        }
        if (!matched) { cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx])); update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls); } out[base + xcoord] = cls;
  } } } return out; }

void append_u32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; i++) out.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
}

void append_u16(std::vector<uint8_t>& out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

std::vector<uint8_t> serialize(const std::vector<uint8_t>& bits, int t_count, int h, int w, const Model& model) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'Q', 'M', 'R', '1'});
  append_u32(out, static_cast<uint32_t>(t_count));
  append_u32(out, static_cast<uint32_t>(h));
  append_u32(out, static_cast<uint32_t>(w));
  for (int ctx = 0; ctx < CTX_COUNT; ctx++) {
    for (auto v : model.event_freq[ctx]) append_u16(out, v);
    for (auto v : model.class_freq[ctx]) append_u16(out, v);
  }
  append_u32(out, static_cast<uint32_t>(bits.size()));
  out.insert(out.end(), bits.begin(), bits.end());
  return out;
}

std::vector<uint8_t> serialize_adaptive(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'Q', 'M', 'A', '5'});
  append_u32(out, static_cast<uint32_t>(t_count));
  append_u32(out, static_cast<uint32_t>(h));
  append_u32(out, static_cast<uint32_t>(w));
  append_u32(out, static_cast<uint32_t>(bits.size()));
  out.insert(out.end(), bits.begin(), bits.end());
  return out;
}

std::vector<uint8_t> serialize_adaptive6(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'Q', 'M', 'A', '6'});
  append_u32(out, static_cast<uint32_t>(t_count));
  append_u32(out, static_cast<uint32_t>(h));
  append_u32(out, static_cast<uint32_t>(w));
  append_u32(out, static_cast<uint32_t>(bits.size()));
  out.insert(out.end(), bits.begin(), bits.end());
  return out;
}

std::vector<uint8_t> serialize_adaptive7(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'Q', 'M', 'A', '7'});
  append_u32(out, static_cast<uint32_t>(t_count));
  append_u32(out, static_cast<uint32_t>(h));
  append_u32(out, static_cast<uint32_t>(w));
  append_u32(out, static_cast<uint32_t>(bits.size()));
  out.insert(out.end(), bits.begin(), bits.end());
  return out;
}

std::vector<uint8_t> serialize_adaptive8(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'Q', 'M', 'A', '8'});
  append_u32(out, static_cast<uint32_t>(t_count));
  append_u32(out, static_cast<uint32_t>(h));
  append_u32(out, static_cast<uint32_t>(w));
  append_u32(out, static_cast<uint32_t>(bits.size()));
  out.insert(out.end(), bits.begin(), bits.end());
  return out;
}

std::vector<uint8_t> serialize_adaptive9(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'Q', 'M', 'A', '9'});
  append_u32(out, static_cast<uint32_t>(t_count));
  append_u32(out, static_cast<uint32_t>(h));
  append_u32(out, static_cast<uint32_t>(w));
  append_u32(out, static_cast<uint32_t>(bits.size()));
  out.insert(out.end(), bits.begin(), bits.end());
  return out;
}

uint32_t read_u32(const std::vector<uint8_t>& data, size_t& off) {
  if (off + 4 > data.size()) throw std::runtime_error("truncated u32");
  uint32_t v = static_cast<uint32_t>(data[off]) | (static_cast<uint32_t>(data[off + 1]) << 8) | (static_cast<uint32_t>(data[off + 2]) << 16) | (static_cast<uint32_t>(data[off + 3]) << 24);
  off += 4;
  return v;
}

std::vector<uint8_t> decode_packed_adaptive6(const std::vector<uint8_t>& packed) {
  if (packed.size() < 20 || packed[0] != 'Q' || packed[1] != 'M' || packed[2] != 'A' || packed[3] != '6') {
    throw std::runtime_error("not a QMA6 range-mask payload");
  }
  size_t off = 4;
  int t_count = static_cast<int>(read_u32(packed, off));
  int h = static_cast<int>(read_u32(packed, off));
  int w = static_cast<int>(read_u32(packed, off));
  uint32_t bit_bytes = read_u32(packed, off);
  if (off + bit_bytes > packed.size()) throw std::runtime_error("truncated bitstream");
  std::vector<uint8_t> bits(packed.begin() + static_cast<std::ptrdiff_t>(off), packed.begin() + static_cast<std::ptrdiff_t>(off + bit_bytes));
  return decode_payload_adaptive6pr(bits, t_count, h, w);
}

std::vector<uint8_t> decode_packed_adaptive7(const std::vector<uint8_t>& packed) {
  if (packed.size() < 20 || packed[0] != 'Q' || packed[1] != 'M' || packed[2] != 'A' || packed[3] != '7') {
    throw std::runtime_error("not a QMA7 range-mask payload");
  }
  size_t off = 4;
  int t_count = static_cast<int>(read_u32(packed, off));
  int h = static_cast<int>(read_u32(packed, off));
  int w = static_cast<int>(read_u32(packed, off));
  uint32_t bit_bytes = read_u32(packed, off);
  if (off + bit_bytes > packed.size()) throw std::runtime_error("truncated bitstream");
  std::vector<uint8_t> bits(packed.begin() + static_cast<std::ptrdiff_t>(off), packed.begin() + static_cast<std::ptrdiff_t>(off + bit_bytes));
  return decode_payload_adaptive7prpd(bits, t_count, h, w);
}

std::vector<uint8_t> decode_packed_adaptive8(const std::vector<uint8_t>& packed) {
  if (packed.size() < 20 || packed[0] != 'Q' || packed[1] != 'M' || packed[2] != 'A' || packed[3] != '8') {
    throw std::runtime_error("not a QMA8 range-mask payload");
  }
  size_t off = 4;
  int t_count = static_cast<int>(read_u32(packed, off));
  int h = static_cast<int>(read_u32(packed, off));
  int w = static_cast<int>(read_u32(packed, off));
  uint32_t bit_bytes = read_u32(packed, off);
  if (off + bit_bytes > packed.size()) throw std::runtime_error("truncated bitstream");
  std::vector<uint8_t> bits(packed.begin() + static_cast<std::ptrdiff_t>(off), packed.begin() + static_cast<std::ptrdiff_t>(off + bit_bytes));
  // QMA8 submission payloads use the best audit variant:
  // prev/left/up/up-left/up-right/prev-right/prev-down/current-up2.
  return decode_payload_adaptive8x(bits, t_count, h, w, 3);
}

std::vector<uint8_t> decode_packed_adaptive9(const std::vector<uint8_t>& packed) {
  if (packed.size() < 20 || packed[0] != 'Q' || packed[1] != 'M' || packed[2] != 'A' || packed[3] != '9') {
    throw std::runtime_error("not a QMA9 range-mask payload");
  }
  size_t off = 4;
  int t_count = static_cast<int>(read_u32(packed, off));
  int h = static_cast<int>(read_u32(packed, off));
  int w = static_cast<int>(read_u32(packed, off));
  uint32_t bit_bytes = read_u32(packed, off);
  if (off + bit_bytes > packed.size()) throw std::runtime_error("truncated bitstream");
  std::vector<uint8_t> bits(packed.begin() + static_cast<std::ptrdiff_t>(off), packed.begin() + static_cast<std::ptrdiff_t>(off + bit_bytes));
  // Best audited QMA9 variant: hierarchical up/left/prev binary decisions
  // with QMA8 up2 plus current left2 context.
  return decode_payload_adaptive9bin(bits, t_count, h, w);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "decode") {
    try {
      std::vector<uint8_t> packed = read_file(argv[2]);
      std::vector<uint8_t> decoded;
      if (packed.size() >= 4 && packed[3] == '9') decoded = decode_packed_adaptive9(packed);
      else if (packed.size() >= 4 && packed[3] == '8') decoded = decode_packed_adaptive8(packed);
      else if (packed.size() >= 4 && packed[3] == '7') decoded = decode_packed_adaptive7(packed);
      else decoded = decode_packed_adaptive6(packed);
      write_file(argv[3], decoded);
      std::cout << "{\"decoded_bytes\":" << decoded.size() << "}" << std::endl;
      return 0;
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\\n";
      return 7;
    }
  }
  if (argc != 6 && argc != 7) {
    std::cerr << "usage: exact_mask_range_codec <raw_u8> <T> <H> <W> <out.bin> [static3|adaptive5|adaptive6pr|adaptive7prpd|adaptive8prpdpl|adaptive8prpdpu|adaptive8prpdleft2|adaptive8prpdup2|adaptive8prpdpdl|adaptive8prpdpdr|adaptive9up2left2|adaptive9up2pu|adaptive9up2pl|adaptive9bin]\\n"
              << "   or: exact_mask_range_codec decode <in.bin> <out.raw>\\n";
    return 2;
  }
  std::string input_path = argv[1];
  int t_count = std::stoi(argv[2]);
  int h = std::stoi(argv[3]);
  int w = std::stoi(argv[4]);
  std::string out_path = argv[5];
  std::string mode = argc == 7 ? argv[6] : "static3";
  std::vector<uint8_t> x = read_file(input_path);
  size_t expected = static_cast<size_t>(t_count) * h * w;
  if (x.size() != expected) {
    std::cerr << "input size mismatch: got " << x.size() << " expected " << expected << "\\n";
    return 3;
  }
  for (uint8_t v : x) {
    if (v > 4) {
      std::cerr << "class value out of range\\n";
      return 4;
    }
  }
  std::vector<uint8_t> bits;
  std::vector<uint8_t> decoded;
  std::vector<uint8_t> packed;
  if (mode == "static3") {
    Model model = build_model(x, t_count, h, w);
    bits = encode_payload(x, t_count, h, w, model);
    decoded = decode_payload(bits, t_count, h, w, model);
    packed = serialize(bits, t_count, h, w, model);
  } else if (mode == "adaptive5") {
    bits = encode_payload_adaptive5(x, t_count, h, w);
    decoded = decode_payload_adaptive5(bits, t_count, h, w);
    packed = serialize_adaptive(bits, t_count, h, w);
  } else if (mode == "adaptive6pr") {
    bits = encode_payload_adaptive6pr(x, t_count, h, w);
    decoded = decode_payload_adaptive6pr(bits, t_count, h, w);
    packed = serialize_adaptive6(bits, t_count, h, w);
  } else if (mode == "adaptive7prpd") {
    bits = encode_payload_adaptive7prpd(x, t_count, h, w);
    decoded = decode_payload_adaptive7prpd(bits, t_count, h, w);
    packed = serialize_adaptive7(bits, t_count, h, w);
  } else if (mode == "adaptive8prpdpl") {
    bits = encode_payload_adaptive8prpdpl(x, t_count, h, w);
    decoded = decode_payload_adaptive8prpdpl(bits, t_count, h, w);
    packed = serialize_adaptive8(bits, t_count, h, w);
  } else if (mode == "adaptive8prpdpu") {
    bits = encode_payload_adaptive8x(x, t_count, h, w, 1);
    decoded = decode_payload_adaptive8x(bits, t_count, h, w, 1);
    packed = serialize_adaptive8(bits, t_count, h, w);
  } else if (mode == "adaptive8prpdleft2") {
    bits = encode_payload_adaptive8x(x, t_count, h, w, 2);
    decoded = decode_payload_adaptive8x(bits, t_count, h, w, 2);
    packed = serialize_adaptive8(bits, t_count, h, w);
  } else if (mode == "adaptive8prpdup2") {
    bits = encode_payload_adaptive8x(x, t_count, h, w, 3);
    decoded = decode_payload_adaptive8x(bits, t_count, h, w, 3);
    packed = serialize_adaptive8(bits, t_count, h, w);
  } else if (mode == "adaptive8prpdpdl") {
    bits = encode_payload_adaptive8x(x, t_count, h, w, 4);
    decoded = decode_payload_adaptive8x(bits, t_count, h, w, 4);
    packed = serialize_adaptive8(bits, t_count, h, w);
  } else if (mode == "adaptive8prpdpdr") {
    bits = encode_payload_adaptive8x(x, t_count, h, w, 5);
    decoded = decode_payload_adaptive8x(bits, t_count, h, w, 5);
    packed = serialize_adaptive8(bits, t_count, h, w);
  } else if (mode == "adaptive9up2left2") {
    bits = encode_payload_adaptive9x(x, t_count, h, w, 2);
    decoded = decode_payload_adaptive9x(bits, t_count, h, w, 2);
    packed = serialize_adaptive9(bits, t_count, h, w);
  } else if (mode == "adaptive9up2pu") {
    bits = encode_payload_adaptive9x(x, t_count, h, w, 1);
    decoded = decode_payload_adaptive9x(bits, t_count, h, w, 1);
    packed = serialize_adaptive9(bits, t_count, h, w);
  } else if (mode == "adaptive9up2pl") {
    bits = encode_payload_adaptive9x(x, t_count, h, w, 0);
    decoded = decode_payload_adaptive9x(bits, t_count, h, w, 0);
    packed = serialize_adaptive9(bits, t_count, h, w);
  } else if (mode == "adaptive9bin") {
    bits = encode_payload_adaptive9bin(x, t_count, h, w);
    decoded = decode_payload_adaptive9bin(bits, t_count, h, w);
    packed = serialize_adaptive9(bits, t_count, h, w);
  } else {
    std::cerr << "unknown mode: " << mode << "\\n";
    return 6;
  }
  if (decoded != x) {
    std::cerr << "decode verification failed\\n";
    return 5;
  }
  write_file(out_path, packed);
  std::cout << "{"
            << "\"mode\":\"" << mode << "\","
            << "\"raw_bytes\":" << x.size() << ","
            << "\"bitstream_bytes\":" << bits.size() << ","
            << "\"packed_bytes\":" << packed.size() << ","
            << "\"model_bytes\":" << (packed.size() - bits.size())
            << "}" << std::endl;
  return 0;
}
