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
constexpr int CTX9_COUNT = 10077696;
constexpr int CLASS_SYMS = 5;
constexpr uint32_t TOP = 0xFFFFFFFFu;
constexpr uint32_t HALF = 0x80000000u;
constexpr uint32_t FIRST_QTR = 0x40000000u;
constexpr uint32_t THIRD_QTR = 0xC0000000u;
constexpr uint32_t SCALE_TOTAL = 65535;
constexpr uint16_t UP_TRUE_INIT = 3;
constexpr uint16_t LEFT_TRUE_INIT = 4;
constexpr uint16_t PREV_TRUE_INIT = 3;
constexpr uint16_t IMPOSSIBLE_FALSE_INIT = 60000;
constexpr uint16_t FALLBACK_OTHER_INIT = 3;

struct AdaptiveModel9Binary {
  std::vector<std::array<uint16_t, 2>> prev_freq;
  std::vector<std::array<uint16_t, 2>> left_freq;
  std::vector<std::array<uint16_t, 2>> up_freq;
  std::vector<std::array<uint16_t, CLASS_SYMS>> class_freq;

  AdaptiveModel9Binary()
      : prev_freq(CTX9_COUNT), left_freq(CTX9_COUNT), up_freq(CTX9_COUNT), class_freq(CTX9_COUNT) {
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
      (void)left2;
      (void)up2;
      (void)pd;
      (void)pr;
      (void)ur;
      (void)ul;

      prev_freq[ctx] = {1, PREV_TRUE_INIT};
      left_freq[ctx] = {1, LEFT_TRUE_INIT};
      up_freq[ctx] = {1, UP_TRUE_INIT};
      class_freq[ctx].fill(1);

      if (up == SENTINEL) up_freq[ctx] = {IMPOSSIBLE_FALSE_INIT, 1};
      if (left == SENTINEL || left == up) left_freq[ctx] = {IMPOSSIBLE_FALSE_INIT, 1};
      if (prev == SENTINEL || prev == up || prev == left) prev_freq[ctx] = {IMPOSSIBLE_FALSE_INIT, 1};
      for (uint8_t cls = 0; cls < CLASS_SYMS; cls++) {
        if (cls != up && cls != left && cls != prev) class_freq[ctx][cls] = FALLBACK_OTHER_INIT;
      }
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

uint8_t get_prev_down(const std::vector<uint8_t>& x, size_t frame_size, int t, int y, int h, int w, int xcoord) {
  if (t == 0 || y + 1 >= h) return SENTINEL;
  return x[static_cast<size_t>(t - 1) * frame_size + static_cast<size_t>(y + 1) * w + xcoord];
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
void update_adaptive(std::array<uint16_t, N>& freq, uint32_t sym) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < N; i++) total += freq[i];
  if (total >= SCALE_TOTAL) {
    for (uint32_t i = 0; i < N; i++) freq[i] = static_cast<uint16_t>(std::max<uint32_t>(1, (freq[i] + 1) >> 1));
  }
  freq[sym] = static_cast<uint16_t>(std::min<uint32_t>(65535, static_cast<uint32_t>(freq[sym]) + 20));
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

std::vector<uint8_t> decode_payload_adaptive9bin(const std::vector<uint8_t>& bits, int t_count, int h, int w) {
  AdaptiveModel9Binary model;
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
        uint8_t left2 = get_left2(out, base, xcoord);
        int ctx = ctx9_id(prev, left, up, ul, ur, pr, pd, up2, left2);
        uint8_t cls = 0;
        uint8_t b = static_cast<uint8_t>(decode_symbol<2>(dec, model.up_freq[ctx]));
        update_adaptive<2>(model.up_freq[ctx], b);
        if (b) {
          cls = up;
        } else {
          b = static_cast<uint8_t>(decode_symbol<2>(dec, model.left_freq[ctx]));
          update_adaptive<2>(model.left_freq[ctx], b);
          if (b) {
            cls = left;
          } else {
            b = static_cast<uint8_t>(decode_symbol<2>(dec, model.prev_freq[ctx]));
            update_adaptive<2>(model.prev_freq[ctx], b);
            if (b) {
              cls = prev;
            } else {
              cls = static_cast<uint8_t>(decode_symbol<CLASS_SYMS>(dec, model.class_freq[ctx]));
              update_adaptive<CLASS_SYMS>(model.class_freq[ctx], cls);
            }
          }
        }
        out[base + xcoord] = cls;
      }
    }
  }
  return out;
}

uint32_t read_u32(const std::vector<uint8_t>& data, size_t& off) {
  if (off + 4 > data.size()) throw std::runtime_error("truncated u32");
  uint32_t v = static_cast<uint32_t>(data[off]) | (static_cast<uint32_t>(data[off + 1]) << 8) | (static_cast<uint32_t>(data[off + 2]) << 16) | (static_cast<uint32_t>(data[off + 3]) << 24);
  off += 4;
  return v;
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
  return decode_payload_adaptive9bin(bits, t_count, h, w);
}

}

int main(int argc, char** argv) {
  if (argc != 4 || std::string(argv[1]) != "decode") {
    std::cerr << "usage: range_mask_codec decode <in.bin> <out.raw>\n";
    return 2;
  }
  try {
    std::vector<uint8_t> decoded = decode_packed_adaptive9(read_file(argv[2]));
    write_file(argv[3], decoded);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 7;
  }
}
