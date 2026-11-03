"""Lossless recoding of RID9 integer arrays; MIT, based on PR151's coder."""
import bisect
import math
import struct


class Writer:
    def __init__(self):
        self.low, self.high, self.pending = 0, 0xffffffff, 0
        self.data, self.byte, self.used = bytearray(), 0, 0

    def bit(self, value):
        self.byte = (self.byte << 1) | value
        self.used += 1
        if self.used == 8:
            self.data.append(self.byte)
            self.byte, self.used = 0, 0

    def emit(self, value):
        self.bit(value)
        for _ in range(self.pending):
            self.bit(1-value)
        self.pending = 0

    def put(self, low, high, total):
        width = self.high-self.low+1
        self.high = self.low+width*high//total-1
        self.low += width*low//total
        while True:
            if self.high < 0x80000000:
                self.emit(0)
            elif self.low >= 0x80000000:
                self.emit(1)
                self.low -= 0x80000000
                self.high -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xc0000000:
                self.pending += 1
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self.emit(0 if self.low < 0x40000000 else 1)
        if self.used:
            self.data.append(self.byte << (8-self.used))
        return bytes(self.data)


class Reader:
    def __init__(self, data):
        self.data, self.position = data, 0
        self.low, self.high, self.code = 0, 0xffffffff, 0
        for _ in range(32):
            self.code = (self.code << 1) | self.bit()

    def bit(self):
        offset, shift = divmod(self.position, 8)
        self.position += 1
        return (self.data[offset] >> (7-shift)) & 1 if offset < len(self.data) else 0

    def get(self, cumulative):
        total = cumulative[-1]
        width = self.high-self.low+1
        value = ((self.code-self.low+1)*total-1)//width
        symbol = bisect.bisect_right(cumulative, value)-1
        if symbol < 0 or symbol+1 >= len(cumulative):
            raise ValueError('Invalid arithmetic symbol')
        self.high = self.low+width*cumulative[symbol+1]//total-1
        self.low += width*cumulative[symbol]//total
        while True:
            if self.high < 0x80000000:
                pass
            elif self.low >= 0x80000000:
                self.low -= 0x80000000
                self.high -= 0x80000000
                self.code -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xc0000000:
                self.low -= 0x40000000
                self.high -= 0x40000000
                self.code -= 0x40000000
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self.bit()
        return symbol


class Counts:
    def __init__(self, size, increment):
        self.values = [1]*size
        self.increment = increment

    def cumulative(self):
        result = [0]
        for value in self.values:
            result.append(result[-1]+value)
        return result

    def update(self, symbol):
        self.values[symbol] += self.increment
        if sum(self.values) > 32768:
            self.values = [(x+1)//2 for x in self.values]


def read_number(data, offset):
    value, shift = 0, 0
    while offset < len(data) and shift < 35:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7
    raise ValueError('Invalid variable length integer')


def parse_header(body):
    if body[:6] != b'SM3R\x01\x09' or len(body) < 7:
        raise ValueError('Expected RID9 renderer')
    offset, specs = 7, []
    if not 0 < body[6] <= 128:
        raise ValueError('Invalid renderer array count')
    for _ in range(body[6]):
        _, offset = read_number(body, offset)
        count, offset = read_number(body, offset)
        if offset+2 > len(body):
            raise ValueError('Truncated renderer header')
        bits, increment = body[offset:offset+2]
        offset += 2
        if not 1 <= bits <= 15 or not increment or not 0 < count <= 1 << 20:
            raise ValueError('Invalid renderer array')
        specs.append((count, bits, increment))
    _, offset = read_number(body, offset)
    if offset+4 > len(body):
        raise ValueError('Truncated renderer metadata')
    length, = struct.unpack_from('<I', body, offset)
    offset += 4+length
    if offset > len(body):
        raise ValueError('Truncated renderer metadata stream')
    return specs, offset


def read_arrays(body):
    specs, offset = parse_header(body)
    reader = Reader(body[offset:])
    arrays = []
    for count, bits, increment in specs:
        model = Counts(1 << bits, increment)
        values = []
        for _ in range(count):
            symbol = reader.get(model.cumulative())
            model.update(symbol)
            values.append(symbol)
        arrays.append(values)
    return specs, offset, arrays


def original_stream(specs, arrays):
    writer = Writer()
    for (_, bits, increment), values in zip(specs, arrays):
        model = Counts(1 << bits, increment)
        for value in values:
            cumulative = model.cumulative()
            writer.put(cumulative[value], cumulative[value+1], cumulative[-1])
            model.update(value)
    return writer.finish()


def rearrange(values, bits, mode):
    if not mode:
        return values
    half = 1 << (bits-1)
    return [2*(v-half) if v >= half else 2*(half-v)-1 for v in values]


def symbols(values, bits, low, edge):
    size, bins = 1 << bits, 1 << (bits-low)
    result = []
    for value in values:
        if value < edge:
            result.append((bins+value, None))
        elif value >= size-edge:
            result.append((bins+edge+value-(size-edge), None))
        else:
            result.append((value >> low, value & ((1 << low)-1)))
    return result, bins+2*edge


def estimate(values, alphabet, increment, low):
    counts, total, result = [1]*alphabet, alphabet, 0.0
    for value, suffix in values:
        result += math.log2(total/counts[value]) + (low if suffix is not None else 0)
        counts[value] += increment
        total += increment
        if total > 32768:
            counts = [(x+1)//2 for x in counts]
            total = sum(counts)
    return result


def find_settings(specs, arrays):
    settings = []
    for (_, bits, old_increment), values in zip(specs, arrays):
        best = None
        for mode in range(2):
            arranged = rearrange(values, bits, mode)
            for low in range(bits):
                for edge in range(5):
                    transformed, alphabet = symbols(arranged, bits, low, edge)
                    for increment in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 255):
                        trial = (estimate(transformed, alphabet, increment, low), mode, low, edge, increment)
                        if best is None or trial < best:
                            best = trial
        settings.append(best[1:])
    return settings


def encode(body, settings=None):
    specs, offset, arrays = read_arrays(body)
    if body[offset:] != original_stream(specs, arrays):
        raise ValueError('Noncanonical RID9 stream')
    if offset > 65535:
        raise ValueError('Renderer header too long')
    if settings is None:
        settings = find_settings(specs, arrays)
    if len(settings) != len(specs):
        raise ValueError('Wrong settings count')
    metadata, writer = bytearray(), Writer()
    for (_, bits, _), values, setting in zip(specs, arrays, settings):
        mode, low, edge, increment = setting
        if mode not in (0, 1) or not 0 <= low < bits or not 0 <= edge <= 7 or not 1 <= increment <= 255:
            raise ValueError('Invalid renderer settings')
        metadata += struct.pack('<H', low | (edge << 4) | (increment << 7) | (mode << 15))
        transformed, alphabet = symbols(rearrange(values, bits, mode), bits, low, edge)
        model = Counts(alphabet, increment)
        for symbol, suffix in transformed:
            cumulative = model.cumulative()
            writer.put(cumulative[symbol], cumulative[symbol+1], cumulative[-1])
            model.update(symbol)
            if suffix is not None and low:
                writer.put(suffix, suffix+1, 1 << low)
    return b'RDQ2'+struct.pack('<H', offset)+body[:offset]+metadata+writer.finish()


def decode(data):
    if data[:4] != b'RDQ2' or len(data) < 6:
        raise ValueError('Expected RDQ2 renderer')
    length, = struct.unpack_from('<H', data, 4)
    prefix = data[6:6+length]
    specs, offset = parse_header(prefix)
    if offset != length:
        raise ValueError('Wrong renderer header length')
    position = 6+length
    if position+2*len(specs) >= len(data):
        raise ValueError('Truncated renderer data')
    descriptors = struct.unpack_from('<'+'H'*len(specs), data, position)
    reader = Reader(data[position+2*len(specs):])
    arrays = []
    for (count, bits, _), descriptor in zip(specs, descriptors):
        low, edge, increment, mode = descriptor & 15, (descriptor >> 4) & 7, (descriptor >> 7) & 255, descriptor >> 15
        if low >= bits or not increment:
            raise ValueError('Invalid renderer descriptor')
        size, bins = 1 << bits, 1 << (bits-low)
        model = Counts(bins+2*edge, increment)
        values = []
        for _ in range(count):
            symbol = reader.get(model.cumulative())
            model.update(symbol)
            if symbol >= bins+edge:
                value = size-edge+symbol-(bins+edge)
            elif symbol >= bins:
                value = symbol-bins
            else:
                suffix = reader.get(range((1 << low)+1)) if low else 0
                value = (symbol << low)+suffix
            if mode:
                value = (value//2 if value % 2 == 0 else -(value//2)-1)+(size//2)
            if not 0 <= value < size:
                raise ValueError('Renderer value out of range')
            values.append(value)
        arrays.append(values)
    return prefix+original_stream(specs, arrays)
