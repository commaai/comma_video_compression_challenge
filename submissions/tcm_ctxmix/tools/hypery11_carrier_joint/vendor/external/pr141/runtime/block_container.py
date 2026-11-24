"""BLK2: independently compressed model blocks with charged layout descriptors.

The original BLK1 format remains readable. BLK2 adds codec 2 (Brotli) and a
high codec bit followed by u8 transform/u8 stride. Transform 1 transposes
byte lanes; all transform choices and block boundaries live in the archive.
"""
import lzma
import struct

MAX_MODELS = 1 << 20


def transpose(data: bytes, stride: int, inverse: bool = False) -> bytes:
    if not 1 <= stride <= 255:
        raise ValueError('Invalid transpose stride')
    size = len(data) // stride * stride
    if inverse:
        out = bytearray(size)
        for lane in range(stride):
            out[lane:size:stride] = data[lane * (size // stride):(lane + 1) * (size // stride)]
        return bytes(out) + data[size:]
    return b''.join(data[lane:size:stride] for lane in range(stride)) + data[size:]


def decode_model_blocks(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < 9 or payload[:4] not in (b'BLK1', b'BLK2'):
        raise ValueError('Invalid block container header')
    version2 = payload[:4] == b'BLK2'
    expected, count = struct.unpack_from('<IB', payload, 4)
    if not 0 < expected <= MAX_MODELS or not count:
        raise ValueError('Invalid model size or block count')
    offset = 9
    output = bytearray()
    for _ in range(count):
        if offset + 4 > len(payload):
            raise ValueError('Truncated block header')
        codec = payload[offset]
        size = int.from_bytes(payload[offset + 1:offset + 4], 'little')
        offset += 4
        stride = None
        if version2 and codec & 128:
            codec &= 127
            if offset + 2 > len(payload):
                raise ValueError('Truncated transform descriptor')
            transform_id, stride = payload[offset:offset+2]
            offset += 2
            if transform_id != 1 or not stride:
                raise ValueError('Unknown block transform')
        if not size or offset + size > len(payload):
            raise ValueError('Truncated or empty block')
        data = payload[offset:offset + size]
        offset += size
        if codec == 1:
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW,
                filters=[{'id': lzma.FILTER_LZMA2, 'dict_size': 1 << 20}])
            data = decoder.decompress(data, max_length=expected - len(output) + 1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError('Invalid LZMA2 stream boundary')
        elif codec == 2 and version2:
            import brotli
            data = brotli.decompress(data)
        elif codec != 0:
            raise ValueError('Unknown block codec')
        if stride is not None:
            data = transpose(data, stride, inverse=True)
        output.extend(data)
        if len(output) > expected:
            raise ValueError('Output exceeds declared model size')
    if len(output) != expected or offset == len(payload):
        raise ValueError('Incomplete models or missing residual/token stream')
    return bytes(output), payload[offset:]
