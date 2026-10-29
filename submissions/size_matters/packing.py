import hashlib
import struct

import brotli
import carrier_pack
import tail_pack
import hpac_codec
import renderer_pack


HEADER = struct.Struct('<4sHHHBB')


def split(data):
    if len(data) < 27 or data[:4] != b'BLK2':
        raise ValueError('Unexpected payload format')
    size, count = struct.unpack_from('<IB', data, 4)
    block_size = int.from_bytes(data[10:13], 'little')
    if count != 1 or data[9] != 0 or size != block_size:
        raise ValueError('Expected one uncompressed model block')
    end = 13 + size
    if end >= len(data):
        raise ValueError('Truncated model block')
    models = data[13:end]
    if models[:6] != b'RX1M\x01\x02':
        raise ValueError('Unexpected model format')
    lengths = struct.unpack_from('<HHH', models, 8)
    if 14 + sum(lengths) != len(models) or not min(lengths):
        raise ValueError('Model lengths do not match')
    parts, offset = [], 14
    for length in lengths:
        parts.append(models[offset:offset + length])
        offset += length
    return models[:8], parts, data[end:]


def join(header, parts, vid):
    lengths = tuple(map(len, parts))
    if not min(lengths) or max(lengths) > 65535:
        raise ValueError('Invalid model section size')
    models = header + struct.pack('<HHH', *lengths) + b''.join(parts)
    return (b'BLK2' + struct.pack('<IB', len(models), 1) + b'\x00'
            + len(models).to_bytes(3, 'little') + models + vid)


def component_hashes(data):
    header, parts, vid = split(data)
    components = {
        'model_header': header,
        'hpac_ihs1': brotli.decompress(parts[0]),
        'semantic_layout': parts[1],
        'carrier': parts[2],
        'residual_tokens_warp': vid,
    }
    return {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
            for name, value in components.items()}


def encode(data):
    header, parts, vid = split(data)
    semantic = parts[1]
    if semantic[:6] != b'LAY1\x01\x00' or len(semantic) < 8:
        raise ValueError('Expected one semantic segment')
    if int.from_bytes(semantic[6:8], 'little') != len(semantic) - 8:
        raise ValueError('Invalid semantic segment length')
    hpac = hpac_codec.encode(brotli.decompress(parts[0]))
    renderer = renderer_pack.encode(semantic[8:])
    carrier = carrier_pack.encode(parts[2])
    vid = tail_pack.encode(vid)
    sizes = len(hpac), len(renderer), len(carrier)
    if not min(sizes) or max(sizes) > 65535:
        raise ValueError('Invalid packed section size')
    return (HEADER.pack(b'SMP3', *sizes, header[6], header[7])
            + hpac + renderer + carrier + vid)


def decode(data):
    if len(data) < HEADER.size:
        raise ValueError('Truncated packed header')
    magic, hpac_size, renderer_size, carrier_size, table_mode, reserved = HEADER.unpack_from(data)
    if magic != b'SMP3' or not min(hpac_size, renderer_size, carrier_size):
        raise ValueError('Unexpected packed format')
    end = HEADER.size + hpac_size + renderer_size + carrier_size
    if end >= len(data):
        raise ValueError('Truncated packed sections')
    offset = HEADER.size
    hpac = hpac_codec.decode(data[offset:offset + hpac_size])
    offset += hpac_size
    semantic = renderer_pack.decode(data[offset:offset + renderer_size])
    offset += renderer_size
    carrier = carrier_pack.decode(data[offset:offset + carrier_size])
    vid = tail_pack.decode(data[end:])
    layout = b'LAY1\x01\x00' + struct.pack('<H', len(semantic)) + semantic
    header = b'RX1M\x01\x02' + bytes([table_mode, reserved])
    return join(header, [brotli.compress(hpac, quality=11), layout, carrier], vid)
