"""Reversible RGB prediction for the RR5 carrier basis; MIT licensed."""
import math
import struct

from renderer_pack import Reader, Writer

HEADER = struct.Struct('<4sHBBBBHH')
PREFIX = 142


def cumulative(counts):
    result = [0]
    for count in counts:
        result.append(result[-1]+count)
    return result


def rounded(value, fractional_bits):
    half = 1 << (fractional_bits-1)
    return (value+half) >> fractional_bits if value >= 0 else -((-value+half) >> fractional_bits)


def zigzag(value):
    return 2*value if value >= 0 else -2*value-1


def unzigzag(value):
    return value//2 if not value & 1 else -(value//2)-1


def read_original(body, atoms, planes, height, width, bits):
    if len(body) < PREFIX:
        raise ValueError('Truncated carrier')
    basis_size = (int.from_bytes(body[:3], 'little')+7)//8
    residue_size = (int.from_bytes(body[3:6], 'little')+7)//8
    if not basis_size or not residue_size or PREFIX+basis_size+residue_size > len(body):
        raise ValueError('Carrier fields exceed input length')
    basis = body[PREFIX:PREFIX+basis_size]
    reader = Reader(basis)
    values = []
    for _ in range(atoms):
        counts = [1]*(1 << bits)
        atom = []
        for _ in range(planes):
            plane = []
            for _ in range(height*width):
                symbol = reader.get(cumulative(counts))
                counts[symbol] += 32
                plane.append(unzigzag(symbol))
            atom.append(plane)
        values.append(atom)
    if write_original(values, bits) != basis:
        raise ValueError('Carrier basis is not the canonical RR5 stream')
    return values, basis_size


def write_original(values, bits):
    writer = Writer()
    for atom in values:
        counts = [1]*(1 << bits)
        for plane in atom:
            for value in plane:
                symbol = zigzag(value)
                limits = cumulative(counts)
                writer.put(limits[symbol], limits[symbol+1], limits[-1])
                counts[symbol] += 32
    return writer.finish()


def make_residuals(atom, plane, weights, fractional_bits, bits):
    half = 1 << (bits-1)
    residuals = []
    for index, value in enumerate(atom[plane]):
        prediction = rounded(sum(weight*atom[p][index] for p, weight in enumerate(weights)), fractional_bits) if weights else 0
        residual = ((value-prediction+half) & ((1 << bits)-1))-half
        residuals.append(zigzag(residual))
    return residuals


def cost(values, increment, bits):
    counts, total, result = [1]*(1 << bits), 1 << bits, 0.0
    for value in values:
        result += math.log2(total/counts[value])
        counts[value] += increment
        total += increment
    return result


def fitted_weights(atom, plane, fractional_bits):
    unit = 1 << fractional_bits
    target = atom[plane]
    if plane == 1:
        x = atom[0]
        denominator = sum(a*a for a in x)
        fitted = [round(unit*sum(a*b for a,b in zip(x,target))/denominator) if denominator else 0]
    else:
        x, y = atom[:2]
        xx = sum(a*a for a in x)
        xy = sum(a*b for a,b in zip(x,y))
        yy = sum(a*a for a in y)
        xz = sum(a*b for a,b in zip(x,target))
        yz = sum(a*b for a,b in zip(y,target))
        denominator = xx*yy-xy*xy
        fitted = [round(unit*(xz*yy-yz*xy)/denominator), round(unit*(yz*xx-xz*xy)/denominator)] if denominator else [0,0]
    return [min(127,max(-128,value)) for value in fitted]


def search(values, fractional_bits, bits):
    weights, residuals = [], []
    for atom in values:
        atom_weights = []
        for plane in range(3):
            if plane:
                fitted = fitted_weights(atom, plane, fractional_bits)
                if plane == 1:
                    candidates = [(a,) for a in range(max(-128,fitted[0]-12),min(127,fitted[0]+12)+1)]
                else:
                    candidates = [(a,b) for a in range(max(-128,fitted[0]-5),min(127,fitted[0]+5)+1) for b in range(max(-128,fitted[1]-5),min(127,fitted[1]+5)+1)]
                trial = min((cost(make_residuals(atom,plane,candidate,fractional_bits,bits),3,bits),candidate) for candidate in candidates)
                selected = trial[1]
                atom_weights.extend(selected)
            else:
                selected = []
            residuals.append(make_residuals(atom,plane,selected,fractional_bits,bits))
        weights.extend(atom_weights)
    increment = min((sum(cost(row,candidate,bits) for row in residuals),candidate) for candidate in range(1,65))[1]
    return weights, increment, residuals


def encode(body):
    atoms, planes, height, width, bits, fractional_bits = 12, 3, 24, 32, 5, 6
    values, old_size = read_original(body,atoms,planes,height,width,bits)
    weights, increment, residuals = search(values,fractional_bits,bits)
    writer = Writer()
    for row in residuals:
        counts = [1]*(1 << bits)
        for symbol in row:
            limits = cumulative(counts)
            writer.put(limits[symbol],limits[symbol+1],limits[-1])
            counts[symbol] += increment
    packed = struct.pack('<'+'b'*len(weights),*weights)+bytes([increment])+writer.finish()
    if len(packed) > 65535:
        raise ValueError('Carrier stream too large')
    result = (HEADER.pack(b'CBP1',len(packed),atoms,planes,bits,fractional_bits,height,width)
              +body[:PREFIX]+packed+body[PREFIX+old_size:])
    if decode(result) != body:
        raise ValueError('Carrier round trip differs')
    return result


def decode(data):
    if len(data) < HEADER.size+PREFIX:
        raise ValueError('Truncated packed carrier')
    magic,size,atoms,planes,bits,fractional_bits,height,width = HEADER.unpack_from(data)
    if magic != b'CBP1' or not 1 <= atoms <= 32 or planes != 3 or not 1 <= bits <= 8 or not 1 <= fractional_bits <= 7 or not 1 <= height <= 128 or not 1 <= width <= 128:
        raise ValueError('Invalid carrier geometry')
    start = HEADER.size+PREFIX
    weight_count = atoms*3
    if size <= weight_count+1 or start+size >= len(data):
        raise ValueError('Invalid carrier stream length')
    prefix = data[HEADER.size:start]
    weights = struct.unpack_from('<'+'b'*weight_count,data,start)
    increment = data[start+weight_count]
    if not increment:
        raise ValueError('Invalid carrier increment')
    reader = Reader(data[start+weight_count+1:start+size])
    values = []
    half, mask = 1 << (bits-1), (1 << bits)-1
    for atom_index in range(atoms):
        atom = []
        for plane in range(planes):
            selected = [] if plane == 0 else weights[atom_index*3:atom_index*3+1] if plane == 1 else weights[atom_index*3+1:atom_index*3+3]
            counts, row = [1]*(1 << bits), []
            for index in range(height*width):
                symbol = reader.get(cumulative(counts))
                counts[symbol] += increment
                prediction = rounded(sum(weight*atom[p][index] for p,weight in enumerate(selected)),fractional_bits) if selected else 0
                value = ((unzigzag(symbol)+prediction+half)&mask)-half
                row.append(value)
            atom.append(row)
        values.append(atom)
    original_basis = write_original(values,bits)
    expected = (int.from_bytes(prefix[:3],'little')+7)//8
    if len(original_basis) != expected:
        raise ValueError('Restored basis length differs')
    return prefix+original_basis+data[start+size:]
