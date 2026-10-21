"""A self-described reversible grouping of alternating metadata/data spans.

All segment lengths are stored in the archive. No tensor shapes or learned
values are required by the decoder.
"""
import struct

MAGIC=b'LAY1'
MAX_SEGMENTS=4096

def pack_segments(segments):
    if not 0<len(segments)<=MAX_SEGMENTS or any(len(s)>65535 for s in segments):
        raise ValueError('Invalid segment count or size')
    return MAGIC+struct.pack('<H',len(segments))+struct.pack('<'+'H'*len(segments),*map(len,segments))+b''.join(segments[::2])+b''.join(segments[1::2])

def unpack_segments(blob):
    if len(blob)<6 or blob[:4]!=MAGIC:raise ValueError('Invalid segment layout')
    count=struct.unpack_from('<H',blob,4)[0]
    if not 0<count<=MAX_SEGMENTS or len(blob)<6+2*count:raise ValueError('Invalid segment count')
    sizes=struct.unpack_from('<'+'H'*count,blob,6)
    offset=6+2*count
    if offset+sum(sizes)!=len(blob):raise ValueError('Segment length mismatch')
    parts=[None]*count
    for parity in (0,1):
        for i in range(parity,count,2):
            parts[i]=blob[offset:offset+sizes[i]]
            offset+=sizes[i]
    return b''.join(parts)
