import struct
from hpac_range import Model, Encoder, Decoder, code_int

HEADER = struct.Struct('<BH2s3s4sH')


def unpack(data, count, bits):
    result=[]
    for i in range(count):
        position=i*bits
        value=(int.from_bytes(data[position//8:position//8+3], 'little') >> (position%8)) & ((1<<bits)-1)
        result.append(value-(1<<bits) if value >= 1<<(bits-1) else value)
    return result


def pack(values,bits):
    result=bytearray((len(values)*bits+7)//8)
    for i,value in enumerate(values):
        value &= (1<<bits)-1
        for j in range(bits):
            result[(i*bits+j)//8] |= ((value>>j)&1)<<((i*bits+j)%8)
    return bytes(result)


def varint(n):
    result=bytearray()
    while n>=128:
        result.append((n&127)|128);n>>=7
    result.append(n)
    return bytes(result)


def readvar(data,offset):
    value=shift=0
    while offset<len(data) and shift<35:
        b=data[offset];offset+=1;value|=(b&127)<<shift
        if not b&128:return value,offset
        shift+=7
    raise ValueError('Invalid integer')


def codes(header):
    lengths=[]
    for i in range(5):
        length=(header[i//2]>>(4 if i%2==0 else 0))&15
        if length:lengths.append((length,i))
    lengths.sort();result={};value=previous=0
    for length,symbol in lengths:
        value<<=length-previous
        result[symbol]=format(value, f'0{length}b')
        value+=1;previous=length
    return result


def warp_decode(data):
    table={v:k for k,v in codes(data[:3]).items()}
    offset=24;result=[]
    for _ in range(600):
        word=''
        while word not in table:
            word+=str((data[offset//8]>>(7-offset%8))&1);offset+=1
            if len(word)>15:raise ValueError('Invalid warp')
        result.append(table[word])
    return result,(offset+7)//8


def warp_encode(values,header):
    mapping=codes(header);bits=''.join(mapping[v] for v in values)
    bits+='0'*(-len(bits)%8)
    return header+bytes(int(bits[i:i+8],2) for i in range(0,len(bits),8))


def split(data):
    extended_size=int.from_bytes(data[-2:],'little')
    extended=data[-extended_size-2:-2]
    main=data[:-extended_size-2]
    warp,warp_size=warp_decode(extended)
    rawwarp=extended[:warp_size]
    if warp_encode(warp,rawwarp[:3])!=rawwarp:raise ValueError('Noncanonical warp')
    extension=extended[warp_size:-2]
    if int.from_bytes(extended[-2:],'little')!=len(extension):raise ValueError('Invalid extension size')
    if extension[0]!=1:raise ValueError('Expected grid4')
    bits,step,count=extension[1:4]
    position=4;ids=[];previous=0
    for _ in range(count):
        delta,position=readvar(extension,position);previous+=delta;ids.append(previous)
    size=(count*16*bits+7)//8
    grids=unpack(extension[position:position+size], count*16,bits)
    previous2=extension[position+size:]
    if previous2[0]!=0x31:raise ValueError('Expected shared previous2')
    return main[:2],unpack(main[2:96],125,6),main[96:],rawwarp[:3],warp,extension[:4],ids,grids,previous2


def transform(c,settings,coeffs=None,warp=None,ids=None,grids=None,gridcount=0,bits=4):
    limit,resmode,warpmode,gridmode=settings
    model=Model(limit);enc=isinstance(c,Encoder)
    if not enc:coeffs=[0]*125;warp=[0]*600;ids=[0]*gridcount;grids=[0]*(16*gridcount)
    for i in range(125):
        predictor=coeffs[i-5] if resmode>=2 and i>=5 else 0
        context=('r',i%5 if resmode%2 else 0)
        coeffs[i]=predictor+code_int(c,model,context,coeffs[i]-predictor if enc else 0,7 if resmode>=2 else 6)
    last=last2=run=0
    for i in range(600):
        if warpmode==0:context=('w',)
        elif warpmode==1:context=('w',last)
        elif warpmode==2:context=('w',last,min(run.bit_length(),4))
        elif warpmode==3:context=('w',min(run.bit_length(),4))
        elif warpmode==4:context=('w',last,last2)
        elif warpmode==5:context=('w',i//64)
        elif warpmode==6:context=('w',i//128,last)
        else:context=('w',i//256,last)
        zero=c.bit(model,(context,'z'),int(warp[i]==0) if enc else 0)
        value=0
        if not zero:
            for b in [1,0]:
                value|=c.bit(model,(context,'v',b,value>>(b+1)),((warp[i]-1)>>b)&1 if enc else 0)<<b
            value+=1
        warp[i]=value
        run=run+1 if value==last else 1;last2,last=last,value
    previous=0
    for i in range(gridcount):
        value=code_int(c,model,('id',),ids[i]-previous if enc else 0)
        ids[i]=previous+value;previous=ids[i]
    for i in range(len(grids)):
        local=i%16
        if gridmode==0:predictor=0
        elif gridmode==1:predictor=grids[i-1] if local else 0
        elif gridmode==2:predictor=grids[i-4] if local>=4 else 0
        else:
            left=grids[i-1] if local%4 else 0
            up=grids[i-4] if local>=4 else 0
            predictor=(left+up)//2
        grids[i]=predictor+code_int(c,model,('g',),grids[i]-predictor if enc else 0,bits+1 if gridmode else bits)
    return coeffs,warp,ids,grids


def encode(data,settings=(16, 0, 1, 0)):
    scale,coeffs,tokens,warphdr,warp,gridhdr,ids,grids,previous2=split(data)
    encoder=Encoder();transform(encoder,settings,coeffs,warp,ids,grids,len(ids),gridhdr[1]);coded=encoder.finish()
    limit,resmode,warpmode,gridmode=settings
    descriptor=limit | (resmode<<8) | (warpmode<<10) | (gridmode<<13)
    return HEADER.pack(len(previous2),len(coded),scale,warphdr,gridhdr,descriptor)+tokens+previous2+coded


def decode(data):
    if len(data)<HEADER.size:raise ValueError('Truncated tail')
    np2,nc,scale,warphdr,gridhdr,descriptor=HEADER.unpack_from(data)
    nt=len(data)-HEADER.size-np2-nc
    settings=(descriptor&255,(descriptor>>8)&3,(descriptor>>10)&7,(descriptor>>13)&3)
    if nt<=0 or descriptor>>15:raise ValueError('Invalid tail')
    pos=HEADER.size;tokens=data[pos:pos+nt];pos+=nt;previous2=data[pos:pos+np2];pos+=np2
    coeffs,warp,ids,grids=transform(Decoder(data[pos:]),settings,gridcount=gridhdr[3],bits=gridhdr[1])
    extension=gridhdr;previous=0
    for value in ids:extension+=varint(value-previous);previous=value
    extension+=pack(grids,gridhdr[1])+previous2
    extended=warp_encode(warp,warphdr)+extension+len(extension).to_bytes(2,'little')
    return scale+pack(coeffs,6)+tokens+extended+len(extended).to_bytes(2,'little')
