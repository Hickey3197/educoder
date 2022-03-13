########1.加载数据
import struct,random,numpy as np
code2type = {0x08: 'B', 0x09: 'b', 0x0B: 'h', 0x0c: 'i', 0x0D: 'f', 0x0E: 'd'}
def readMatrix(filename):
    with open(filename,'rb') as f:
        buff = f.read()
        offset = 0
        fmt = '>HBB'#格式定义，>表示高位在前，I表示4字节整数
        _,dcode,dimslen = struct.unpack_from(fmt,buff,offset)
        offset += struct.calcsize(fmt)

        fmt = '>{}I'.format(dimslen)
        shapes = struct.unpack_from(fmt,buff,offset)
        offset += struct.calcsize(fmt)

        fmt = '>'+ str(np.prod(shapes)) + code2type[dcode]
        matrix = struct.unpack_from(fmt,buff,offset)
        matrix = np.reshape(matrix,shapes).astype(code2type[dcode])

    return matrix

def dataReader(imgfile, labelfile, batch_size, drop_last):
    pass