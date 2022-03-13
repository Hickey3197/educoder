import numpy as np
import struct
import CreateMatrix
from WriteMatrix import writeMatrixList
type2code_dict = {'uint8':0x08,'int8':0x09,'int16':0x0B,'int32':0x0C,'float32':0x0D,'float64':0x0E}
code2type_dict = {0x08: 'B', 0x09: 'b', 0x0B: 'h', 0x0c: 'i', 0x0D: 'f', 0x0E: 'd'}

def readMatrixFromFile(filename):
    with open(filename,'rb') as f:
        data_buf = f.read()
        off_set = 0
        file_head_fmt = '>HBB'#格式定义，>表示高位在前，I表示4字节整数
        _,dcode,dimslen = struct.unpack_from(file_head_fmt,data_buf,off_set)
        off_set += struct.calcsize(file_head_fmt)

        file_head_fmt = '>{}I'.format(dimslen)
        shapes = struct.unpack_from(file_head_fmt,data_buf,off_set)
        off_set += struct.calcsize(file_head_fmt)

        data_fmt = '>'+ str(np.prod(shapes)) + code2type_dict[dcode]
        matrix_list = struct.unpack_from(data_fmt,data_buf,off_set)
        matrix_list = np.reshape(matrix_list,shapes)

    return matrix_list
    
if __name__ == '__main__':
    matrix_list = CreateMatrix.createMatrixList(len=10,rows=2,cols=3)
    writeMatrixList('/data/workspace/myshixun/creatematrix/data/data.idx3.ubyte',matrix_list)
    print('read')

    matrix_list = readMatrixFromFile('/data/workspace/myshixun/creatematrix/data/data.idx3.ubyte')
    print(matrix_list)