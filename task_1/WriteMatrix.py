import numpy as np
import struct

def writeMatrixList(filename,matrix_list):
    matrixData = [2051]
    matrixData.extend(matrix_list.shape)
    matrixData.extend(matrix_list.flatten())
    index = 0
    with open(filename,'wb') as f:
        for data in matrixData:
            if index < 4:
                s = struct.pack('>i', data)
                f.write(s)
            else:
                s = struct.pack('>B', data)
                f.write(s)
            index +=1