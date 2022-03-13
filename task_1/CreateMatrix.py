import numpy as np
import struct

def createMatrixList(len,rows,cols):
    MatrixList = np.zeros((len,rows,cols),dtype=np.int8)
    for i in range(len):
        for j in range(rows):
            for k in range(cols):
               MatrixList[i][j][k] = (i+1)*10+(j)*cols+(k+1)
    return MatrixList