import numpy as np
import struct
from CreatMatrix import createMatrixList
from WriteMatrix import writeMatrixList
from BatchReader import trainBatchReader

if __name__ == '__main__':
    *test,batch,drop = input().split(' ')
    print(test,batch,drop)
    length,rows,cols=list(map(int, test))
    matrix_list = createMatrixList(length,rows,cols)
    writeMatrixList('/data/workspace/myshixun/creatematrix/data/data.idx3.ubyte',matrix_list)
    reader = trainBatchReader(idx_filename='/data/workspace/myshixun/creatematrix/data/data.idx3.ubyte',batch_size=int(batch), drop_last=(drop=='True'))
    print(type(reader))
    for id, data in enumerate(reader):
        print(id,data)