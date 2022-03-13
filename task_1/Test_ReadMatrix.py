import numpy as np
import CreateMatrix
import WriteMatrix
import ReadMatrix
if __name__ == '__main__':
    test = input().split(' ')
    length,rows,cols=list(map(int, test))
    matrix_list = CreateMatrix.createMatrixList(length,rows,cols)
    WriteMatrix.writeMatrixList('/data/workspace/myshixun/creatematrix/data/data.idx3.ubyte',matrix_list)
    matrix_list = ReadMatrix.readMatrixFromFile('/data/workspace/myshixun/creatematrix/data/data.idx3.ubyte')
    print(type(matrix_list))
    print(matrix_list)