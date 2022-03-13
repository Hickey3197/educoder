import numpy as np
import struct
def readMatrixFromFile(filename):
    data = open(filename, 'rb').read()
    off_set = 0
    header = '>iiii'
    magic_number, num_matrix, rows, cols = struct.unpack_from(header, data, off_set)
    off_set += struct.calcsize(header)
    fmt_Matrix = '>{}B'.format(num_matrix*rows*cols)
    matrix_list = struct.unpack_from(fmt_Matrix, data, off_set)
    matrix_list = np.reshape(matrix_list, (num_matrix, rows, cols))
    return matrix_list