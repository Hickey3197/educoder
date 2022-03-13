import numpy as np
import struct
def trainBatchReader(idx_filename, batch_size, drop_last):
    data = open(idx_filename, 'rb').read()
    off_set = 0
    header = '>iiii'
    magic_number, num_matrix, rows, cols = struct.unpack_from(header, data, off_set)
    off_set += struct.calcsize(header)
    fmt_Matrix = '>{}B'.format(num_matrix*rows*cols)
    matrix_list = struct.unpack_from(fmt_Matrix, data, off_set)
    matrix_list = np.reshape(matrix_list, (num_matrix, rows, cols))
    num_batch= 0
    if drop_last:
        num_batch = num_matrix // batch_size
    else:
        if(num_matrix % batch_size == 0):
            num_batch = num_matrix // batch_size
        else:
            num_batch = num_matrix // batch_size + 1
    for i in range(num_batch):
        matrix_batch = matrix_list[batch_size*i:batch_size*(i+1)]
        yield list(matrix_batch)