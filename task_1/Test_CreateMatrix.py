from CreateMatrix import createMatrixList

if __name__ == '__main__':
    test = input().split(' ')
    length,rows,cols=list(map(int, test))
    matrix_list = createMatrixList(length,rows,cols)
    print(matrix_list)