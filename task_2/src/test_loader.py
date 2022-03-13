rom loader import dataReader
if __name__ == '__main__':
    BATCH_SIZE = 5
    test_loader = dataReader('/data/workspace/myshixun/mnist/data/t10k-images-idx3-ubyte', '/data/workspace/myshixun/mnist/data/t10k-labels-idx1-ubyte', BATCH_SIZE, False)
    for data in test_loader:
        images, labels = zip(*data)
        print(len(images),images[0].shape,type(images[0]),len(labels),labels[0].shape,type(labels[0]))
        break