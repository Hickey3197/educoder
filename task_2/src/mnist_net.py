import numpy as np
import torch
import torch.nn as nn,torch.nn.functional as F,torch.optim as optim
from loader import dataReader


#########2.定义卷积神经网络
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
            pass



    def forward(self, x):
        pass


# 3.训练网络
def train(loader):
    model = MnistNet()
    ######## 让我们使用分类交叉熵损失和带有动量的 SGD。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(loader):
            inputs, labels = zip(*data)
            inputs = np.array(inputs).astype('float32')
            labels = np.array(labels).astype('int64')
            inputs = torch.from_numpy(inputs).unsqueeze(1)#扩展通道维度 NCHW
            labels = torch.from_numpy(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
            if i==199:
                break
    print('Finished Training')
    return model

# 4.测试网络
def test(PATH,loader):
    # 让我们重新加载保存的模型
    model = MnistNet()
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = zip(*data)
            images = np.array(images).astype('float32')
            labels = np.array(labels).astype('int64')
            images = torch.from_numpy(images).unsqueeze(1)
            labels = torch.from_numpy(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)#torch.argmax
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {:d} test images: {:f}%'.format(total,100 * correct / total))
    return model

if __name__ == '__main__':
    BATCH_SIZE = 16
    PATH = '/data/workspace/myshixun/mnist/model/mnist_model.pth'
    train_loader = dataReader('/data/workspace/myshixun/mnist/data/train-images-idx3-ubyte', '/data/workspace/myshixun/mnist/data/train-labels-idx1-ubyte', BATCH_SIZE, True)
    test_loader = dataReader('/data/workspace/myshixun/mnist/data/t10k-images-idx3-ubyte', '/data/workspace/myshixun/mnist/data/t10k-labels-idx1-ubyte', BATCH_SIZE, False)
    model = train(train_loader)
    #快速保存我们训练过的模型：
    torch.save(model.state_dict(), PATH)
    test(PATH,test_loader)                                                                   