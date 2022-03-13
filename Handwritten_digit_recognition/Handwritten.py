import numpy
import scipy.special
import matplotlib.pyplot
import imageio
import glob

# 神经网络类定义
class neuralNetwork:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置每个输入、隐藏、输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 链接权值矩阵，wih and who
        # 数组中的权重是w_i_j，其中链路是从节点i到下一层的节点j
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 学习速率
        self.lr = learningrate

        # 激活函数是s型函数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算信号到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐含层出现的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算信号到最终的输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层出现的信号
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差为(目标值-实际值)
        output_errors = targets - final_outputs
        # 隐藏层错误是output_errors，按权重分割，在隐藏节点处重新组合
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层之间的链接的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 更新输入层和隐藏层之间的链接的权值
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # 查询神经网络
    def query(self, inputs_list):
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算信号到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐含层出现的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算信号到最终的输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层出现的信号
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# 输入、隐藏和输出节点的数量
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# 学习速率
learning_rate = 0.1

# 创建神经网络实例
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# 将mnist训练数据CSV文件加载到列表中
training_data_file = open("MNIST_data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练神经网络

# epochs是训练数据集用于训练的次数
epochs = 10

for e in range(epochs):
    # 检查训练数据集中的所有记录
    for record in training_data_list:
        # 用逗号分隔记录
        all_values = record.split(',')
        # 规模和转移输入
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 创建目标输出值(都是0.01，除了所需的标签为0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0]是该记录的目标标签
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# 测试神经网络

# 将mnist测试数据csv文件加载到列表中
test_data_file = open("MNIST_data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 记录神经网络执行情况，最初为空
scorecard = []
# 遍历测试数据集中的所有记录
for record in test_data_list:
    all_values = record.split(',')
    # 正确答案为第一个值
    correct_label = int(all_values[0])
    # 规模和转移输入
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # 查询神经网络
    outputs = n.query(inputs)
    # 最大值的索引对应于标签
    label = numpy.argmax(outputs)
    # print("Answer label is:",correct_label," ; ",label," is network's answer")
    # 判断结果正不正确
    if (label == correct_label):
        # 神经网络的测试结果与正确结果匹配，如果正确scorecard加1
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

# 计算准确率
scorecard_array = numpy.asarray(scorecard)
print("准确率为：", scorecard_array.sum() / scorecard_array.size)


# 用自己写的的图像测试数据集
our_own_dataset = []

# 加载png图像数据作为测试数据集
for image_file_name in glob.glob('Number4.png'):
    # 使用文件名设置正确的标签
    label = int(image_file_name[-5:-4])
    # 将png文件图像转为数组
    print("加载文件：", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    # 每张图片都由一个28 ×28 的矩阵表示，每张图片都由一个784 维的向量表示（28*28=784）
    # 将数组的值减去了255.0。常规而言，0指的是黑色，255指的是白色，但是，MNIST数据集使用相反的方式表示，因此将值逆转过来以匹配MNIST数据
    # 从28x28重塑到包含784个值的列表，反转值
    img_data = 255.0 - img_array.reshape(784)
    # 然后将数据缩放到范围从0.01到1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    # 附加标签和图像数据来测试数据集
    record = numpy.append(label, img_data)
    our_own_dataset.append(record)
    pass

# 用我们自己的图像来测试神经网络

# 记录测试
item = 0
# plot image
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')
# 正确答案为第一个值
correct_label = our_own_dataset[item][0]
# 数据是剩余值
inputs = our_own_dataset[item][1:]
# 查询神经网络
outputs = n.query(inputs)
print (outputs)
# 最大值的索引对应于标签
label = numpy.argmax(outputs)
print("神经网络测试结果：", label)
# 判断结果正不正确
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass

