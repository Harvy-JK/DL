import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

data_path = 'hour.csv'
rides = pd.read_csv(data_path)
rides.head()
counts = rides['cnt'][:50]
x = np.arange(len(counts))
y = np.array(counts)
plt.figure(figsize=(10, 7))
plt.plot(x, y, 'o-')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
counts = rides['cnt'][:50]
# 输入变量，1，2，3...这样的一维数组
x = torch.tensor(np.arange(len(counts), dtype=float) / len(counts), requires_grad=True)
# 输出变量，它是从数据counts中读取的每一辆车的行驶时间，共100个数据点的一维数组，作为标准答案
y = torch.tensor(np.array(counts, dtype=float), requires_grad=True)

sz = 10  # 设置隐藏层的神经元的数量
weights = torch.randn((1, sz), dtype=torch.double, requires_grad=True)  # 初始化输入层到隐藏层的权重矩阵，它的尺寸是（1，10）
biases = torch.randn(sz, dtype=torch.double, requires_grad=True)  # 初始化隐藏层节点的偏置向量，它是尺寸为10的一位向量
weights2 = torch.randn((sz, 1), dtype=torch.double, requires_grad=True)  # 初始化隐藏层到输出层的权重矩阵，它的尺寸是（10，1）

learning_rate = 0.0001  # 学习率，针对权重w与变差b更新的步长
losses = []
x = x.view(50, -1)
y = y.view(50, -1)
for i in range(1000000):
    hidden = x * weights + biases
    hidden = torch.sigmoid(hidden)
    predictions = hidden.mm(weights2)
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    if i % 10000 == 0:
        print('loss', loss)

    # ****************************************************************************************
    # 接下来开始梯度下降算法，将误差反向传播
    loss.backward()  # 对损失函数进行梯度反传
    # 利用上一步计算中得到的weights，biases等梯度信息更新weights或biases的数值
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)

    # 清空所有变量的梯度值
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

x_data = x.data.numpy()  # 获得x包裹的数据
plt.figure(figsize=(10, 7))  # 设定绘图窗口大小
xplot, = plt.plot(x_data, y.data.numpy(), 'o')  # 绘制原始数据
yplot, = plt.plot(x_data, predictions.data.numpy())  # 绘制拟合数据
plt.xlabel('X')  # 更改坐标轴标注
plt.ylabel('Y')  # 更改坐标轴标注
plt.legend([xplot, yplot], ['Data', 'Prediction under 500000 epochs'])  # 绘制图例
plt.show()

counts_predict = rides['cnt'][50:100]  # 读取待预测的接下来的50个数据点

# 首先对接下来的50个数据点进行选取，注意x应该取51，52，……，100，然后再归一化
x = torch.tensor((np.arange(50, 100, dtype=float) / len(counts)), requires_grad=True)
# 读取下50个点的y数值，不需要做归一化
y = torch.tensor(np.array(counts_predict, dtype=float), requires_grad=True)

x = x.view(50, -1)
y = y.view(50, -1)

# 从输入层到隐含层的计算
hidden = x * weights + biases

# 将sigmoid函数作用在隐含层的每一个神经元上
hidden = torch.sigmoid(hidden)

# 隐含层输出到输出层，计算得到最终预测
predictions = hidden.mm(weights2)

# 计算预测数据上的损失函数
loss = torch.mean((predictions - y) ** 2)
print(loss)

x_data = x.data.numpy()  # 获得x包裹的数据
plt.figure(figsize=(10, 7))  # 设定绘图窗口大小
xplot, = plt.plot(x_data, y.data.numpy(), 'o')  # 绘制原始数据
yplot, = plt.plot(x_data, predictions.data.numpy())  # 绘制拟合数据
plt.xlabel('X')  # 更改坐标轴标注
plt.ylabel('Y')  # 更改坐标轴标注
plt.legend([xplot, yplot], ['Data', 'Prediction'])  # 绘制图例
plt.show()

