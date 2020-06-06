import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time

# ****************************************
# 先对数据进行预处理
data_path = 'hour.csv'
rides = pd.read_csv(data_path)
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
print(data, end='\n\n')

# 数值类型变量的处理，归一化
quant_features = ['cnt', 'temp', 'hum', 'windspeed']  # 需要归一化处理的数值型变量
scaled_features = {}  # 字典，储存每个变量的[均值，方差]
for each in quant_features:  # 依次读取选择的“类”
    mean, std = data[each].mean(), data[each].std()  # 计算“类”的均值与方差
    scaled_features[each] = [mean, std]  # 将计算后的[均值，方差]存储到字典中
    data.loc[:, each] = (data[each] - mean) / std  # 归一化公式，使数值型变量区间[-1，1]

# 训练集，测试集
train_data = data[:-21 * 24]
test_data = data[-21 * 24:]

target_fields = ['cnt', 'casual', 'registered']
features, targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
print(features, end='\n\n')
print(targets, end='\n\n')

# 将数据从pandas dataframe转换为numpy
X = features.values
print(X, end=('\n\n'))
Y = targets['cnt'].values
Y = Y.astype(float)
Y = np.reshape(Y, [len(Y), 1])  # 使Y数据变为（16875，1）维

input_size = features.shape[1]  # data.shape[0或1]，0是读行数，1是读列数。
hidden_size = 10  # 隐藏层数
output_size = 1  # 输出层数
batch_size = 128  # 设立batch数，一撮数据，2^n。

tic = time.time()
# 权重与偏差
weights1 = torch.randn([input_size, hidden_size], dtype=torch.double, requires_grad=True)
biases1 = torch.randn([hidden_size], dtype=torch.double, requires_grad=True)
weights2 = torch.randn([hidden_size, output_size], dtype=torch.double, requires_grad=True)


def neu(x):
    hidden = x.mm(weights1) + biases1.expand(x.size()[0], hidden_size)
    hidden = torch.sigmoid(hidden)
    output = hidden.mm(weights2)
    return output


def cost(x, y):
    error = torch.mean((x - y) ** 2)
    return error


def zero_grad():
    if weights1.grad is not None and biases1.grad is not None and weights2.grad is not None:
        weights1.grad.data.zero_()
        weights2.grad.data.zero_()
        biases1.grad.data.zero_()


def optimizer_step(learning_rate):
    weights1.data.add_(-learning_rate * weights1.grad.data)
    weights2.data.add(-learning_rate * weights2.grad.data)
    biases1.data.add(-learning_rate * biases1.grad.data)


losses = []
for i in range(1000):  # 1000次迭代
    batch_loss = []  # batch_loss
    # range(0,16875,128)，范围（0，16875），每次跳跃128
    for start in range(0, len(X), batch_size):
        # 切分数据集
        end = start + batch_size if start + batch_size < len(X) else len(X)
        # xx,yy分别为特征变量的一批数据，目标变量的一批数据
        xx = torch.tensor(X[start:end], dtype=torch.double, requires_grad=True)
        yy = torch.tensor(Y[start:end], dtype=torch.double, requires_grad=True)
        prediction = neu(xx)  # 计算正向传播的预测值
        loss = cost(prediction, yy)  # 计算成本函数
        zero_grad()  # 先对权重以及偏差的求导值清零
        loss.backward()  # 这时对成本函数求导，反向传播
        optimizer_step(0.01)  # 设置学习率，更新权重与偏差
        batch_loss.append(loss.data.numpy())  # 一次迭代中，存储每一批数据的成本函数值

    if i % 100 == 0:
        # 每100次迭代后，计算batch均值，并添加至losses数列中
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

toc = time.time()
print('2.0版本时间： ' + str(toc - tic))
fig = plt.figure(figsize=(10, 7))
plt.plot(np.arange(len(losses)) * 100, losses, 'o-')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()
