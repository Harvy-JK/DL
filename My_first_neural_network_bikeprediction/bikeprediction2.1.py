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

# 设置训练集，测试集
train_data = data[:-21 * 24]
test_data = data[-21 * 24:]

target_fields = ['cnt', 'casual', 'registered']  # 设置目标值
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

# 定义神经网络框架，feature.shape[1]个输入层单元，10个隐藏层，1个输出层
input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128
tic = time.time()
neu = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(hidden_size, output_size))
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(neu.parameters(), lr=0.01)
losses = []
for i in range(1000):
    # 每128个样本点被划分为一个撮，在循环的时候一批一批地读取
    batch_loss = []
    # start和end分别是提取一个batch数据的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = torch.tensor(X[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(Y[start:end], dtype=torch.float, requires_grad=True)
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # 每隔100步输出一下损失值（loss）
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

toc = time.time()
print('2.1版本时间： ' + str(toc - tic))
print(losses)  # 打印输出损失值
fig = plt.figure(figsize=(10, 7))
plt.plot(np.arange(len(losses)) * 100, losses, 'o-')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

# 测试神经网络,测试集带入
targets = test_targets['cnt']
targets = targets.values.reshape([len(targets), 1])
targets = targets.astype(float)

x = torch.tensor(test_features.values, dtype=torch.float, requires_grad=True)
y = torch.tensor(targets, dtype=torch.float, requires_grad=True)
print(x[:2])
print(len(x))

predict = neu(x)
print(neu(x))
print(len(neu(x)))
predict = predict.data.numpy()

# 这时候对比预测值与实际值，区别是横坐标要更改为日期

fig, ax = plt.subplots(figsize=(10, 7))  # ax为坐标系，下面会解释
mean, std = scaled_features['cnt']  # 这里方差与均值来自于cnt类内。从字典字典中直接提取。
print((predict * std + mean)[:10])  # 归一化公式返回值，原本区间[-1,1]，变为原本范围。
ax.plot(predict * std + mean, label='Prediction', linestyle='--')  # 预测值用虚线表达
ax.plot(targets * std + mean, label='Data', linestyle='-')  # 原数据用实线表达
ax.legend()  # 图标上表明每条曲线的文字说明
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')

# 对x轴进行标注，注意这里dates天数，不是data。
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
print(dates)
dates = dates.apply(lambda d: d.strftime('%b %d'))
print(dates)
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)  # x轴上点文字说明，旋转45°
plt.show()

# *****************************************************************8
# 神经网络的诊断

# 先选出三天预测不准的日期，Dec 22, 23，24
# 将这三天的数据聚集到一起，存入subset和subtargets中
bool1 = rides['dteday'] == '2012-12-22'
bool2 = rides['dteday'] == '2012-12-23'
bool3 = rides['dteday'] == '2012-12-24'

# tup原形tuple元组，元组与列表相似，元组内元素不可以修改，只可以连接合并
# zip()函数的作用，zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，下面会解释
bools = [any(tup) for tup in zip(bool1, bool2, bool3)]
print(bools)
subset = test_features.loc[rides[bools].index]
print(subset)
print(rides[bools].index)
subtargets = test_targets.loc[rides[bools].index]
print(subtargets)
subtargets = subtargets['cnt']
print(subtargets)
subtargets = subtargets.values.reshape([len(subtargets), 1])
print(subtargets)


# 定义了一个函数可以提取网络的权重信息，所有的网络参数信息全部存储在了neu的named_parameters集合中了
def feature(X, net):
    X = torch.tensor(X, dtype=torch.float, requires_grad=False)  # 只需提取超参数的数值，不需要求导
    dic = dict(net.named_parameters())  # 提取出来这个集合
    weights = dic['0.weight']  # 可以按照层数.名称来索引集合中的相应参数值
    biases = dic['0.bias']  # 可以按照层数.名称来索引集合中的相应参数值
    h = torch.sigmoid(X.mm(weights.t()) + biases.expand([len(X), len(biases)]))  # 隐含层的计算过程
    return h  # 输出层的计算


# 将这几天的数据输入到神经网络中，读取出隐含层神经元的激活数值，存入results中
results = feature(subset.values, neu).data.numpy()
print(results)
# 这些数据对应的预测值（输出层）
predict = neu(torch.tensor(subset.values, dtype=torch.float, requires_grad=True)).data.numpy()
# 将预测值还原成原始数据的数值范围
mean, std = scaled_features['cnt']
predict = predict * std + mean
subtargets = subtargets * std + mean
# 将所有的神经元激活水平画在同一张图上，蓝色的是模型预测的数值
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(results[:, :], '.:', alpha=0.3)  # alpha调节透明度
print(results[:, :])
ax.plot((predict - min(predict)) / (max(predict) - min(predict)), 'bs-', label='Prediction')  # 偏差比重
ax.plot((subtargets - min(predict)) / (max(predict) - min(predict)), 'ro-', label='Real')
ax.plot(results[:, 3], ':*', alpha=1, label='Neuro 4')
print(results[:, 3])

ax.set_xlim(right=len(predict))
ax.legend()
plt.ylabel('Normalized Values')

dates = pd.to_datetime(rides.loc[subset.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()

# 找到了与峰值相应的神经元，把它到输入层的权重输出出来
dic = dict(neu.named_parameters())
weights = dic['0.weight'][3]
plt.plot(weights.data.numpy(), 'o-')
print(weights.data.numpy())
plt.xlabel('Input Neurons')
plt.ylabel('Weight')
plt.show()

# 将所有的神经元激活水平画在同一张图上，蓝色的是模型预测的数值
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(results[:, :], '.:', alpha=0.3)  # alpha调节透明度
ax.plot((predict - min(predict)) / (max(predict) - min(predict)), 'bs-', label='Prediction')  # 偏差比重
ax.plot((subtargets - min(predict)) / (max(predict) - min(predict)), 'ro-', label='Real')
ax.plot(results[:, 1], ':*', alpha=1, label='Neuro 2')
print(results[:, 1])

ax.set_xlim(right=len(predict))
ax.legend()
plt.ylabel('Normalized Values')

dates = pd.to_datetime(rides.loc[subset.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)  # _= 这里我猜测的是，前面设定了x轴上有几个点，这里就是设置点的名称
plt.show()

# 找到了与峰值相应的神经元，把它到输入层的权重输出出来
dic = dict(neu.named_parameters())
weights = dic['0.weight'][1]
plt.plot(weights.data.numpy(), 'o-')
print(weights.data.numpy())
plt.xlabel('Input Neurons')
plt.ylabel('Weight')
plt.show()

weights = dic['0.weight']
print(weights)
biases = dic['0.bias']
print(biases)
