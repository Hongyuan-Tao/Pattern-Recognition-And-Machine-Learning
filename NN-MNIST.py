import torch
from torch import nn
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from d2l import torch as d2l
import time
from torchvision import datasets, transforms
from tqdm import tqdm,trange

device=torch.device('cuda')

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_mnist_labels(labels):  #@save
    """返回MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    cmp.to(device)
    return float(cmp.type(y.dtype).sum())

def predict_nn(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X=X.to(device)
            y=y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net,train_iter,loss,updater):
    metric = Accumulator(3)
    for X, y in train_iter:
        X=X.to(device)
        y=y.to(device)
        # 计算梯度并更新参数
        y_hat = net(X.to(device))
        #print(y_hat)
        l = loss(y_hat, y.to(device))
        updater.zero_grad()
        l.mean().backward()
        updater.step()

        #print(y_hat, y)
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_nn(net,train_iter,test_iter,loss, num_epochs, updater):  #@save
    LOSSs=[]
    acc_trains=[]
    acc_tests=[]
    for epoch in trange(num_epochs):
        LOSS,acc_train =train_epoch(net,train_iter, loss, updater)
        acc_test=predict_nn(net,test_iter)
        LOSSs.append(LOSS)
        acc_trains.append(acc_train)
        acc_tests.append(acc_test)
    return LOSSs,acc_trains,acc_tests
    

def predict(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        X=X.to(device)
        y=y.to(device)
        break
    trues = get_mnist_labels(y)
    preds = get_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].to(torch.device('cpu')).reshape((n, 28, 28)), 1, n, titles=preds[0:n])
    plt.show()

num_epochs=5

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

net = net.to(device)

trainer = torch.optim.SGD(net.parameters(), lr=0.9)
loss = nn.CrossEntropyLoss(reduction='none')

# 定义数据预处理操作
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载并加载MNIST训练集
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_iter = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

# 下载并加载MNIST测试集
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_iter = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

starttime = time.time()
LOSSs,acc_trains,acc_tests=train_nn(net, train_iter,test_iter,loss, num_epochs, trainer)
endtime = time.time()
dtime = endtime - starttime
#输出耗时
print("耗时：",dtime,"s")
print(f'预测准确率为：{acc_tests[-1]}')

x=np.linspace(1,num_epochs,num_epochs)

plt.figure(num=1)
plt.title("Epoch")
plt.xlabel("epoch-loss")    #设置x轴标注
plt.ylabel("loss")
plt.plot(x,LOSSs,color='blue',label='train loss')#画直线

plt.figure(num=2)
plt.title("Epoch")
plt.xlabel("epoch-accuracy")    #设置x轴标注
plt.ylabel("accuracy")
plt.plot(x,acc_trains,color='yellow',label='train loss')
plt.plot(x,acc_tests,color='red',label='test loss')

plt.show()

predict(net, test_iter)





