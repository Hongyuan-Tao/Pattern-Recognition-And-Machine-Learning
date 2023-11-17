import torch
from torch import nn
import numpy as np
from torch.utils import data
from sklearn.datasets import load_iris
from d2l import torch as d2l
import time

device=torch.device('cpu')

def load_data(batch_size):  #@save
    # 加载IRIS数据集
    iris = load_iris()
    X, Y = iris.data, iris.target

    Y=Y.reshape(-1, 1)
    
    dataset=np.concatenate((X, Y), axis=1)
    
    #生成数据集并测试
    ratio = 0.6
    split = int(ratio * len(X))
    np.random.seed(0)
    dataset = np.random.permutation(dataset)
    
    dataset=torch.from_numpy(dataset).to(device)
    iris_train=dataset[:split]
    iris_test=dataset[split:]
    
    return iris_train,iris_test

def batch_generater(x,y,batch_size,Shuffle=True):
    batch_count=0
    if(Shuffle==True):
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    while True:
        start=batch_count*batch_size
        end=min(start+batch_size,len(x))
        if start > end:
            break
        batch_count=batch_count+1
        yield x[start:end],y[start:end]
    
def train_epoch(net,train,loss,updater):
    
    for X, y in batch_generater(train[:,0:4].float(),train[:,4].long(),1,Shuffle=True):
        # 计算梯度并更新参数
        y_hat = net(X)
        #print(y_hat)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()

def train_nn(net, train,loss, num_epochs, updater):  #@save
    for epoch in range(num_epochs):
        train_epoch(net, train, loss, updater)
    
    
def predict_nn(net, test):  #@save
    countt=0
    count=0
    for X, y in batch_generater(test[:,0:4].float(),test[:,4].long(),1,Shuffle=True):
        #print(y)
        #print(net(X))
        j=y==net(X).argmax(axis=1)
        countt = countt+len([x for x in j if x == 1])
        count=count+len(j)
    return countt/count

num_epochs=100

net = nn.Sequential(nn.Linear(4, 10),
                    nn.Sigmoid(),
                    nn.Linear(10, 20),
                    nn.Sigmoid(),
                    nn.Linear(20, 3))

net = net.to(device)

trainer = torch.optim.SGD(net.parameters(), lr=0.02)
loss = nn.CrossEntropyLoss(reduction='none')

iris_train,iris_test = load_data(1)

starttime = time.time()
train_nn(net, iris_train,loss, num_epochs, trainer)
endtime = time.time()
dtime = endtime - starttime
#输出耗时
print("耗时：",dtime,"s")

print(f'网络参数为： {net.state_dict()}'
          )

print(f'预测准确率为：{predict_nn(net, iris_test)}')

