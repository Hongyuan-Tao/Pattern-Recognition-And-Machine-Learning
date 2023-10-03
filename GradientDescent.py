import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

def GInverse():
    W=np.dot(np.dot(np.linalg.inv(np.dot(x_train.T,x_train)),x_train.T),y_train)
    W=W.flatten()
    print(W)
    train_loss = np.sqrt(np.square(x_train @ W - y_train).mean())
    test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
    return W,train_loss,test_loss
    
def GD():
    W = np.random.normal(size=x_train.shape[1])
    for i in range(num_epoch):
        # 初始化批量生成器
        grad = 2*(x_train.T @ x_train @ W - (x_train.T @ y_train).flatten()) / len(x_train)#要时刻关注二维数组和一维数组
        # 更新参数
        #print(grad)
        W = W - learning_rate * grad
    
    train_loss = np.sqrt(np.square(x_train @ W - y_train).mean())
    test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
    
    #print(W)
    return W,train_loss,test_loss

def batch_generator(x, y, batch_size, shuffle=True):
    # 批量计数器
    batch_count = 0
    if shuffle:
        # 随机生成0到len(x)-1的下标
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    while True:
        start = batch_count * batch_size
        end = min(start + batch_size, len(x))
        if start >= end:
            # 已经遍历一遍，结束生成
            break
        batch_count += 1
        yield x[start: end], y[start: end]

def SGD():
    # 拼接原始矩阵
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
    # 随机初始化参数
    W = np.random.normal(size=x_train.shape[1])

    # 随机梯度下降
    # 为了观察迭代过程，我们记录每一次迭代后在训练集和测试集上的均方根误差
    train_losses = []
    test_losses = []
    for i in range(num_epoch):
        # 初始化批量生成器
        batch_g = batch_generator(x_train, y_train, batch_size, shuffle=True)
        train_loss = 0
        for x_batch, y_batch in batch_g:
            # 计算梯度
            grad = (x_batch.T @ x_batch @ W - (x_batch.T @ y_batch).flatten()) 
            # 更新参数
            W = W - learning_rate * grad / len(x_batch)
            # 累加平方误差
            train_loss += np.square(x_batch @ W - y_batch).sum()
        # 计算训练和测试误差
        train_loss = np.sqrt(train_loss / len(X))
        train_losses.append(train_loss)
        test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
        test_losses.append(test_loss)

    return W, train_losses, test_losses#返回损失列表

def Adagrad():
    W = np.random.normal(size=x_train.shape[1])
    summ=0
    for i in range(num_epoch):
        # 初始化批量生成器
        grad = 2*(x_train.T @ x_train @ W - (x_train.T @ y_train).flatten()) / len(x_train)#要时刻关注二维数组和一维数组
        summ=summ+grad.T@grad
        # 更新参数
        #print(grad)
        learning_rate=0.05/math.sqrt(summ/(i+1))
        W = W - learning_rate * grad
    
    train_loss = np.sqrt(np.square(x_train @ W - y_train).mean())
    test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
    
    print(W)
    return W,train_loss,test_loss

def RMSProp():
    W = np.random.normal(size=x_train.shape[1])
    summ=0
    for i in range(num_epoch):
        # 初始化批量生成器
        grad = 2*(x_train.T @ x_train @ W - (x_train.T @ y_train).flatten()) / len(x_train)#要时刻关注二维数组和一维数组
        
        # 更新参数
        #print(grad)
        learning_rate=0.05/(alpha*math.sqrt(summ/(i+1))+(1-alpha)*grad.T@grad)
        
        W = W - learning_rate * grad
        summ=summ+grad.T@grad
    
    train_loss = np.sqrt(np.square(x_train @ W - y_train).mean())
    test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
    
    print(W)
    return W,train_loss,test_loss
 
def Momentum():
    W = np.random.normal(size=x_train.shape[1])
    m=np.zeros(x_train.shape[1])
    for i in range(num_epoch):
        # 初始化批量生成器
        grad = 2*(x_train.T @ x_train @ W - (x_train.T @ y_train).flatten()) / len(x_train)#要时刻关注二维数组和一维数组
        # 更新参数
        #print(grad)
        m=lambdaa*m-learning_rate * grad
        W = W + m
    
    train_loss = np.sqrt(np.square(x_train @ W - y_train).mean())
    test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
    
    print(W)
    
    return W,train_loss,test_loss

def Adam():
    W = np.random.normal(size=x_train.shape[1])
    summ=0
    m=np.zeros(x_train.shape[1])
    for i in range(num_epoch):
        # 初始化批量生成器
        grad = 2*(x_train.T @ x_train @ W - (x_train.T @ y_train).flatten()) / len(x_train)#要时刻关注二维数组和一维数组
        # 更新参数
        #print(grad)
        learning_rate=0.05/(alpha*math.sqrt(summ/(i+1))+(1-alpha)*grad.T@grad)   
        m=lambdaa*m-learning_rate * grad
        W = W + m
        summ=summ+grad.T@grad
    
    train_loss = np.sqrt(np.square(x_train @ W - y_train).mean())
    test_loss = np.sqrt(np.square(x_test @ W - y_test).mean())
    
    print(W)
    
    return W,train_loss,test_loss
   
def num_fault(W,x,y):
    count=0
    for i in range(len(y)):
        Y=np.dot(W,x[i])
        if(np.sign(Y)!=np.sign(y[i])):
            count=count+1

    return count

def DATA():
    data11=[]
    data22=[]
    mean1 = [1,0]
    cov1 = [[1,0],[0,1]]
    data1 = np.random.multivariate_normal(mean1,cov1,200)
    for data in data1: 
        data11.append(np.append(data,[1]))
    #print(data1)
    mean2 = [0,1]
    cov2 = [[1,0],[0,1]]
    data2 = np.random.multivariate_normal(mean2,cov2,200)
    for data in data2: 
        data22.append(np.append(data,[-1]))

    data3=np.concatenate((data11, data22))
    print("生成数据集:",data3)
    return data3

#给定超参数
num_epoch=150
learning_rate=0.01
batch_size=32
alpha=0.9
lambdaa=0.9

#生成数据
dataset = DATA()
# 划分训练集与测试集
x_train=[]
x_test=[]
ratio = 0.8
split = int(ratio * len(dataset))
np.random.seed(0)
dataset = np.random.permutation(dataset)
# y的维度调整为(len(data), 1)，与后续模型匹配
x_train1, y_train = dataset[:split, :2], dataset[:split, -1].reshape(-1, 1)
x_test1, y_test = dataset[split:, :2], dataset[split:, -1].reshape(-1, 1)
for data in x_train1: 
    x_train.append(np.append(data,[1]))
    
for data in x_test1: 
    x_test.append(np.append(data,[1]))
    

x_train=np.array(x_train)
x_test=np.array(x_test)

#print(x_train)
#print(y_train)

x1=[]
y1=[]
x2=[]
y2=[]

#绘制数据散点图
for i in range(len(dataset)):
    if(dataset[i][-1]==1):
        x1.append(dataset[i][0])
        y1.append(dataset[i][1])
    else:
        x2.append(dataset[i][0])
        y2.append(dataset[i][1])

plt.scatter(x1,y1,marker='o')
plt.scatter(x2,y2,marker='x')


starttime = time.time()
#分类
W1,train_loss1,test_loss1=GInverse()
#print(W.shape)
endtime = time.time()
dtime = endtime - starttime

#输出耗时
print("耗时:",dtime,"s")

plt.title("GInverse")
plt.xlabel("X1")    #设置x轴标注
plt.ylabel("X2")
plt.plot([-8,4],[(8*W1[0]-W1[2])/W1[1],((-4)*W1[0]-W1[2])/W1[1]],'g')#画直线
#计算并输出分类准确率
accurate1=(len(y_train)-num_fault(W1,x_train,y_train))/len(y_train)
accurate2=(len(y_test)-num_fault(W1,x_test,y_test))/len(y_test)
print("训练集分类准确率:",accurate1)
print("测试集分类准确率:",accurate2)
plt.show()

#分类
W,train_loss,test_loss=GD()
print('W:',W)
print("训练集损失函数:",train_loss)
print("测试集损失函数:",test_loss)
#print(W.shape)
endtime = time.time()
dtime = endtime - starttime

#输出耗时
print("耗时:",dtime,"s")

plt.title("GD")
plt.xlabel("X1")    #设置x轴标注
plt.ylabel("X2")
plt.scatter(x1,y1,marker='o')
plt.scatter(x2,y2,marker='x')
plt.plot([-8,4],[(8*W[0]-W[2])/W[1],((-4)*W[0]-W[2])/W[1]],'g')#画直线
#计算并输出分类准确率
accurate1=(len(y_train)-num_fault(W,x_train,y_train))/len(y_train)
accurate2=(len(y_test)-num_fault(W,x_test,y_test))/len(y_test)
print("训练集分类准确率:",accurate1)
print("测试集分类准确率:",accurate2)
plt.show()

#绘制损失函数随epoch的变化曲线
train_loss1=[]
test_loss1=[]
for num_epoch in range(1,150):
    W,train_loss,test_loss=GD()
    train_loss1.append(train_loss)
    test_loss1.append(test_loss)

x=np.linspace(10,150,149)


plt.title("Epoch")
plt.xlabel("epoch")    #设置x轴标注
plt.ylabel("loss")
plt.plot(x,train_loss1,color='blue',label='train loss')#画直线
plt.plot(x,test_loss1,color='yellow',label='test loss')
plt.show()

    
    
    
            
                

            
                
            
            
                
