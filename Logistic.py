import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

def Logistic(x):
    return 1/(1+np.exp(-x))


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
        
    
    
def SGD(x_train,y_train):
    W = np.random.normal(size=(x_train.shape[1],))
    train_loss=[]
    for i in range(num_epoch):
        batch_g=batch_generater(x_train,y_train,batch_size)
        for x_g,y_g in batch_g:
            pred=Logistic(x_g@W).reshape((len(x_g),1))#数组升维
            grad=-(x_g.T@(y_g-pred)).T+l2*W
            W=W-learning_rate*grad
            W=W.flatten()
        pred=Logistic(x_train@W).reshape((len(x_train),1))
        loss=-y_train.T@np.log(pred)-(1-y_train).T@np.log(1-pred)+l2 * np.linalg.norm(W) ** 2 / 2
        loss=loss[0][0]
        train_loss.append(loss/len(x_train))
    return W,train_loss


def DATA():
    data11=[]
    data22=[]
    mean1 = [-5,0]
    cov1 = [[1,0],[0,1]]
    data1 = np.random.multivariate_normal(mean1,cov1,200)
    data1 = np.concatenate([data1, np.ones((data1.shape[0], 1))], axis=1)
    #for data in data1: 
        #data11.append(np.append(data,[1]))
    #print(data1)
    mean2 = [0,5]
    cov2 = [[1,0],[0,1]]
    data2 = np.random.multivariate_normal(mean2,cov2,200)
    data2 = np.concatenate([data2, np.zeros((data2.shape[0], 1))], axis=1)
    #for data in data2: 
        #data22.append(np.append(data,[0]))

    data3=np.concatenate((data1, data2))
    print("生成数据集:",data3)
    return data3

#给定超参数
num_epoch=100
learning_rate=0.001
batch_size=32
l2=1.0

#生成数据
dataset = DATA()
# 划分训练集与测试集
ratio = 0.8
split = int(ratio * len(dataset))
np.random.seed(0)
dataset = np.random.permutation(dataset)
# y的维度调整为(len(data), 1)，与后续模型匹配
x_train, y_train = dataset[:split, :], dataset[:split, -1].reshape(-1, 1)
x_test, y_test = dataset[split:, :], dataset[split:, -1].reshape(-1, 1)

#绘制数据散点图
pos_index = np.where(y_train == 1)
neg_index = np.where(y_train == 0)
plt.scatter(x_train[pos_index, 0], x_train[pos_index, 1], 
    marker='o', color='yellow', s=10)
plt.scatter(x_train[neg_index, 0], x_train[neg_index, 1], 
    marker='x', color='blue', s=10)
plt.xlabel('X1 axis')
plt.ylabel('X2 axis')

#计算分类权重
starttime = time.time()
W,train_loss=SGD(x_train,y_train)
train_loss=np.array(train_loss)
#print(W.shape)
endtime = time.time()
dtime = endtime - starttime

#输出耗时
print("耗时:",dtime,"s")

#绘制分类面
# 直线方程：W0 * x_1 + W1 * x_2 + W2 = 0
plt.plot([-8,4],[(8*W[0]-W[2])/W[1],((-4)*W[0]-W[2])/W[1]],'g')#画直线
plt.show()

#绘制损失函数随epoch的变化曲线
x=np.linspace(1,num_epoch,num_epoch)
plt.title("Epoch")
plt.xlabel("epoch")    #设置x轴标注
plt.ylabel("loss")
plt.plot(x,train_loss,color='blue',label='train loss')#画直线
plt.show()



#分类
print("对测试集输出分类结果为:")
pred_1=np.array(list(np.where(Logistic(x_test@W)>=0.5))).flatten()
pred_0=np.array(list(np.where(Logistic(x_test@W)<0.5))).flatten()
for i in pred_1:
    print('数据点:',x_test[i][0],x_test[i][1],'真实类别:',y_test[i],'计算类别:',1,'概率值:',Logistic(W@x_test[i].T))
for i in pred_0:
    print('数据点:',x_test[i][0],x_test[i][1],'真实类别:',y_test[i],'计算类别:',0,'概率值:',1-Logistic(W@x_test[i].T))









    