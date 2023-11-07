import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import time

# 加载IRIS数据集
iris = load_iris()
X, Y = iris.data, iris.target


# 将 Y 转换为可以用0、1表示的向量
tmp = np.zeros((Y.shape[0],3)) # 寄存器
for i in range(Y.shape[0]):
    if Y[i] == 0:
        tmp[i] = [1,0,0]
    if Y[i] == 1:
        tmp[i] = [0,1,0]
    if Y[i] == 2:
        tmp[i] = [0,0,1]
Y = tmp

dataset=np.concatenate((X, Y), axis=1)

#print(dataset)
#生成数据集并测试
ratio = 0.6
split = int(ratio * len(X))
np.random.seed(0)
dataset = np.random.permutation(dataset)
X_test = dataset[split:,0:4]
X_train = dataset[:split,0:4]
Y_test = dataset[split:,4:]
Y_train = dataset[:split,4:]
#print(Y)

# 决策函数 decision function
def softmax(x,w):
    x = x.reshape(x.shape[0],1) # 转化为列向量
    a = (w.T)@x
    a = a - max(a) # 防止数据上溢出
    return np.exp(a)/(np.sum(np.exp(a)))

# 交叉熵损失函数 loss function
def loss(x,y,w):
    sum = 0; # 初始化

    for i in range(x.shape[0]):
        y_hat = softmax(x[i],w) # 预测值
        sum += np.dot(y[i],np.log(y_hat))

    return -sum/x.shape[0] # 求均值

# 梯度函数 optimizer
def gradient(x,y,w):
    # 这里注意，一维数组无法进行转置，只能先变成二维数组
    y_hat = softmax(x,w) # 预测值
    y = y.reshape(y.shape[0],1) # 变为二维矩阵
    error = (y-y_hat)
    x = x.reshape(x.shape[0],1)
    return -x @ error.T # 返回该样本点所在的梯度值

# 训练函数 train function
def train(x,y,w,lr=0.05,epoch=300): # 学习率是0.05,最大的迭代次数是epoch=300
    train_err = []
    test_err = []
    for i in range(epoch):
        reg = np.zeros((w.shape[0],w.shape[1])); # 存储梯度值的寄存器初始化
        if loss(x,y,w) > 0:
            for j in range(x.shape[0]):
                reg += gradient(x[j],y[j],w) # 获得所有样本梯度的累加
            reg = reg/x.shape[0] # 获得梯度均值
            w = w - lr*reg # 损失值大于0，计算梯度，更新权值
        test_err.append(test(X_test, Y_test,w))
        train_err.append(test(X_train,Y_train,w))
        # print('epoch:',i,'train error:',train_err[-1],'test error:',test_err[-1])
    return w,train_err,test_err


# 定义测试函数 Tset Function
def test(x,y,w):
    right = 0
    for i in range(x.shape[0]):
        max = np.argmax(softmax(x[i],w)) # 最大值所在位置
        max_y = np.argmax(y[i]) # 找到y中1的位置，就是所属的分类类别
        if max == max_y:
            right += 1
    return 1- right/x.shape[0]


w = np.ones((X_train.shape[1],Y_train.shape[1]))


starttime = time.time()

w,train_err,test_err = train(X_train,Y_train,w) 

endtime = time.time()
dtime = endtime - starttime
#输出耗时
print("耗时：",dtime,"s")

print(w.T)

print("训练集测试分类准确率：",1-train_err[-1])
print("测试集测试分类准确率：",1-test_err[-1])
# 绘制训练误差 train error
plt.plot(train_err)
plt.title('Softmax')
plt.xlabel('epoch')
plt.ylabel('train error')
plt.ylim((-0.3, 1))
plt.grid()
plt.show()



#参考文章：https://blog.csdn.net/weixin_53195427/article/details/130358620

