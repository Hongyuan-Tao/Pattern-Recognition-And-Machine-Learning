import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time
from tqdm import tqdm,trange

# 加载MNIST数据集
(x_train,y_train), (x_test,y_test) = mnist.load_data()  #第一次需要下载，不过很快
# x_train 60000张28*28的图片，图片上为0-9的数字   y_train：60000个标签，对应于x_train
#x_test：10000张28*28的图片  y_test：10000个标签，对应于x_test
print('x_shape: ',x_train.shape)    # (60000, 28, 28)
print('y_shape: ',y_train.shape)    # (60000,)
print('x_test_shape: ',x_test.shape) # (10000, 28, 28)
print('y_test_shape: ',y_test.shape) #  (10000,)
# 60000, 28, 28)->(60000, 784)
X_train = x_train.reshape(x_train.shape[0],-1)/255.0
X_test = x_test.reshape(x_test.shape[0],-1)/255.0



# 将 Y 转换为可以用0、1表示的向量
tmp1 = np.zeros((y_train.shape[0],10)) # 寄存器
tmp2 = np.zeros((y_test.shape[0],10)) # 寄存器
for i in range(y_train.shape[0]):
    tmp1[i] = np.zeros(10)
    tmp1[i][y_train[i]] = 1
    
for i in range(y_test.shape[0]):
    tmp2[i] = np.zeros(10)
    tmp2[i][y_test[i]] = 1
    
Y_train = tmp1
Y_test = tmp2
print(1)

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

# 决策函数 decision function
def softmax(x,w):
    x = x.reshape(x.shape[0],1) # 转化为列向量
    a = (w.T)@x
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
def train(x,y,w,lr=0.05,epoch=10): 
    train_err = []
    test_err = []
    for i in trange(epoch):
        batch_g=batch_generater(x,y,batch_size)
        for x_g,y_g in batch_g:
            reg = np.zeros((w.shape[0],w.shape[1])); # 存储梯度值的寄存器初始化
            if loss(x_g,y_g,w) > 0:
                for j in range(x_g.shape[0]):
                    reg += gradient(x_g[j],y_g[j],w) # 获得所有样本梯度的累加
                reg = reg/x_g.shape[0] # 获得梯度均值
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

#定义超参数
batch_size=256

w = np.random.normal(0, 0.01, (784, 10))

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

print("下面随机抽取MNIST中的十个数据，观察分类结果:")
arr = np.random.randint(0, 60001, size=10)
for i in arr:
    max = np.argmax(softmax(X_train[i],w)) # 最大值所在位置
    print("第",i,"个手写字符的真实值是:",y_train[i]," ","预测值是:",max,";")

#参考文章：https://blog.csdn.net/weixin_53195427/article/details/130358620

