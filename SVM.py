import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import time
import math
from tqdm import tqdm,trange


def poly_kernel(d):
    def k(x,y):
        return np.inner(x, y)**d
    return k

def gaussian(sigma):
    def k(x,y):
        return np.exp(-np.inner(x-y,x-y)/(2*sigma**2))
    return k

def SMO(x,y,ker,C,max_iter):
    m=x.shape[0]
    K=np.zeros((m,m))
    alpha=np.zeros(m)
    for i in range(m):
        for j in range(m):
            K[i,j]=ker(x[i],x[j])
    for l in trange(max_iter):
        for i in range(m):
            j=np.random.choice([l for l in range(m) if l!=i])
            
            eta=K[i,i]+K[j,j]-K[i,j]
            e_i=np.sum(y*alpha*K[:,i])-y[i]
            e_j=np.sum(y*alpha*K[:,j])-y[j]
            alpha_i=alpha[i]+(e_j-e_i)/(eta+1e-5)*y[i]
            zeta=alpha[i]*y[i]+alpha[j]*y[j]
            if y[i]==y[j]:
                lower=max(0,zeta/y[i]-C)
                upper=min(C,zeta/y[i])
            else:
                lower=max(0,zeta/y[i])
                upper=min(C,zeta/y[i]+C)
            alpha_i=np.clip(alpha_i,lower,upper)
            alpha_j=(zeta-alpha_i*y[i])/y[j]
            
            alpha[i],alpha[j]=alpha_i,alpha_j
    return alpha

def Dual_SVM(x,y):
    alpha=SMO(x,y,ker=np.inner,C=1e8,max_iter=1000)
    sup_idx=alpha>1e-3
    #print(sup_idx)
    print("支持向量个数:",np.sum(sup_idx))
    w=np.sum((alpha[sup_idx]*y[sup_idx]).reshape(-1,1)*x[sup_idx],axis=0)
    wx=x@w.reshape(-1,1)
    b=-(np.max(wx[y==-1])+np.min(wx[y==1]))/2
    print("参数:",w,b)
    return w,b,sup_idx

def Kernel_SVM(x,y,ker):
    alpha=SMO(x,y,ker,C=1e8,max_iter=500)
    sup_idx=alpha>1e-5
    return alpha,sup_idx

def wx(x_new,ker,sup_x,sup_y,sup_alpha):
    s=0
    for xi,yi,ai in zip(sup_x,sup_y,sup_alpha):
        s=s+yi*ai*ker(xi,x_new)
    return s
    
def DATA():
    data11=[]
    data22=[]
    mean1 = [3,0]
    cov1 = [[1,0],[0,1]]
    data1 = np.random.multivariate_normal(mean1,cov1,200)
    data1 = np.concatenate([data1, np.ones((data1.shape[0], 1))], axis=1)
    #for data in data1: 
        #data11.append(np.append(data,[1]))
    #print(data1)
    mean2 = [0,3]
    cov2 = [[1,0],[0,1]]
    data2 = np.random.multivariate_normal(mean2,cov2,200)
    data2 = np.concatenate([data2, -np.ones((data2.shape[0], 1))], axis=1)
    #for data in data2: 
        #data22.append(np.append(data,[0]))

    data3=np.concatenate((data1, data2))
    print("生成数据集:",data3)
    return data3


#生成数据
dataset = DATA()
# 划分训练集与测试集
ratio = 0.8
split = int(ratio * len(dataset))
np.random.seed(0)
dataset = np.random.permutation(dataset)
# y的维度调整为(len(data), 1)，与后续模型匹配
x_train, y_train = dataset[:split, :2], dataset[:split, -1]
x_test, y_test = dataset[split:, :2], dataset[split:, -1]

'''plt.figure(1)
#绘制数据散点图
pos_index = np.where(y_train == 1)
neg_index = np.where(y_train == -1)
plt.scatter(x_train[pos_index, 0], x_train[pos_index, 1], 
    marker='o', color='red', s=10)
plt.scatter(x_train[neg_index, 0], x_train[neg_index, 1], 
    marker='x', color='blue', s=10)
plt.xlabel('X1 axis')
plt.ylabel('X2 axis')'''

'''#利用对偶支撑向量机计算分类权重
starttime = time.time()
W,b,sup_idx=Dual_SVM(x_train,y_train)
endtime = time.time()
dtime = endtime - starttime
#输出耗时
print("耗时:",dtime,"s")

#绘制分类面
# 直线方程：W0 * x_1 + W1 * x_2 + W2 = 0
plt.plot([-4,4],[(4*W[0]-b)/W[1],((-4)*W[0]-b)/W[1]],'g')#画直线

#绘制支撑向量
plt.scatter(x_train[sup_idx,0],x_train[sup_idx,1],marker='o',color='none',edgecolor='purple',s=150,label='support vectors')
plt.show()'''


#-----------------------------------------------------------------------#
#利用核支撑向量机进行分类
#plt.figure(1)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
#axs = axs.flatten()
cmap = ListedColormap(['coral', 'royalblue'])
kernels=[np.inner,gaussian(0.1)]



alpha,sup_idx=Kernel_SVM(x_train,y_train,gaussian(0.1))
sup_x=x_train[sup_idx]
sup_y=y_train[sup_idx]
sup_alpha=alpha[sup_idx]
neg=[wx(xi,gaussian(0.1),sup_x,sup_y,sup_alpha) for xi in sup_x[sup_y==-1]]
pos=[wx(xi,gaussian(0.1),sup_x,sup_y,sup_alpha) for xi in sup_x[sup_y==1]]
b = -(np.max(neg) + np.min(pos))/2
# 构造网格并用 SVM 预测分类
G = np.linspace(-6, 6, 100)
G = np.meshgrid(G, G)
X = np.array([G[0].flatten(), G[1].flatten()]).T # 转换为每行一个向量的形式
Y = np.array([wx(xi,gaussian(0.1),sup_x,sup_y,sup_alpha) + b for xi in X])
print(Y)
Y[Y < 0] = -1  
Y[Y >= 0] = 1                                                                                                                     
Y = Y.reshape(G[0].shape)
    
axs.contourf(G[0], G[1], Y, cmap=cmap, alpha=0.5)
    # 绘制原数据集的点
axs.scatter(x_train[y_train == -1, 0], x_train[y_train == -1, 1], color='red', label='y=-1')
axs.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], marker='x', color='blue', label='y=1')
axs.set_xlabel(r'$x_1$')
axs.set_ylabel(r'$x_2$')                                                                                                                                                                           
axs.legend()

plt.show()

'''
#分类
print("对测试集输出分类结果为:")
pred_1=np.array(list(np.where(Logistic(x_test@W)>=0.5))).flatten()
pred_0=np.array(list(np.where(Logistic(x_test@W)<0.5))).flatten()
for i in pred_1:
    print('数据点:',x_test[i][0],x_test[i][1],'真实类别:',y_test[i],'计算类别:',1,'概率值:',Logistic(W@x_test[i].T))
for i in pred_0:
    print('数据点:',x_test[i][0],x_test[i][1],'真实类别:',y_test[i],'计算类别:',0,'概率值:',1-Logistic(W@x_test[i].T))
'''








    