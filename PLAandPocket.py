#使用PLA算法与Pocket算法进行分类
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def PLA():
    W=np.zeros(2)
    b=0
    count=0
    judge=True
    while True:
        count=count+1
        judge=True
        for i in range(len(dataset)):
            X=np.array(dataset[i][:-1])
            Y=np.dot(W,X)+b
            if(np.sign(Y)==np.sign(dataset[i][-1])):
                continue
            else:
                judge=False
                W = W + (dataset[i][-1]) * X
                b = b + dataset[i][-1]
                
        if(judge==True):
            break
        
    print("W:",W)
    print("b:",b)
    print("count",count)
    
    return W,b
    

def Pocket():
    W = np.ones(2)
    b = 0
    best_W=W
    best_b=b
    count=0
    nummin=1000
    while True:
        count=count+1
        judge=True
        error=[]
        for i in range(len(dataset)):
            X=np.array(dataset[i][:-1])
            Y=np.dot(W,X)+b
            if(np.sign(Y)==np.sign(dataset[i][-1])):
                continue
            else:
                judge=False
                error.append(dataset[i])
                
        if(judge==False):
            j = random.randint(0,len(error)-1)
            W = W + (error[j][-1]) * np.array(error[j][:-1])
            b = b + error[j][-1]
            num=num_fault(W,b,dataset)
            if(num<nummin):
                nummin=num
                best_W=W
                best_b=b
                
        if(judge==True or count==100):
            break
    
    print("W:",best_W)
    print("b:",best_b)
    
    return W,b
    
def num_fault(W,b,dataset):
    count=0
    for i in range(len(dataset)):
        X=np.array(dataset[i][:-1])
        Y=np.dot(W,X)+b
        if(np.sign(Y)!=np.sign(dataset[i][-1])):
            count+=count
            
    return count

def DATA():
    data11=[]
    data22=[]
    mean1 = [-5,0]
    cov1 = [[1,0],[0,1]]
    data1 = np.random.multivariate_normal(mean1,cov1,200)
    for data in data1: 
        data11.append(np.append(data,[1]))
    #print(data1)
    mean2 = [0,5]
    cov2 = [[1,0],[0,1]]
    data2 = np.random.multivariate_normal(mean2,cov2,200)
    for data in data2: 
        data22.append(np.append(data,[-1]))

    data3=np.concatenate((data11, data22))
    print(data3)
    return data3
    
    

#生成数据集并测试
dataset = DATA()
ratio = 0.8
split = int(ratio * len(dataset))
np.random.seed(0)
dataset = np.random.permutation(dataset)
dataset_test = dataset[split:, :]
dataset = dataset[:split, :]

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
W,b=PLA()
endtime = time.time()
dtime = endtime - starttime

#输出耗时
print(dtime,"s")

plt.plot([-8,2],[(8*W[0]-b)/W[1],((-2)*W[0]-b)/W[1]],'g')#画直线
#计算并输出分类准确率
accurate1=(len(dataset)-num_fault(W,b,dataset))/len(dataset)
accurate2=(len(dataset_test)-num_fault(W,b,dataset_test))/len(dataset_test)
print("训练集分类准确率:",accurate1)
print("测试集分类准确率:",accurate2)
plt.show()



    
    
    
            
                

            
                
            
            
                
