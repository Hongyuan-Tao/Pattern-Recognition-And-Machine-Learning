import numpy as np
from sklearn.datasets import load_iris
import random
from collections import Counter

# 加载IRIS数据集
iris = load_iris()
X, Y = iris.data, iris.target
Y = Y.reshape(len(Y), 1)
dataset=np.concatenate((X, Y), axis=1)
#print(dataset)
#生成数据集并测试
ratio = 0.6
split = int(ratio * len(X))
np.random.seed(0)
dataset = np.random.permutation(dataset)
X_test = dataset[split:,0:4]
X = dataset[:split,0:4]
Y_test = dataset[split:,4]
Y = dataset[:split,4]
#print(Y)
y_value = np.unique(Y)

def ovo():
    models=[]
    new_datas=[]
    #计算类别数目
    k = len(y_value)
    #将K个类别中的两两类别数据进行组合,并对y值进行处理
    for i in range(k-1):
        c_i = y_value[i]
        for j in range(i+1,k):
            c_j = y_value[j]
            for x,y in zip(X,Y):
                if y == c_i or y == c_j:
                    new_datas=np.append(new_datas,np.hstack((x,np.array([2*float(y==c_i)-1]))))
                    
            
            new_datas = np.array(new_datas)
            new_datas=new_datas.reshape(int(len(new_datas)/5), 5)
            #print(new_datas)
            
            w,b = Pocket(new_datas)
            models.append([(c_i,c_j),w,b])
    return models

def Pocket(dataset):
    W = np.ones(4)
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
            count=count+1

            
    return count

def ovo_predict(X,Y,models):
    num_true=0
    result = []
    for i in range(len(X)):
        pre = []
        for cls,w,b in models:
            #print((w@X[i]+b))
            pre.append(cls[0] if (w@X[i]+b)>=0 else cls[1])
        #print(pre)
        #print(most_common_number(pre))
        result= y_value[int(most_common_number(pre))]
        if(Y[i]==result):
            num_true=num_true+1
    return num_true/len(X)

def most_common_number(arr):
    counter = Counter(arr)
    return counter.most_common(1)[0][0]

models=ovo()
print("训练集预测准确率为：",ovo_predict(X,Y,models))
print("测试集预测准确率为：",ovo_predict(X_test,Y_test,models))




