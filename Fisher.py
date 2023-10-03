import numpy as np
import matplotlib.pyplot as plt

def fisher(x1,x2):
    u1=np.mean(x1,axis=0)
    s1=(x1-u1).T@(x1-u1)
    u2=np.mean(x2,axis=0)
    s2=(x2-u2).T@(x2-u2)
    sw=s1+s2
    
    w=np.linalg.inv(sw)@(u1 - u2)
    
    return w,u1,u2

def predict(test_data,w,u1,u2):
    diff_1 = w.dot(u1.T)
    diff_2 = w.dot(u2.T)

    diff_cur = w.dot(test_data.T)
    
    return [1 if abs(diff_1-diff_cur[i])<abs(diff_2-diff_cur[i]) else -1 for i in range(len(diff_cur))]

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
    
    print("生成数据集:",data3)
    print("类型：",type(data3))
    return data3
    
def acc(result,dataset):
    count=0
    for i in range(len(dataset)):
        if result[i]==dataset[i][-1]:
            count=count+1
    
    return count/(len(dataset))

#生成数据集并测试
dataset = DATA()
ratio = 0.8
split = int(ratio * len(dataset))
np.random.seed(0)
dataset = np.random.permutation(dataset)
dataset_test = dataset[split:, :]
dataset_train = dataset[:split, :]

x1=[]
y1=[]
x2=[]
y2=[]


for i in range(len(dataset_train)):
    if(dataset_train[i][-1]==1):
        x1.append(dataset_train[i][:-1])
    else:
        x2.append(dataset_train[i][:-1])

#plt.scatter(x1,y1,marker='o')
#plt.scatter(x2,y2,marker='x')

x_test=[]
y_test=[]

for i in range(len(dataset_test)):
    x_test.append(dataset_test[i][:-1])
    y_test.append(dataset_test[i][-1])



#计算最佳投影向量
w,u1,u2=fisher(x1,x2)  
#计算分类阈值
s=np.dot(w,(u1+u2)/2)
print("最佳投影向量:",w)
print("分类阈值:",s)

train_data=[]
test_data=[]

for i in range(len(dataset_train)):
    train_data.append(dataset_train[i][:-1])

for i in range(len(dataset_test)):
    test_data.append(dataset_test[i][:-1])

train_data=np.array(train_data)
test_data=np.array(test_data)
#print(train_data)
#对训练集进行分类
result1=predict(train_data,w,u1,u2)
#对测试集进行分类
result2=predict(test_data,w,u1,u2)

#计算分类准确率
accurate1=acc(result1,dataset_train)
accurate2=acc(result2,dataset_test)

print("在训练集上的分类准确率为:",accurate1)
print("在测试集上的分类准确率为:",accurate2)

#绘制散点图与投影向量
x11=[]
x12=[]
x21=[]
x22=[]


for i in range(len(dataset_train)):
    if(dataset_train[i][-1]==1):
        x11.append(dataset_train[i][0])
        x12.append(dataset_train[i][1])
    else:
        x21.append(dataset_train[i][0])
        x22.append(dataset_train[i][1])

plt.scatter(x11,x12,marker='o')
plt.scatter(x21,x22,marker='x')

plt.plot([300*w[0],-300*w[0]],[300*w[1],-300*w[1]],'g')#画最佳投影向量
plt.show()



    
    
    
    