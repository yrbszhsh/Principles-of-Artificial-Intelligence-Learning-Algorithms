
# coding: utf-8

# In[1]:


#Read in the data from the files.
from mnist import MNIST
import numpy as np

mndata = MNIST('..\CSE 253\HW1\mnist')

trainData, trainLabel = mndata.load_training()

testData, testLabel = mndata.load_testing()

trainData = np.asarray(trainData) / 255.0
trainLabel = np.asarray(trainLabel)
testData = np.asarray(testData) / 255.0
testLabel = np.asarray(testLabel)


# In[2]:


# 1-NN algorithm
def NN(dataSet, labelSet, testD):
    testL = np.zeros((len(testD), 1))
    for i in range(len(testD)):
        dist = np.sum(np.power(dataSet - testD[i], 2),axis=1)
        testL[i] = labelSet[np.argmin(dist)]
        if i % (len(testD) / 10) == 0:
            print (i)
    return testL


# # Condensed Nearest Neighbors

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)

import random
def myPrototype(data, label, M):
    trainD = []
    trainL = []
    num = 0
    ptr = 0
    rand = random.sample(range(0,len(data)), len(data))
    trainD.append(data[rand[0]])
    trainL.append(label[rand[0]])
    ptr +=1
    num += 1
    while num < M:
        if rand[ptr] == -1:
            ptr += 1
            continue
        neigh.fit(trainD,trainL)
        pred = neigh.predict(data[rand[ptr]].reshape(1, -1))
        if pred != label[rand[ptr]]:
            trainD.append(data[rand[ptr]])
            trainL.append(label[rand[ptr]])
            num += 1
            rand[ptr] = -1
        if ptr % 2000 == 0:
            print (ptr)
            print (num)
        ptr += 1
        if ptr == len(rand):
            ptr = 0
    trainD = np.asarray(trainD)
    trianL = np.asarray(trainL)
    print(ptr)
    print(num)
    return trainD, trainL


# 1.Pick 1,000 prototype

# In[43]:


accCNN1000 = np.zeros(5)
for i in range(5):
    xTrain, yTrain = myPrototype(trainData, trainLabel, 1000)
    neigh.fit(xTrain, yTrain)
    prediction = neigh.predict(testData)
    accCNN1000[i] = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
print (accCNN1000)


# In[10]:


prediction = NN(xTrain, yTrain, testData)
acc = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
acc


# 2.Pick 5,000 prototype

# In[21]:


xTrain, yTrain = myPrototype(trainData, trainLabel, 5000)
xTrain.shape


# In[23]:


neigh.fit(xTrain,yTrain)
prediction = neigh.predict(testData)
acc = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
acc


# 3.Pick 6,000 prototype

# In[25]:


xTrain, yTrain = myPrototype(trainData, trainLabel, 6000)
xTrain.shape


# In[26]:


neigh.fit(xTrain,yTrain)
prediction = neigh.predict(testData)
acc = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
acc

