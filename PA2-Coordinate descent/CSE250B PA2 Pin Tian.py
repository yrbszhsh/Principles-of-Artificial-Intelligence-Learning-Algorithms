
# coding: utf-8

# In[210]:


#Read in data and just keep the first two classes so as to create a binary problem
get_ipython().magic('matplotlib inline')
import numpy as np
from sklearn.datasets import load_wine
data, labels = load_wine(True)
data = data[:(59 + 71)]
labels = np.expand_dims(labels[:(59 + 71)], axis = 1)


# # Logistic Regression

# In[446]:


def logit(prediction):
    return 1.0 / (1 + np.exp(-prediction))
def CalculateLossAcc(X, y, w, b):
    prediction = logit(X.dot(w.T) + b)
    loss = 0
    for i in range(len(y)):
        if y[i] == 0:
            loss -= np.log(1-prediction[i])
        else:
            loss -= np.log(prediction[i])
    loss = 1.0 * loss / len(X)
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    acc = 1.0 * np.sum([a == b for (a, b) in zip(prediction, y)]) / len(y)
    return loss, acc


# In[451]:


from sklearn import linear_model
losslogistic = np.zeros(3000)
acclogistic = np.zeros(3000)
for i in range(3000):
    logistic = linear_model.LogisticRegression(C = 1e10, max_iter = i, solver = 'sag')
    logistic.fit(data,labels)
    weight = logistic.coef_
    bias = logistic.intercept_
    iteration = logistic.n_iter_
    loss, acc = CalculateLossAcc(data, labels, weight, bias)
    losslogistic[i] = loss[0]
    acclogistic[i] = acc


# In[452]:


import matplotlib
import matplotlib.pyplot as plt

x = [i for i in range(len(losslogistic))]
plt.figure(figsize = (8,6))
plt.plot(x, losslogistic)
plt.xlabel('# of iteration')
plt.ylabel('loss')
plt.title('Loss of Logistic Regression')
plt.show()


# In[453]:


x = [i for i in range(len(acclogistic))]
plt.figure(figsize = (8,6))
plt.plot(x, acclogistic)
plt.xlabel('# of iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy of Logistic Regression')
plt.show()


# # My Coordinate Descent Method

# In[414]:


import numpy as np
from sklearn import linear_model
class LogisticRegressionWithGoordinateDescent():
    def __init__(self, X, y, lr = 1e-3, maxIter = 1000, random = False):
        self.X = self.addBias(X)
        self.y = y
        self.lr = lr
        self.w = np.zeros((len(self.X[0]),1))
        self.maxIter = maxIter
        self.random = random
        self.loss = []
        self.acc = []
        
    def addBias(self, X):
        return np.insert(X, 0, 1, axis = 1)
    
    def logit(self,prediction):
        return 1.0 / (1 + np.exp(-prediction))
    
    def findIndex(self, derE):
        if self.random:
            return np.random.randint(0, 13)
        else:
            return np.argmax(np.abs(derE))
    
    def CalculateLossAcc(self):
        prediction = self.logit(self.X.dot(self.w))
        loss = 0
        for i in range(len(self.y)):
            if self.y[i] == 0:
                loss -= np.log(1 - prediction[i])
            else:
                loss -= np.log(prediction[i])
        loss = 1.0 * loss / len(self.y)
        prediction[prediction > 0.5] = 1
        prediction[prediction <= 0.5] = 0
        acc = 1.0 * np.sum([a == b for (a, b) in zip(prediction, self.y)]) / len(self.y)
        return loss, acc
    
    def fit(self):
        it = 0
        while it < self.maxIter:
            prediction = self.logit(self.X.dot(self.w))
            derE = np.sum((prediction - self.y) * self.X, axis=0) / len(self.X)
            wIndex = self.findIndex(derE)
            self.w[wIndex] = self.w[wIndex] - self.lr * derE[wIndex]
            loss, acc = self.CalculateLossAcc()       
            print 'loss = %8.6f' % loss
            print 'acc = %5.2f' % acc
            self.loss.append(loss)
            self.acc.append(acc)       
            it += 1


# In[415]:


cd = LogisticRegressionWithGoordinateDescent(data, labels, lr = 1e-5, maxIter = 3000)


# In[416]:


cd.fit()


# In[417]:


print (len(cd.loss))
print max(cd.acc)


# In[418]:


x = [i for i in range(len(cd.loss))]
plt.figure(figsize = (8,6))
plt.plot(x, cd.loss)
plt.xlabel('# of iteration')
plt.ylabel('loss')
plt.title('Loss of Coordinate Descent')
plt.show()


# In[459]:


x = [i for i in range(len(cd.acc))]
plt.figure(figsize = (8,6))
plt.plot(x, cd.acc)
plt.xlabel('# of iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy of Coordinate Descent')
plt.show()


# # Uniformly Random Coordinate Descent

# In[460]:


cd1 = LogisticRegressionWithGoordinateDescent(data, labels, lr = 1e-4, maxIter = 3000, random = True)


# In[461]:


cd1.fit()


# In[462]:


x = [i for i in range(len(cd1.loss))]
plt.figure(figsize = (8,6))
plt.plot(x, cd1.loss)
plt.xlabel('# of iteration')
plt.ylabel('loss')
plt.title('Loss of Random Select')
plt.show()


# In[463]:


x = [i for i in range(len(cd1.acc))]
plt.figure(figsize = (8,6))
plt.plot(x, cd1.acc)
plt.xlabel('# of iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy of Random Select')
plt.show()


# In[464]:


x = [i for i in range(len(cd1.loss))]
plt.figure(figsize = (8,6))
rsoss, = plt.plot(x, cd1.loss)
cdloss, = plt.plot(x, cd.loss)
lrloss, = plt.plot(x, losslogistic)
plt.legend([logloss, cdloss, lrloss], ['Random Select', 'Coordinate Descent', 'Logistic Regression'])
plt.grid(True)
plt.xlabel('# of iteration')
plt.ylabel('loss')
plt.title('Loss of three Algorithm')
plt.show()

