#!/usr/bin/env python
# coding: utf-8

# ### **TASK 1**
# Use the implementation of perceptron from the last lab manual and model the truth table of XOR.
# 
# ![XOR](./XOR.png "XOR")
# 

# ### **TASK 2**
# 1. Import the diabetes dataset. You can find data description in the description file.
# 2. Use pandas plots to have a look and feel of the different attributes of the data.
# 3. Apply perceptron algorithm to model the data distribution.
# 4. Use mean squared error as your cost/ error function.
# 5. Use the below given derivatives to find gradients of each W.
# 
# ![MSE](./MSE.png "MSE")
# 
# ![MSE derivative](./derivativeMSE.png "MSE derivative")
# 

# In[1]:


# Task 1
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score, confusion_matrix, classification_report


# In[2]:


# initialize list of lists
data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['A', 'B', 'Output'])

# print dataframe.
df


# In[3]:


X_train = df[['A','B']]
X_test = X_train
y_train = df['Output']
y_test = df['Output']
# type(Xtrain)


# In[4]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[5]:


X_train
# y_train


# In[6]:


W = np.random.rand(2,1)
W.shape


# In[7]:


b = np.random.rand()
b


# In[8]:


# Net Input function


# In[9]:


X_train = X_train.T
Z = np.dot(W.T, X_train) + b


# In[10]:


Z


# In[11]:


# Activation Function


# In[12]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[13]:


A = [sigmoid(z) for z in Z[0]]


# In[14]:


# A = sigmoid(Z)


# In[15]:


# A


# In[16]:


print(W.T.shape)
print(X_train.shape)


# In[17]:


alpha  = 0.44
numOfTrainSamples = X_train.shape[1]
b = np.random.rand()

for i in range(10):
        Z = np.dot(W.T, X_train) + b
        A = sigmoid(Z)
        A = np.where(A < 0.5, 0, 1)
        re_y_test = np.reshape(y_test, (1,4))
        J = log_loss(re_y_test,A)
#         print("Error in each epoch: ",J)
        dz = A - re_y_test
        dw = np.dot(X_train, dz.T)/numOfTrainSamples
        db = np.sum(dz, axis=1)/numOfTrainSamples
        W = W - alpha * dw
        b = b - alpha * db   
        
        print(" ")
        #calculating accuracy and confusion matrix for training data-set
        print('Model Accuracy :', accuracy_score(A[0], re_y_test[0]))


# In[18]:


#confusion matrix
matrix = confusion_matrix(A[0], re_y_test[0], labels=[1,0])
matrix


# In[ ]:





# In[19]:


# Task 2


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


columns = [i for i in range(1, 11)]
columns.append('Label')
v = pd.read_csv('diabetes.txt', delimiter = '\t')
df = pd.read_csv("diabetes.txt", delimiter = "\t", names = columns, header = None)
df.head()


# In[22]:


v.plot(kind = 'scatter', x = 'BP', y = 'Y')


# In[23]:


v.plot(kind = 'scatter', x = 'BMI', y = 'Y')


# In[24]:


import seaborn as sns

corrmat = v.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20, 20))

g = sns.heatmap(v[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# In[25]:


v.corr()


# In[26]:


# df[columns[:-1]]
# df[columns[-1]] //label


# In[27]:


df = df[1:]
df = df.astype(float)
df


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(df[columns[:-1]], df[columns[-1]], test_size = 0.33, random_state = 33)


# In[29]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[46]:


W = np.random.rand(10,1)
W


# In[47]:


b = np.random.rand()
b


# In[48]:


print(W.shape)
print(X_train.shape)


# In[49]:


#for training data
numOfTrainSamples = X_train.shape[1]
numOfFeatures = X_train.shape[0]
Z = np.zeros(numOfTrainSamples)

alpha = 0.4
for i in range(10):
    Z = np.dot(W.T,X_train.T) + b
    re_y_train = np.reshape(y_train, (1, 296))
    
    J = mean_squared_error(re_y_train, Z)
    dz = Z - re_y_train

    dw = np.dot(X_train.T,dz.T)/numOfTrainSamples
    db = np.sum(dz,axis =1)/numOfTrainSamples
   
    W = W - alpha * dw
    b = b - alpha *db

print(J)
print(W)


# In[50]:


#for testing data
numOfTestSamples = X_test.shape[1]
numOfFeatures = X_test.shape[0]
Z = np.zeros(numOfTestSamples)

alpha = 0.4
for i in range(10):
    Z = np.dot(W.T,X_test.T) + b

    re_y_test = np.reshape(y_test, (1, 146))

    J = mean_squared_error(re_y_test, Z)
    dz = Z - re_y_test

    dw = np.dot(X_test.T,dz.T)/numOfTestSamples
    db = np.sum(dz,axis =1)/numOfTestSamples
   
    W = W - alpha * dw
    b = b - alpha *db

print(J)
print(W)


# In[ ]:




