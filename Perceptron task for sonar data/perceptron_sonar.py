#!/usr/bin/env python
# coding: utf-8

# # What is a Perceptron
# A perceptron is a building block of neural network. 
# 
# A single perceptron is capable to classify an input into two classes (Binary classifier i-e 0 / 1).
# ![A perceptron looks like this.](./perceptron_node.png "Perceptron")

# ## Different parts of a perceptron:
# 
# 1. Input features (X) from dataset.
# 2. Weights (W), one for each input feature and a bias B.
# 3. A Net input Function.
# 4. Activation Function to normalize the values between (0 - 1).
# 5. Output (0/1 or Yes/No or Dog/ Cat etc.)
# 
# 

# ## Sonar Dataset
# 
# Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp.
# 
# The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


columns = [i for i in range(1,61)]
columns.append("label")
df = pd.read_csv("sonar.all-data",delimiter = ",",names = columns,header = None)


# In[3]:


df.head()


# Replacing R with 0 and M with 1 as our perceptron can only deal with numbers.

# In[4]:


df["label"].replace({'R': 0, 'M': 1},inplace = True)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df[columns[:-1]], df[columns[-1]], test_size=0.33, random_state=42)


# In[6]:


X_train.head()


# In[7]:


y_train.head()


# In[8]:


y_test.head()


# In[9]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[10]:


X_train.shape


# ## Weights
# Let's initialize Weights for each input feature. We have 60 features so we need to define 60 wieghts.

# In[11]:


W = np.random.rand(60,1)
W.shape


# ## Bias
# Let's initialize Bias.
# 

# In[12]:


b = np.random.rand()
b


# ## Forward Pass
# Forward pass contains two steps:
# 1. **Net Input Function:** where we multiply each feature x with it's corresponding weight w and then sum all of the resulting values to get a single value Z. 
# 2. **Activation Function:** Applying activaton funciton on Z.
# 

# ### Net Input Function
# We have to comput the Net Input Function for all the training samples.
# 
# 
# ![Net input function.](./NIF.png "Net input function")
# 

# In[13]:


X_train = X_train.T
X_train.shape


# In[14]:


X_train[0].shape


# In[15]:


numOfTrainSamples = X_train.shape[1]
numOfFeatures = X_train.shape[0]
Z = np.zeros(numOfTrainSamples)

for i in range(numOfTrainSamples):
    for j in range(numOfFeatures): 
        z = float(X_train[j][i] * W[j])
        Z[i] = Z[i]+z
    Z[i] = Z[i] + b
    


# In[16]:


len(Z)


# In[17]:


Z[:5]


# Same net input function can be computed in an optimized manner by using vectorized code.
# 

# In[18]:


W.shape


# In[19]:


X_train.shape


# In[20]:


Z = np.dot(W.T,X_train,) + b


# In[21]:


Z.shape


# In[22]:


Z[0,:5]


# ### Activation Funciton
# We apply activation function to normalize the output values between 0 and 1.
# Most commonly used Activation Functions are:
# 1. Sigmoid
# 2. Relu
# 3. Leaky Relu
# 4. tanh and more
# 
# We will use sigmoid for our example.
# 
# 
# ![Sigmoid function.](./sigmoid.png "Sigmoid Function")

# In[23]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[24]:


A = [sigmoid(z) for z in Z[0]]


# In[25]:


A[:5]


# More optimized way

# In[26]:


A = sigmoid(Z)


# In[27]:


A[0,:5]


# ### What's Next?
# We have computed the output values, now what to do with them? We need the perceptron to answer in Rock / Mine or in other words 0 / 1.
# We need to apply a threshold on the output values. In most cases a threshold of 0.5 is used. All the output values greater than 0.5 will be considered as 1 and less than 0.5 will be considered as 0.

# In[28]:


A = np.where(A < 0.5, 0, 1)


# In[29]:


A


# In[30]:


y_train


# In[31]:


print(y_train.shape)
print(A.shape)


# In[32]:


y_train = np.expand_dims(y_train,axis =0)


# In[33]:


y_train.shape


# ## Output Analysis
# Our perceptron has not properly categorized the input. We have a lot of errors in it. Let's correct our perceptron.
# 

# ## Back Propagation
# In Back propagation we compute errors / loss/ cost using a loss function and then tell each weight that how much it has contributed in the error which is done by taking partial derivative of Loss function with respect to each weight. 
# 
# ### Error Functions:
# 1. Mean Error Loss
# 2. Mean Squared Error 
# 3. Mean Absolute Error
# 4. Mean Squared Logarithmic Error Loss (MSLE)
# 5. Mean Percentage Error
# 6. Mean Absolute Percentage Error
# 7. Binary Classification Losses Binary Cross Entropy
# 8. Multi-Class Cross-Entropy
# 9. Squared Hinge Loss
# 10. Hinge Loss
# 
# 
# For our example we will be using Binary Cross Entropy Loss:
# ![Binary cross entropy loss.](./loss.png "Loss function")

# In[34]:


def binary_cross_entropy(A, Y):
    return -(Y * np.log(A) + (1 - Y) * np.log(1 - A)).mean()


# In[35]:


J = binary_cross_entropy(A, y_train)
print(J)


# Our implementation of loss function cannot handle log of 0 which is equal to 1 ( log(0) = 1 ), that's why we will use library function for now.

# In[36]:


from sklearn.metrics import log_loss
J = log_loss(y_train,A)


# In[37]:


J


# ## Computing Gradients/ Slopes/ Derivatives
# Below are the partial derivatives of Loss function.
# 
# ![dz.](./dz.png "dz")
# 
# ![dw.](./dw.png "dw")
# 
# ![db.](./db.png "db")

# In[38]:


dz = A - y_train


# In[39]:


X_train.shape


# In[40]:


dz.shape


# We need to compute derivative of each weight for each input.

# In[41]:


dw = np.zeros(len(W))
for i in range(len(W)):
    for j in range(X_train.shape[1]):
        #print(str(i)+ " "+ str(j))
        #print(X_train[i][j])
        dw[i] = dw[i] + dz[0][j]*X_train[i][j]
    dw[i] = dw[i]/X_train.shape[1]


# In[42]:


dw[:5]


# In[43]:


numOfTrainSamples


# More optimized way

# In[44]:


dw =  np.dot(X_train,dz.T)/numOfTrainSamples


# In[45]:


dw[:5]


# For bias we need just need the mean of sum of all dz.
# 

# In[46]:


db = np.sum(dz,axis =1)/numOfTrainSamples


# ## Gradient Desent Step
# Now we will update all the weights according to their slopes.
# 
# **Learning Rate (alpha)**
# alpha is used to control the gradients, if we keep the alpha too high our gradients will diverge from minimum and if we take the alpha too low, the gradients will converge to minimum slowly.
# 
# alpha range [0,1]
# 
# let's suppose alpha is 0.001
# 
# ### Update formulas for weight and bias
# 
# ![w_update.](./w_update.png "w_update")
# 
# 
# ![b_update.](./b_update.png "b_update")

# In[47]:


# W = 0
# b = 0
# alpha = 0.8


# In[48]:


# W = W - alpha * dw


# In[49]:


# b = b - alpha *db


# ## Epoch
# 1 Forward and 1 Backward pass is known as 1 epoch.
# 
# ## Task
# 1. Write code to perform N number of epochs until the loss gets close to zero.
# 2. Compute the loss after each epcoh using sklearn loss function.
# 3. Once the perceptron gets trained, test the trained perceptron on testing data and report test accuracy, confusion matrix.
# 4. Try different values of alpha and see how it affects the training process.
# 5. Use the above vectorized code to make 2 layer Neural Network. 1st layer will contain 2 Perceptrons and last layer will contain 1 perceptron. See how it affects the performance using accuracy and confusion matrix.
# 

# In[50]:


from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[51]:


#calculating the loss for training data-set
alpha  = 0.6

for i in range(100):
        Z = np.dot(W.T, X_train,) + b
        A = sigmoid(Z)
        A = np.where(A < 0.5, 0, 1)
#         print('A', A.shape)
#         print('y_train', y_train.shape)
        J = log_loss(y_train,A)
        print("Error in each epoch: ",J)
        dz = A - y_train
        dw = np.dot(X_train, dz.T)/numOfTrainSamples
        db = np.sum(dz, axis=1)/numOfTrainSamples
        W = W - alpha * dw
        b = b - alpha * db   
        
print(" ")
#calculating accuracy and confusion matrix for training data-set
print('Model Accuracy :', accuracy_score(A[0], y_train[0]))

#confusion matrix
matrix = confusion_matrix(A[0], y_train[0], labels=[1,0])
matrix


# In[52]:


#calculating loss for testing data-set
alpha = 0.6

for i in range(50):
        Z = np.dot(W.T, X_test.T,) + b
        A = sigmoid(Z)
        A = np.where(A < 0.5, 0, 1)
#         print('A', A.shape)
#         print('y_test', y_test.shape)
        re_y_test = np.reshape(y_test, (1, 69))
#         print('Reshaped y_test', re_y_test.shape )
        J = log_loss(re_y_test,A)
        print("Error in each epoch: ",J)
        dz = A - y_test
        dw = np.dot(X_test.T, dz.T)/numOfTrainSamples
        db = np.sum(dz, axis=1)/numOfTrainSamples
        W = W - alpha * dw
        b = b - alpha * db

print(" ")

#calculating the model accuracy and confusion matrix for test data-set
print('Model Accuracy', accuracy_score(A[0], re_y_test[0]))
matrix = confusion_matrix(A[0], re_y_test[0], labels=[1,0])
matrix


# In[ ]:





# In[ ]:




