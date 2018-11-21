
# coding: utf-8

# In[1]:


import pandas
import numpy as np
from numpy.linalg import inv
from sklearn import datasets, linear_model


# In[2]:


data=pandas.read_csv('df_NaN.csv')


# In[3]:


#set Developing to 1 and Developed to 0
data['Status'][data['Status'] == 'Developing']  = 0
data['Status'][data['Status'] == 'Developed']  = 1


#data normalization for further processing
#data.apply(normalization,axis=0)


# In[4]:


#extract target data

target=data['Life expectancy ']
target=np.array(target)
#Ignore the country and year effect
data=data.drop(['Country','Year','Unnamed: 0','Life expectancy '],axis=1)
#normalize data


# In[5]:


#extract normalization parameters and normalize original data
dataMax=np.array(data.max())
dataMin=np.array(data.min())
dataMean=np.array(data.mean())
data=data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# is it necessary to normalize label data?
life=target.copy()
#target=(target-target.min())/(target.max()-target.min())


# In[6]:


train_x=np.array(data)


# In[29]:


def least_square(train_x, train_y):
    """
    use least square method to reach the regression model
    input：
          train_x, training data
          train_y, labels in training data
    """  
    weights = inv(np.dot(train_x.T ,train_x).astype(float)).dot(train_x.T).dot(train_y) 
    return weights.astype(float)

least_square_weights=least_square(train_x,target)
#prove to be very inaccurate


# In[30]:


prediction=np.dot(train_x,least_square_weights)


# In[31]:


prediction


# In[32]:


MSE=np.sum((target-prediction)**2)/prediction.size


# In[33]:


MSE


# In[8]:


"""
def gradient_descent(train_x, train_y, maxCycle, alpha):
     
    numSamples, numFeatures = np.shape(train_x)
    weights = np.zeros((numFeatures,1))
     
    for i in range(maxCycle):
        h = train_x.dot(weights)
        err = h - train_y           
        weights = weights - (alpha*err.T.dot(train_x)).T
        print(err)
    return weights.astype(float)
"""


# In[9]:


#train a shallow neural network with pytorch


import torch
import sys
from torch.autograd import Variable

import torch .nn. functional as F

torch_x=Variable(torch.from_numpy(train_x.astype(float)).cuda(),requires_grad = True)
torch_y=Variable(torch.from_numpy(target).cuda(),requires_grad = False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(19,57).double()
        #self.fc2 = torch.nn.Linear(19,19).double()
        self.fc2 = torch.nn.Linear(57,1).double()
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))
        x=self.fc2(x)
        return x
    
model=Model().cuda()
criterion=torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
         
  


# In[10]:


y_pred=model(torch_x)


# In[11]:


y_pred


# In[12]:


torch_y=torch_y.reshape([2938,1])


# In[13]:


y_pred.shape


# In[14]:


T=5000
B=100
NB=30
N=2938
for epoch in range(T):
    running_loss=0.0
    idxminibatches = np. random . permutation (NB)
    for k in range(NB):
        i = idxminibatches [k]
        idxsmp=np.arange(B*i,min(B*(i+1),N))
        inputs = torch_x[idxsmp]
        labels = torch_y[idxsmp]
        
        optimizer.zero_grad()

        y_pred=model(inputs)

        loss=criterion(y_pred,labels)
        
        loss.backward()

        optimizer.step()
        running_loss+=loss[0]
        if k==29:
            
            print(epoch,running_loss/30)
            running_loss=0.0



        


            


# In[15]:


prediction=model(torch_x)


# In[16]:


prediction


# In[18]:


test_x=torch_x[300:600]
test_y=torch_y[300:600]
pred_y=model(test_x)
loss=criterion(pred_y,test_y)
print(loss)


# In[ ]:


#dataset has't been divided into training set, test set, validation set
#

