# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:19:09 2018

@author: ericli
"""


import pandas
import numpy as np
from numpy.linalg import inv
import torch
from torch.autograd import Variable
import torch .nn. functional as F


#read data
data=pandas.read_csv('df_NaN.csv')

#set Developing to 1 and Developed to 0
data['Status'][data['Status'] == 'Developing']  = 0
data['Status'][data['Status'] == 'Developed']  = 1


#extract target data

target=np.array(data['Life expectancy '])

#Ignore the country and year 
data=data.drop(['Country','Year','Unnamed: 0','Life expectancy '],axis=1)

#extract normalization parameters and normalize original data
dataMax=np.array(data.max())
dataMin=np.array(data.min())
dataMean=np.array(data.mean())
data=data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# is it necessary to normalize label data? seems not
life=target.copy()
#target=(target-target.min())/(target.max()-target.min())

train_x=np.array(data)


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
#not very accurate

prediction=np.dot(train_x,least_square_weights)
MSE=np.sum((target-prediction)**2)/prediction.size
print('MSE of leas square method',MSE)

"""
#not working, need to debug later
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


#train a shallow neural network with pytorch, or linear regression with pytorch
#convert numpy data to torch variables
#use GPU if avaiable
if torch.cuda.is_available():
    torch_x=Variable(torch.from_numpy(train_x.astype(float)).cuda(),requires_grad = True)
    torch_y=Variable(torch.from_numpy(target).cuda(),requires_grad = False)
else:
    torch_x=Variable(torch.from_numpy(train_x.astype(float)),requires_grad = True)
    torch_y=Variable(torch.from_numpy(target),requires_grad = False)

class ShallowModel(torch.nn.Module):
    def __init__(self):
        super(ShallowModel, self).__init__()
        self.fc1 = torch.nn.Linear(19,57).double()
        #self.fc2 = torch.nn.Linear(19,19).double()
        self.fc2 = torch.nn.Linear(57,1).double()
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))
        x=self.fc2(x)
        return x
    
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(19,1).double()
        #self.fc2 = torch.nn.Linear(19,19).double()
        #self.fc2 = torch.nn.Linear(57,1).double()
        
    def forward(self,x):
        #x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))
        x=self.fc1(x)
        return x
    

def train_model(model,epochs,trainsize=0.8):
    assert isinstance(model,str)
    assert isinstance(epochs,int) and epochs>1
    
    if torch.cuda.is_available():
        if model=='shallow':
            model=ShallowModel().cuda()
        elif model=='linear':
            model=LinearModel().cuda()
    else:
        if model=='shallow':
            model=ShallowModel()
        elif model=='linear':
            model=LinearModel()
    
    criterion=torch.nn.MSELoss()
    optimizer= torch.optim.SGD(model.parameters(), lr=0.01)
    
    #separate trainning set and testset
    #shuffle the training set
    
    shuffleindex=np.arange(torch_x.size()[0])
    np.random.shuffle(shuffleindex)

    train_index=int(trainsize*torch_x.size()[0])
    
    train_x=torch_x[shuffleindex[0:train_index]]
    train_y=torch_y[shuffleindex[0:train_index]]
    
    test_x=torch_x[shuffleindex[train_index:]]
    test_y=torch_y[shuffleindex[train_index:]]
    
    T=epochs
    B=100
    NB=int(train_index/100)+1
    N=train_index
    #train the model
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
            if k==NB-1:
                print(epoch,running_loss/NB)
                running_loss=0.0
                
    #assess the model on test set
    
    y_pred=model(test_x)
    loss=criterion(y_pred,test_y)
    print('MSE on test set',loss[0])
    
    return model
         

train_model('linear',1000,0.9)
model=train_model('shallow',1000,0.9)

def prediction(data,model):
    assert isinstance(data,np.ndarray)
    data=Variable(torch.from_numpy(data),requires_grad = False)
    #assert data.shape[1]==19
    #normalize data according to original dataset to fit the model
    data=(data- dataMin) / (dataMax - dataMin)
    newmodel=model.cpu()
    return newmodel(data).item()
    


testdata=np.array(data.iloc[2039])
#print(testdata)
result=prediction(testdata,model)
print(result)

