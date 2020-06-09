#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0,4.0,5.0]
y_data = [3.0,6.0,9.0,12.0,15.0]

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

w_list = []
MSE_list = []

for w in np.arange(0.0,6.5,0.5):
    print("w=", w)
    loss_sum=0
    
    for x,y in zip(x_data,y_data):
        
        y_pred = forward(x)
        loss_val = loss(x,y)
        loss_sum += loss_val
        print("\t",x,y,y_pred,loss_val)
        
    print("MSE=", loss_sum/len(x_data))
    w_list.append(w)
    MSE_list.append(loss_sum/len(x_data))
    
plt.plot(w_list, MSE_list)
plt.ylabel('MSE')
plt.xlabel('w')
plt.show()


# In[ ]:


x_data = np.array([1.0,2.0,3.0,4.0,5.0], dtype=np.float32)
y_data = np.array([3.0,6.0,9.0,12.0,15.0], dtype=np.float32)


# In[ ]:


w = 1.0
w_list = [w]
loss_list = []

def forward(x):
    return x*w

def loss(y,y_pred):
    return ((y_pred - y)**2).mean()

def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()


# In[ ]:


epochs = 10

for epoch in range(epochs):
    y_pred = forward(x_data)
    loss_val = loss(y_data,y_pred)
    
    grad_val = gradient(x_data,y_data,y_pred)
    
    w = w - 0.01*grad_val
    
    w_list.append(w)
    loss_list.append(loss_val)
    
    print(f'epoch {epoch}: w = {w:.3f}: loss {loss_val:.3f}')


# In[ ]:


plt.plot(loss_list)


# In[ ]:


plt.plot(w_list)


# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# In[2]:


class Model(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred


# In[3]:


torch.manual_seed(3)

model = Model(1,1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


# In[4]:


X = torch.tensor([1,2,3,4,5],dtype=torch.float).view(-1,1)
y = torch.tensor([3,6,9,12,15],dtype=torch.float).view(-1,1)


# In[10]:


epochs = 1000
loss_list = []
w_list = []

for epoch in range(epochs):
    
    y_pred = model.forward(X)
    loss_val = criterion(y_pred,y)
    
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    
    loss_list.append(loss_val)
    w_list.append(model.linear.weight.item())


# In[11]:


plt.plot(loss_list)


# In[12]:


plt.plot(w_list)


# In[13]:


w_list[-1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




