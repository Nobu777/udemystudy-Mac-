#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


df


# In[10]:


class Model(nn.Module):
    def __init__(self,in_features=4,h1=8,h2=9,out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# In[5]:


X = df.drop('target',axis=1).values


# In[6]:


y = df['target'].values


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)


# In[8]:


y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# In[11]:


torch.manual_seed(3)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# In[12]:


epochs = 100
loss_list = []

for epoch in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss)
    
    if (epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1} loss:{loss.item():.4f}')


# In[13]:


plt.plot(loss_list)


# In[15]:


with torch.no_grad():
    predicted_y = model.forward(X_test)
    loss = criterion(predicted_y, y_test)
print(loss.item())


# In[16]:


torch.save(model.state_dict(),'IrisClassificationModel.pt')


# In[17]:


model.eval()


# In[18]:


new_iris = torch.tensor([5.6,3.7,2.1,0.7])


# In[19]:


with torch.no_grad():
    print(model(new_iris))
    print(model(new_iris).argmax())


# In[20]:


new_model = Model()
new_model.load_state_dict(torch.load('IrisClassificationModel.pt'))
new_model.eval()


# In[21]:


with torch.no_grad():
    print(new_model(new_iris))
    print(new_model(new_iris).argmax())


# In[ ]:




