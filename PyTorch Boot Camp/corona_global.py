#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas.plotting import register_matplotlib_converters


# In[10]:


df = pd.read_csv('time_series_covid19_confirmed_global.csv')


# In[11]:


df.head()


# In[12]:


df = df.iloc[:,37:]
df


# In[13]:


daily_global = df.sum(axis=0)


# In[14]:


daily_global.index = pd.to_datetime(daily_global.index)


# In[15]:


daily_global


# In[16]:


plt.plot(daily_global)


# In[17]:


y=daily_global.values.astype(float)


# In[18]:


test_size = 3
train_original_data = y[:-test_size]
test_original_data = y[-test_size:]


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))


# In[20]:


train_normalized = scaler.fit_transform(train_original_data.reshape(-1,1))
train_normalized.shape


# In[21]:


train_normalized = torch.FloatTensor(train_normalized).view(-1)
window_size = 3


# In[22]:


def sequence_creator(input_data,window):
    dataset = []
    data_len = len(input_data)
    for i in range(data_len - window):
        window_fr = input_data[i:i+window]
        label = input_data[i+window:i+window+1]
        dataset.append((window_fr, label))
    return dataset


# In[23]:


train_data = sequence_creator(train_normalized, window_size)


# In[33]:


class LSTM_Corona(nn.Module):
    def __init__(self, in_size=1, h_size=30, out_size=1):
        super().__init__()
        self.h_size = h_size
        self.lstm = nn.LSTM(in_size,h_size)
        self.fc = nn.Linear(h_size,out_size)

        self.hidden = (torch.zeros(1,1,h_size),torch.zeros(1,1,h_size))

    def forward(self, sequence_data):
        lstm_out, self.hidden = self.lstm(sequence_data.view(len(sequence_data),1,-1),self.hidden)
        pred=self.fc(lstm_out.view(len(sequence_data),-1))

        return pred[-1]


# In[34]:


torch.manual_seed(3)
model = LSTM_Corona()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# In[38]:


epochs = 100

for epoch in range(epochs):
    for sequence_in, y_train in train_data:
        y_pred = model(sequence_in)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1} Loss {loss.item():.3f}')


# In[40]:


test = 3

preds = train_normalized[-window_size:].tolist()

model.eval()

for i in range(test):
    sequence = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1,model.h_size), torch.zeros(1,1,model.h_size))
        preds.append(model(sequence).item())


# In[41]:


predictions = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1,1))


# In[42]:


predictions


# In[43]:


daily_global[-3:]


# In[45]:


x = np.arange('2020-04-07','2020-04-10', dtype='datetime64[D]').astype('datetime64[D]')
x


# In[46]:


plt.figure(figsize=(12,5))
plt.grid(True)
plt.plot(daily_global)
plt.plot(x,predictions)
plt.show()


# In[47]:


epochs = 200
model.train()

y_normalized = scaler.fit_transform(y.reshape(-1,1))
y_normalized = torch.FloatTensor(y_normalized).view(-1)
full_data = sequence_creator(y_normalized, window_size)


# In[48]:


for epoch in range(epochs):
    for sequence_in, y_train in full_data:

        y_pred = model(sequence_in)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1} Loss{loss.item():.3f}')


# In[49]:


future = 3

preds = y_normalized[-window_size:].tolist()

model.eval()

for i in range(future):
    sequence = torch.FloatTensor(preds[-window_size:])

    with torch.no_grad():
        model.hidden =(torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))
        preds.append(model(sequence).item())

predictions = scaler.inverse_transform(np.array(preds).reshape(-1,1))

x = np.arange('2020-04-10','2020-04-13', dtype='datetime64[D]').astype('datetime64[D]')


# In[53]:


plt.figure(figsize=(12,5))
plt.title('The number of person affected by Corona virus globally')
plt.grid(True)
plt.plot(daily_global)
plt.plot(x, predictions[window_size:])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
