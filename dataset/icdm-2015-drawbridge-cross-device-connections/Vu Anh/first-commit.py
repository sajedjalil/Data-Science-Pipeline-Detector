# In[31]:

import pandas as pd


# In[3]:

cookies = pd.read_csv('../input/cookie_all_basic.csv')


# In[4]:

cookies


# In[5]:

test = pd.read_csv('../input/dev_test_basic.csv')


# In[6]:

test


# In[7]:

train = pd.read_csv('../input/dev_train_basic.csv')


# In[8]:

train


# In[13]:

data = pd.merge(train, cookies, left_on='drawbridge_handle', right_on='drawbridge_handle', how='inner')


# In[14]:

data


# In[18]:

data['cookie_id'].value_counts()


# In[26]:

predict = pd.DataFrame({'device_id': test['device_id'], 'cookie_id': 'id_3908095'})


# In[38]:

predict.to_csv('submision.csv', header=True, index=False)


# In[ ]:
