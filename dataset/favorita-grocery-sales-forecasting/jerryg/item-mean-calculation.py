# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import gc

dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}

data_train = pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date'],usecols=[1,2,3,4]
                    ,skiprows=range(1, 6000000) #Skip dates before 2016-08-01
                    )
print("Proceed data_train")
# print(len(data_train[["item_nbr"]].unique()))
   
# In[9]:


data_item=pd.read_csv('../input/items.csv')
print("Proceed data_item")
# In[11]:


data_train.loc[(data_train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
data_train["unit_sales"].fillna(0,inplace=True)
data_train['unit_sales'] =  data_train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
data_train['dow'] = data_train['date'].dt.dayofweek
print("Proceed Conversion")

# # calculate mean

# In[12]:


def getBase(indexs,df_org,col_name):
    df_tmp_base=df_org[indexs+['unit_sales']].groupby(indexs)['unit_sales'].mean().to_frame(col_name)
    df_tmp_base.reset_index(inplace=True)
    return df_tmp_base


# # Item

# In[13]:


df_train=pd.merge(data_train[["item_nbr","unit_sales"]],data_item,on=["item_nbr"])
print("Proceed df_train")


# In[17]:


df_item=pd.merge(data_item,getBase(['item_nbr'],data_train,'itemBase'),how="left",on="item_nbr")
del data_item
gc.collect()
print("Proceed df_item")

# In[19]:


df_item=pd.merge(df_item,df_item[["itemBase","family"]].groupby(["family"])["itemBase"].mean().to_frame('ifBase'),how="left",left_on="family",right_index=True)
print("Proceed Family-Item")
df_item=pd.merge(df_item
                 ,df_train[["unit_sales","family"]].groupby(["family"])["unit_sales"].mean().to_frame('familyBase')
                ,how="left",left_on="family",right_index=True
                )
print("Proceed Family")

# In[20]:


df_item=pd.merge(df_item
                 ,df_train[["unit_sales","family","class"]].groupby(["family","class"])["unit_sales"].mean().to_frame('fcBase')
                ,how="left",left_on=["family","class"],right_index=True
                )
print("Proceed Family-Item-Class")

# In[21]:


df_item=pd.merge(df_item,df_item[["itemBase","class"]].groupby(["class"])["itemBase"].mean().to_frame('icBase'),how="left",left_on="class",right_index=True)
print("Proceed Class-Item")
df_item=pd.merge(df_item
                 ,df_train[["unit_sales","class"]].groupby(["class"])["unit_sales"].mean().to_frame('classBase')
                ,how="left",left_on="class",right_index=True
                )
print("Proceed Class")

# In[22]:


df_item=df_item.drop(["family","class"],axis=1)


# In[23]:


df_train=df_train.drop(["family","class"],axis=1)
print("Proceed DropCol")

# In[26]:


df_item.to_csv("items.csv.gz",index=False,compression="gzip")
print("Proceed Write File")

# In[27]:


print(df_item.isnull().sum())