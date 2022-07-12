# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

# coding: utf-8

# In[ ]:





# In[12]:

destination=pd.read_csv("../input/destinations.csv")
train=pd.read_csv("../input/train.csv", parse_dates=['srch_ci', 'srch_co'],nrows=10000)

test=pd.read_csv("../input/test.csv",parse_dates=['srch_ci', 'srch_co'],nrows=10000)

train.head(n=3)


# In[13]:

destination.head(n=3)


# In[15]:




features=['site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance',  'is_mobile', 'is_package',
       'channel', 'srch_adults_cnt',
       'srch_children_cnt', 'srch_rm_cnt', 
        'is_booking', 'cnt', 'hotel_continent',
       'hotel_country', 'hotel_market']

tmp=train[train['is_booking']==1]

#drop all the na observation rows
tmp=tmp.dropna(axis=0)
len(tmp)

#defining the parameters of RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=5)
#fitting the model on the dataset
rf.fit(tmp[features],tmp["hotel_cluster"])


