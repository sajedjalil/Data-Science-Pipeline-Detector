
"""
Created on Mon Sep 18 15:49:13 2017

@author: sudhir
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sy

#Import data set
train = pd.read_csv('../input/train_2017.csv')
properties= pd.read_csv('../input/properties_2017.csv',low_memory=False)
sample = pd.read_csv('../input/sample_submission.csv')

for c,dtype in zip(properties.columns,properties.dtypes):
    if dtype == np.float64:
        properties[c]=properties[c].astype(np.float32)
train.head()
properties.head()

train_df= pd.merge(train,properties,on='parcelid',how='left')
train_df.info()
train_df.fillna(0)

#traget variable
sns.distplot((train['logerror']))
plt.show()
for c in train_df.dtypes[train_df.dtypes == object].index.values:
    train_df[c] = (train_df[c] == True)
#

X = train_df.drop(['parcelid','logerror','transactiondate', 
                         'propertyzoningdesc', 'propertycountylandusecode'],axis=1)
y = train_df['logerror'].values.astype(np.float64)

print('training linear regression model','--'*10)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lm= LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.7,random_state=100)
train_columns = x_train.columns

lm.fit(x_train.fillna(0),y_train)
prd=lm.predict(x_test.fillna(0))

# print('R2:',lm.score(x_test,y_test))
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(properties, on='parcelid', how='left')

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
prd = lm.predict(x_test.fillna(0))

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = prd

print('Writing csv ...')
sub.to_csv('linear_model.csv', index=False, float_format='%.4f')
