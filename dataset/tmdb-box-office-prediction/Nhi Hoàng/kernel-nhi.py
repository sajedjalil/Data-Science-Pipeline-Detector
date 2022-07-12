# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2.) Import datasets
original_df_trainval = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
original_df_test_X = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")

# DATA PREPARATION TO EXPORT
df_trainval_y_temp = original_df_trainval[['id','revenue']].set_index('id')
# a.) Use the `id` feature as the index column of the data frame
df_train_temp = original_df_trainval.set_index('id')
    # b.) Only use easy to process features
    #  Warning: huge information loss here, you should propably include more features in your production code.
df_train_temp = df_train_temp[['budget', 'original_language' ,'popularity', 'runtime', 'status']]   
    # c.) One-Hot-Encoding for all nominal data
#df_train_temp = pd.get_dummies(df_train_temp)
    # d.) The `runtime` feature is not filled in 2 of the rows. We replace those empty cells / NaN values with a 0.    #  Warning: in production code, please use a better method to deal with missing cells like interpolation or additional `is_missing` feature columns.
df_train_temp = df_train_temp.fillna(0)
# a.) Use the `id` feature as the index column of the data frame
df_test_temp = original_df_test_X.set_index('id')
    # b.) Only use easy to process features
    #  Warning: huge information loss here, you should propably include more features in your production code.
df_test_temp = df_test_temp[['budget', 'original_language' ,'popularity', 'runtime', 'status']]   
    # c.) One-Hot-Encoding for all nominal data
#df_test_temp = pd.get_dummies(df_test_temp)
df_test_temp = df_test_temp.fillna(0)
#df_train_temp, df_test_temp = df_train_temp.align(df_test_temp, join='outer', axis=1, fill_value=0)

# Copy dataset
X_trainval= df_train_temp
y_train= df_trainval_y_temp['revenue']
X_test= df_test_temp
X_test_copy= df_test_temp
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:44:14 2019

@author: ASUS
"""

#%%
#%% Import library

import seaborn as sns
from sklearn.linear_model import LinearRegression


#%% EDA
X_trainval.info() # Complete
X_trainval.head()
X_trainval.drop(['original_language','status'],axis = 1,inplace = True)
X_test.drop(['original_language','status'],axis = 1,inplace = True)
#%% split Train-test
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_train,test_size=0.3,
                                               random_state=0)
#%%model
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)

#%% evaluation
ypred=regressor.predict(X_val)
from sklearn import metrics
print('RMSE',np.sqrt(metrics.mean_squared_error(y_val,ypred)))
print('R2',metrics.explained_variance_score(y_val,ypred))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_val,ypred)))
print('R2',1-metrics.mean_squared_error(y_val,ypred)/np.var(y_val))
#%%
#y_train_pred=regressor.predict(X_train)
#y_val_pred=regressor.predict(X_val)
y_test_pred=regressor.predict(X_test)
#%%
df_test=X_test.assign(revenue=y_test_pred)
df_test_y=pd.DataFrame({'id':df_test.index,'revenue':df_test['revenue']}).set_index('id')
df_test_y.to_csv('submission.csv')
pd.read_csv('submission.csv').head(5)
