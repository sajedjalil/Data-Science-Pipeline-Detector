# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


df = pd.read_csv('../input/train.csv')


target = np.log1p(df['target'])

df = df.drop(['ID','target'], axis=1)

#Getting the columns with only one None zero value
unique_df = df.nunique().reset_index()
constant_df = unique_df[unique_df[0]==1]
constant_df = constant_df['index'].tolist()

#Droping the columns with only one None zero value
df = df.drop(constant_df, axis=1)

column = df.columns

#Creating 100 new columns by adding batch of 47 columns as one column
a = 0
b = 47
for i in range(0,100):
    if b>len(column):
        df[i] = df.iloc[:,a:].sum(1)
    else:
        df[i] = df.iloc[:,a:b].sum(1)
    a = a + 47
    b = b + 47

df = df.drop(column, axis=1)
columns_1 = df.columns

for i in columns_1:
    df[i] = np.log1p(df[i]+1)
   
    
#Selecting column 0 - 100
train = df[columns_1]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=45)

#XGBOOST
boost = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 
                             gamma=0.75, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=755, reg_alpha=0, 
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7)
boost.fit(X_train, y_train)
y = boost.predict(X_test)
print(min(y), max(y), y.mean())
print(min(y_test), max(y_test), y_test.mean())
print("XGBOOST Score:")
print(np.sqrt(mean_squared_error(y_test, y)))

#RandomeForest
model = RandomForestRegressor(n_estimators=900, max_features='log2', random_state =9)
model.fit(X_train, y_train)
y_r = model.predict(X_test)
print(min(y_r), max(y_r), y_r.mean())
print(min(y_test), max(y_test), y_test.mean())
print("RandomeForest Score:")
print(np.sqrt(mean_squared_error(y_test, y_r)))

#Averaing the xgboos and Randomforest
y_result = (y * 6) + (y_r * 4)
y_result = y_result/10
print(min(y_result), max(y_result), y_result.mean())
print(min(y_test), max(y_test), y_test.mean())
print("Average Score:")
print(np.sqrt(mean_squared_error(y_test, y_result)))