# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection, preprocessing
from sklearn.linear_model import HuberRegressor
import xgboost as xgb

import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#read input files
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
macro_df=pd.read_csv('../input/macro.csv')
id_test=test_df.id
#print shape of all three files

#print(train_df.shape)

#print(test_df.shape)

#print(macro_df.shape)

# from shapes we infer that test data is just over 25 percent of train_ data

# check for na

#NAs =pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train', 'Test'])
#print(NAs[NAs.sum(axis=1) > 0].sort_values(by='Train', ascending=False))

train_df.drop(['hospital_beds_raion',                    
'build_year',
'state',       
'cafe_sum_500_min_price_avg',     
'cafe_sum_500_max_price_avg',       
'cafe_avg_price_500'],axis=1,inplace=True )                    

test_df.drop(['hospital_beds_raion',                    
'build_year',
'state',       
'cafe_sum_500_min_price_avg',     
'cafe_sum_500_max_price_avg',       
'cafe_avg_price_500'],axis=1,inplace=True ) 

y_train = train_df["price_doc"]
x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test_df.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)
        

x_train.fillna(x_train.mean(),inplace=True)
x_test.fillna(x_test.mean(),inplace=True)

model=HuberRegressor()

model.fit(x_train,y_train)



y_predict = model.predict(x_test)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


output.to_csv('300520171500.csv', index=False)