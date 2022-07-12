# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:32:34 2017

@author: e3020186
"""

#import required packages
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold 
from sklearn.ensemble import RandomForestRegressor

#load the data into dataframes
#submission=pd.read_csv('D:\\Kaggle\\Trip_Duration\\sample_submission.zip')
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Explore the data
train.head()
test.head()

train.describe()
test.describe()


#Confirm if there are any nan values
train.isnull().sum()
test.isnull().sum()


#Explore the data

train['vendor_id'].unique() # Unique values 1 and 2
train['passenger_count'].unique() #Values range from 0-9
train['pickup_longitude'].unique()
test['vendor_id'].unique() # Unique values 1 and 2
test['passenger_count'].unique() #Values range from 0-9


#Clean up trip duration in train data
#We see that trip duration is 1 or 0 which is not possible. so we clean up all values that are greater that 2 standard deviations
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]

#####################################################################

#Clean up pickup_date and pickup_time

train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])

#Seperate the date and time part
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

train.loc[:, 'pickup_time'] = train['pickup_datetime'].dt.time
test.loc[:, 'pickup_time'] = test['pickup_datetime'].dt.time

train.drop('pickup_datetime', axis=1, inplace=True)
test.drop('pickup_datetime', axis=1, inplace=True)

#####################################################################

#Convert dropoff_dat_time to date format

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)


#####################################################################

#Convert store_and_fwd_flag to number value

train['store_and_fwd_flag'].replace(('Y','N'),(1,0),inplace=True)
test['store_and_fwd_flag'].replace(('Y','N'),(1,0),inplace=True)



###################################################################

target = 'trip_duration'
IDcol = ['id']

predictors = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag']

alg=RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
alg.fit(train[predictors],train[target])

test[target]=alg.predict(test[predictors])

IDcol.append(target)
#filename='D:\\Kaggle\\Trip_Duration\\sample_submission.zip'
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv('../working/submission.csv', index=False)