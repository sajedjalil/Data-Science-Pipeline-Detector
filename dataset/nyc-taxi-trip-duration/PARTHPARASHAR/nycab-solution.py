#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:04:17 2017

@author: parth
"""
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
train_df=pd.read_csv("../input/train.csv")
#print(test_df)../
test_df=pd.read_csv("../input/test.csv")
train_df=train_df[:625134]
use=test_df["id"]
#combine=[train_df,test_df]
#rint(train_df.tail())
df1=train_df[["passenger_count", "trip_duration"]].groupby(['passenger_count'], as_index=False).mean().sort_values(by='trip_duration', ascending=False)
train_df=train_df.drop(["id",'pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
test_df=test_df.drop(["id",'pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)

#df2=train_df[["trip_duration","passenger_count"]].head()
#print(test_df)

#print(train_df.dtypes)
#for dataset in train_df:
#    if train_df['store_and_fwd_flag'].any() is 'N':
#        train_df['store_and_fwd_flag']=train_df['store_and_fwd_flag'].replace('N',0).astype(float)
#    elif train_df['store_and_fwd_flag'].any() is 'Y':
#        train_df['store_and_fwd_flag']=train_df['store_and_fwd_flag'].replace('Y',1).astype(float)
combine = [train_df, test_df]
for dataset in combine:
    dataset['store_and_fwd_flag'] = dataset['store_and_fwd_flag'].map( {'N': 0, 'Y': 1} ).astype(int)
#for dataset in test1_df:
#    if test_df['store_and_fwd_flag'].any() is 'N':
#        test_df['store_and_fwd_flag']=test_df['store_and_fwd_flag'].replace('N',0)
#    elif test_df['store_and_fwd_flag'].any() is 'Y':
#        test_df['store_and_fwd_flag']=test_df['store_and_fwd_flag'].replace('Y',1)
#print(train_df)
        #for dataset in train_df:
#    dataset['store_and_fwd_flag'] = dataset['store_and_fwd_flag'].map( {'N': 1, 'Y': 0} ).astype(int)
#print(train_df,test_df)
print("hello1")
x_train = train_df.drop("trip_duration", axis=1)
y_train = train_df["trip_duration"]
x_test  = test_df
#print(x_test)
print("hello11")
#rf=RandomForestRegressor(n_estimators=50,max_depth=50,min_samples_split=10)
#rf.fit(x_train,y_train)
#print("hello111")
##y_pred=rf.predict(x_test)
#print("hello1111")
#
#acc_random_forest = round(rf.score(x_train, y_train) * 100, 2)
#
#print(acc_random_forest)
#linear_svc = LinearSVC()
#linear_svc.fit(x_train, y_train)
#print("hello11111")
##Y_pred = linear_svc.predict(X_test)
#acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
#print(acc_linear_svc)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
print("hello111111")
y_pred = knn.predict(x_test)
#acc_knn = round(knn.score(x_train, y_train) * 100, 2)
#print("acc_knn")
#print(acc_random_forest,acc_linear_svc ,acc_knn)

submission = pd.DataFrame({
        "id": use,
        "trip_duration": y_pred
    })
submission.to_csv('../output/submission3.csv', index=False)
