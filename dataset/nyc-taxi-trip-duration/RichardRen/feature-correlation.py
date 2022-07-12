# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from haversine import haversine
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv("../input/train.csv")
train.head()

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
train['distance'] = train.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]), (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
train["pickup_weekday"] = train["pickup_datetime"].dt.weekday
train["pickup_hour"] = train["pickup_datetime"].dt.hour
train["pickup_month"] = train["pickup_datetime"].dt.month
train["store_and_fwd_flag"] = train["store_and_fwd_flag"].map(lambda x: int(x=='N'))
train["weekend"] = train["pickup_weekday"].map(lambda x: int(x==5 or x==6))



feature_cols = ["vendor_id", "passenger_count", "pickup_month", "pickup_weekday", "pickup_hour", "distance",
               "weekend",
               "store_and_fwd_flag"]
               
for feature in feature_cols:
    pearson_correlation=train[feature].corr(train['trip_duration'], method='pearson')
    spearman_correlation= train[feature].corr(train['trip_duration'], method='spearman')
    print(feature+ ": pearson-<" + str(pearson_correlation) + "> spearman-<"+str(spearman_correlation)+ ">")




#print("hello")
# Any results you write to the current directory are saved as output.