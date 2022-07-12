# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression 
from haversine import haversine

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


train['distance'] = train.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),(x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)

test['distance'] = test.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),(x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)

feature_col=["distance"]
x_train=train[feature_col]
y_train=train['trip_duration'].values
x_pred=test[feature_col]
lm=LinearRegression()
lm.fit(x_train, y_train)
y_pred=lm.predict(x_pred)
# Any results you write to the current directory are saved as output.

result=pd.DataFrame()
result['id']=test['id']
result['trip_duration']=y_pred
result.to_csv('lm.csv', index=False)