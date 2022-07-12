# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/train.csv")

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import csv
from sklearn import metrics
from datetime import datetime

def rmsle(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
    
print "Import the dataset"
print str(datetime.now())
df=pd.read_csv('../input/train.csv', header=0)
df_tst=pd.read_csv('../input/test.csv', header=0)
X=df.iloc[:,2:].values
y=df.iloc[:,1].values.reshape(-1,1)
X1=df_tst.iloc[:,1:]

print "Normalize the data"
from sklearn.preprocessing import MinMaxScaler
minmax_X = MinMaxScaler()
minmax_y = MinMaxScaler()
X = minmax_X.fit_transform(X)
y = minmax_y.fit_transform(y)

print "Fitting Linear model"
print str(datetime.now())
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)
print "Predicting the Test set results for Linear"
print str(datetime.now())
y_pred = linreg.predict(X)
#print y
#print y_pred
print "Calculate error value MSLE"
print str(datetime.now())
err = rmsle(y, y_pred)
#err = np.sqrt(metrics.mean_squared_error(y, y_pred))
print "Linear error =",err

print "Fitting SVR to the dataset"
print str(datetime.now())
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X, y)
print "Predicting a new result for SVR"
print str(datetime.now())
y_pred1 = regressor.predict(X)
print "Calculate error value MSLE"
print str(datetime.now())
err1 = rmsle(y, y_pred1)
#err1 = np.sqrt(metrics.mean_squared_error(y, y_pred1))
print "SVR Error=",err1

print "Fitting Random Forest Regression to the dataset"
print str(datetime.now())
from sklearn.ensemble import RandomForestRegressor
rand = RandomForestRegressor(n_estimators = 10, random_state = 0)
rand.fit(X, y)
print "Predicting a new result for Random Forest"
print str(datetime.now())
y_pred2 = rand.predict(X)
print "Calculate error value MSLE"
print str(datetime.now)
err2 = rmsle(y, y_pred2)
#err2 = np.sqrt(metrics.mean_squared_error(y, y_pred2))
print "RandomForest=",err2

print "Fitting Decision Tree Regression to the dataset"
print str(datetime.now())
from sklearn.tree import DecisionTreeRegressor
dec = DecisionTreeRegressor(random_state = 0)
dec.fit(X, y)
print "Predicting a new result for Decission Treee"
print str(datetime.now())
y_pred3 = dec.predict(X)
print "Calculate error value MSLE"
print str(datetime.now())
err3 = rmsle(y, y_pred3)
#err3 = np.sqrt(metrics.mean_squared_error(y, y_pred3))
print "DecissionTree=",err3

final = linreg.predict(X1)
df_tst['target'] = final
df_tst[['ID','target']].to_csv('../input/result.csv', header=0)