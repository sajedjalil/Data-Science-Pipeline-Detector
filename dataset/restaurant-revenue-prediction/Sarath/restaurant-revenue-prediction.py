#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

#date to no of days conversion
test['Days_Open'] = (pd.to_datetime('2015-03-23') - pd.to_datetime(test['Open Date'])).dt.days
test = test.drop('Open Date', axis=1)
train['Days_Open'] = (pd.to_datetime('2015-03-23') - pd.to_datetime(train['Open Date'])).dt.days
train = train.drop('Open Date', axis=1)

#Dropping City and ID from dataset 
test = test.drop('City', axis=1)
test = test.drop('Id', axis=1)
train = train.drop('City', axis=1)
train = train.drop('Id', axis=1)

#Splitting to independent and dependent Variables
X = train.drop('revenue', axis=1)
Y = train[['revenue']]


#Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X = LabelEncoder()
X['Type'] = label_X.fit_transform(X['Type'])
X['City Group'] = label_X.fit_transform(X['City Group'])
onehot = OneHotEncoder(categorical_features = [1])
X = onehot.fit_transform(X).toarray()
X = X[:,1:]

label_test = LabelEncoder()
test['Type'] = label_X.fit_transform(test['Type'])
test['City Group'] = label_X.fit_transform(test['City Group'])
onehot = OneHotEncoder(categorical_features = [1])
test = onehot.fit_transform(test).toarray()
test = test[:,1:]
test = np.delete(test,[2],1)

#dummy trap avoidence
test = np.append(arr = np.ones((100000,1)).astype(int),values = test,axis = 1)
X = np.append(arr = np.ones((137,1)).astype(int),values = X,axis = 1)


#random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)

# Predicting a new result
y_pred = regressor.predict(test)