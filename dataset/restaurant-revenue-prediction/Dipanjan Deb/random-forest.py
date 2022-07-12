# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
#%%

data_dir = "../input/"
trainData = pd.read_csv(data_dir + 'train.csv', parse_dates = [1])
testData = pd.read_csv(data_dir + 'test.csv', parse_dates = [1])

#%%  
numerical_label = ['P' + str(i) for i in range(1,38)]
  
trainData['age'] = (datetime.now() - trainData['Open Date']).astype('timedelta64[D]') / 365   
trainData['Type'] = LabelEncoder().fit_transform(trainData['Type'])
trainData['City Group'] = LabelEncoder().fit_transform(trainData['City Group'])


testData['age'] = (datetime.now() - testData['Open Date']).astype('timedelta64[D]') / 365   
testData['Type'] = LabelEncoder().fit_transform(testData['Type'])
testData['City Group'] = LabelEncoder().fit_transform(testData['City Group'])

#%%
X_names = numerical_label + ['age', 'Type', 'City Group']
clf=RandomForestRegressor(n_estimators=1000, max_features=2)
X_train = trainData[X_names]
y_train = trainData['revenue']

clf.fit(X_train, y_train)

X_test = testData[X_names]
pred = clf.predict(X_test)

#%%
result = pd.read_csv(data_dir + 'sampleSubmission.csv')
result['Prediction'] = pred
result.to_csv('result_Py.csv',index=False)
