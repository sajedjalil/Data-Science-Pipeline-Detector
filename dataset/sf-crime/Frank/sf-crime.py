# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import math
import os
from math import log
#from pyspark import SparkContext
#from pyspark.sql import SQLContext
from zipfile import ZipFile

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

if __name__ == '__main__':
   classes = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']
   PdDistricts = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
   WeekDay_dict = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
   kf = KFold(n_splits=5, shuffle=True)
   
   #zf_train = ZipFile('./test.csv.zip')
   #zf_test = ZipFile('./test.csv.zip')
   train = pd.read_csv('../input/train.csv',parse_dates=['Dates'])
   test = pd.read_csv('../input/test.csv',parse_dates=['Dates'])
   
   train['Category'] = train['Category'].map(lambda c: classes.index(c)).astype('category')
   train['PdDistrict'] = train['PdDistrict'].map(lambda c: PdDistricts.index(c)).astype('category')
   train['DayOfWeek'] = train['DayOfWeek'].replace(WeekDay_dict).astype('category')
   train['Year'] = train['Dates'].map(lambda x : x.year).astype('category')
   train['Month'] = train['Dates'].map(lambda x : x.month).astype('category')
   train['Day'] = train['Dates'].map(lambda x : x.day).astype('category')
   train['Hour'] = train['Dates'].map(lambda x : x.hour)
   train['x5'] = train['X'].map(lambda x : round(x,5))
   train['y5'] = train['Y'].map(lambda x : round(x,5))
   train['x4'] = train['X'].map(lambda x : round(x,4))
   train['y4'] = train['Y'].map(lambda x : round(x,4))
   train['x3'] = train['X'].map(lambda x : round(x,3))
   train['y3'] = train['Y'].map(lambda x : round(x,3))
   train['x2'] = train['X'].map(lambda x : round(x,2))
   train['y2'] = train['Y'].map(lambda x : round(x,2))
   
   #test['Category'] = test['Category'].map(lambda c: classes.index(c)).astype('category')
   test['PdDistrict'] = test['PdDistrict'].map(lambda c: PdDistricts.index(c)).astype('category')
   test['DayOfWeek'] = test['DayOfWeek'].replace(WeekDay_dict).astype('category')
   test['Year'] = test['Dates'].map(lambda x : x.year).astype('category')
   test['Month'] = test['Dates'].map(lambda x : x.month).astype('category')
   test['Day'] = test['Dates'].map(lambda x : x.day).astype('category')
   test['Hour'] = test['Dates'].map(lambda x : x.hour)
   test['x5'] = test['X'].map(lambda x : round(x,5))
   test['y5'] = test['Y'].map(lambda x : round(x,5))
   test['x4'] = test['X'].map(lambda x : round(x,4))
   test['y4'] = test['Y'].map(lambda x : round(x,4))
   test['x3'] = test['X'].map(lambda x : round(x,3))
   test['y3'] = test['Y'].map(lambda x : round(x,3))
   test['x2'] = test['X'].map(lambda x : round(x,2))
   test['y2'] = test['Y'].map(lambda x : round(x,2))
   #train = train.drop(['Dates','Descript', 'Resolution', 'Address'], axis = 1)
   feats = ['Category','Year','Month','Day','Hour','DayOfWeek','PdDistrict','X','Y','x5','y5','x4','y4','x3','y3','x2','y2']
   #train[feats].to_csv('train_features.csv', index = False)
   #test[feats[1:]].to_csv('test_features.csv', index = False)
   print (train.head())
   print (train.dtypes)
   
   #if os.path.isfile('./train_features.csv.zip'):
   #  zf = ZipFile('./train_features.csv.zip')
   #  train = pd.read_csv(zf.open('train_features.csv'))
   
   
   print ('==================Features Extracted==================')
   #print (train.head())
   #print (train.dtypes)
   ##corr = train_data.corr()
   ##print(corr)
   
   feats = ['Year','Month','Day','Hour','DayOfWeek','PdDistrict','X','Y','x5','y5']
   #nrow, ncol = train_data.shape[0], train_data.shape[1]
   #print (nrow, ncol)
   for train_index, valid_index in kf.split(train):
      print("%s %s" % (train_index, valid_index))
   #train, valid = train_data.iloc[train_index,:], train_data.iloc[valid_index,:]
   #feats = ['Year', 'Date', 'Hour', 'DayOfWeek', 'PdDistrict']
   train_x, train_y = train.iloc[train_index,:][feats], train.iloc[train_index,:]['Category']
   valid_x = train.iloc[valid_index,:][feats]
   #print (train_x.head())
   valid_y = pd.DataFrame()
   valid_y['Category'] = train.iloc[valid_index,:]['Category']
   valid_y['Id'] = range(0,len(valid_x))
   #print (valid_y.head())
   
   del train
   print ('==================Raw Train Data splitted==================')
   rfc = RandomForestClassifier(n_estimators=75, max_depth=18)
   rfc.fit(train_x, train_y)
   
   del train_x, train_y
   print ('==================Classifier Trained==================')
   rfc_probs = rfc.predict_proba(valid_x)
   valid_y['Predict'] = rfc.predict(valid_x)
   
   results = valid_y.apply(lambda x: x['Predict']==x['Category'], axis=1)
   test_accuracy = sum(results)/len(valid_x)
   test_error_rate = 1 - test_accuracy
   print (test_accuracy, test_error_rate)
   #result = pd.DataFrame(rfc.predict_proba(valid_x), index=valid_x.index, columns=rfc.classes_)
   #print(result)
   print ('==================Classifier Tested==================')
   [m, n] = rfc_probs.shape
   print (m, n, len(classes))
   if n<len(classes):
      rfc_probs = np.append(rfc_probs, np.zeros([m,len(classes)-n]),1)
   np.savetxt('valid_submission.csv', rfc_probs, delimiter=',')
   #rfc_probs.to_csv('valid_submission.csv', index = False)
   
   def rep_extreme(array):
      [m, n] = array.shape
      for i in range(0,m):
         for j in range(0,n):
            array[i][j] = max(min(array[i][j],1-10**(-15)),10**(-15))
      return array
   rfc_probs = rep_extreme(rfc_probs)
   #print (rfc_probs)
   
   # internal logloss function
   def logloss(actual, predicted):
      n_class = len(actual)
      n_sample = predicted.shape[0]
      sum = 0
      for j in range(0,n_sample):
         real_class = actual[j]
         #print (predicted[j][real_class])
         sum += log(predicted[j][real_class])
      return -sum/n_sample
   score = logloss(valid_y['Category'].as_matrix(), rfc_probs)
   print (score)
   print ('Done!')