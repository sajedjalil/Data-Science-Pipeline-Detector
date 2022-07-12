import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from numpy import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import datetime
from sklearn.metrics import log_loss


fig = plt.figure()
traindata=pd.read_json("../input/train.json")
testdata=pd.read_json("../input/test.json")
train=traindata[['building_id','photos','features','manager_id','bathrooms','bedrooms','created','latitude','longitude','price','interest_level']]
test=testdata[['building_id','photos','features','manager_id','bathrooms','bedrooms','created','latitude','longitude','price']]
train=train[np.abs(train.latitude-train.latitude.mean())<=(3*train.latitude.std())] 
train=train[np.abs(train.longitude-train.longitude.mean())<=(3*train.longitude.std())]
train=train[np.abs(train.bedrooms-train.bedrooms.mean())<=(3*train.bedrooms.std())] 
train=train[np.abs(train.bathrooms-train.bathrooms.mean())<=(3*train.bathrooms.std())]

train=train[np.abs(train.price-train.price.mean())<=(3*train.price.std())]
train["created"] = pd.to_datetime(train["created"])
train["created_year"] = train["created"].dt.year
train["created_month"] = train["created"].dt.month
train["created_day"] = train["created"].dt.day
train["created_hour"] = train["created"].dt.hour

test["created"] = pd.to_datetime(test["created"])
test["created_year"] = test["created"].dt.year
test["created_month"] = test["created"].dt.month
test["created_day"] = test["created"].dt.day
test["created_hour"] = test["created"].dt.hour


target_map={"high":0, "medium":1, "low":2}
train['interest_level']=train["interest_level"].apply(lambda x: target_map[x])
train['photos']=[len(x) for x in train['photos']]
train['features']=[len(x) for x in train['features']]
test['photos']=[len(x) for x in test['photos']]
test['features']=[len(x) for x in test['features']]
#train.to_csv('cleaned.csv',c)

Cleaned_train_features=train[['building_id','photos','features','manager_id','bathrooms','bedrooms','created_year','created_month','created_day','created_hour','latitude','longitude','price']]
data_test=test[['building_id','photos','features','manager_id','bathrooms','bedrooms','created_year','created_month','created_day','created_hour','latitude','longitude','price']]
Cleaned_train_features['type']="train"
data_test['type']="test"
totaldata=Cleaned_train_features.append(data_test,ignore_index=True)
totaldata['building_id']=pd.Categorical(totaldata.building_id).codes
totaldata['manager_id']=pd.Categorical(totaldata.manager_id).codes


trd=totaldata[(totaldata['type']=='train')]
ted=totaldata[(totaldata['type']=='test')]

Cleaned_train_features=trd[['building_id','photos','features','manager_id','bathrooms','bedrooms','created_year','created_month','created_day','created_hour','latitude','longitude','price']]
Cleaned_train_labels=train['interest_level']
data_test=ted[['building_id','photos','features','manager_id','bathrooms','bedrooms','created_year','created_month','created_day','created_hour','latitude','longitude','price']]


#Considering 20% cross validation data from the train data
data,labels = Cleaned_train_features,Cleaned_train_labels
data_train, train_test_data, labels_train, labels_test = train_test_split(data,labels,test_size=0.20)
TrainData_Target= labels_train.as_matrix()


#Calculating score on the train data 
#rfc_train = RandomForestClassifier(class_weight='balanced',n_estimators=1000,criterion='entropy')

rfc_train = RandomForestClassifier(class_weight='balanced',n_estimators=1000,criterion='entropy')	
rfc_train.fit(data_train,TrainData_Target)
RandomForest_Predictions=rfc_train.predict(train_test_data)

Accurate_score_rf=accuracy_score(RandomForest_Predictions,labels_test)

score=rfc_train.score(train_test_data,labels_test)
predict_probabilities=rfc_train.predict_proba(train_test_data)
print(log_loss(labels_test,predict_probabilities))
print(score)
print(Accurate_score_rf)
print(rfc_train)
print(confusion_matrix(labels_test,RandomForest_Predictions))


#Building model on the test data
TrainData_Target= Cleaned_train_labels.as_matrix()
rfc = RandomForestClassifier(class_weight='balanced',n_estimators=1000,criterion='entropy')
rfc.fit(Cleaned_train_features,TrainData_Target)
RandomForest_Predictions=rfc.predict(data_test)
print(RandomForest_Predictions)
print(rfc.predict_proba(data_test))	
result=pd.DataFrame(rfc.predict_proba(data_test),columns=['high','medium','low'])
td=testdata[['listing_id']].reset_index()
result['listing_id']=td['listing_id']
result.to_csv("submission.csv")
