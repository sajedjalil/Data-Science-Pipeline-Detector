import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import os
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

### preprocessing training data ###
def create_month(x):
    return int(x.split('-')[1])

def create_day(x):
    return int(x.split('-')[2])
    
def get_data():
    """
    reused the code from 'Beating the Benchmark' by Abhihsek with some modifications
    """
    # Load dataset 
    train = pd.read_csv('../input/train.csv')  
    test = pd.read_csv('../input/test.csv')
    sample = pd.read_csv('../input/sampleSubmission.csv')
    weather = pd.read_csv('../input/weather.csv')
    
    # Get labels
    labels = train.WnvPresent.values
    
    # Not using codesum for this benchmark
    weather = weather.drop('CodeSum', axis=1)
    
    # Split station 1 and 2 and join horizontally
    weather_stn1 = weather[weather['Station']==1]
    weather_stn2 = weather[weather['Station']==2]
    weather_stn1 = weather_stn1.drop('Station', axis=1)
    weather_stn2 = weather_stn2.drop('Station', axis=1)
    weather = weather_stn1.merge(weather_stn2, on='Date')
    
    # replace some missing values and T with -1
    weather = weather.replace('M', -1)
    weather = weather.replace('-', -1)
    weather = weather.replace('T', -1)
    weather = weather.replace(' T', -1)
    weather = weather.replace('  T', -1)
    # Functions to extract month and day from dataset
    # You can also use parse_dates of Pandas.
    
    train['month'] = train.Date.apply(create_month)
    train['day'] = train.Date.apply(create_day)
    test['month'] = test.Date.apply(create_month)
    test['day'] = test.Date.apply(create_day)

    # drop address columns
    train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
    test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

    # Merge with weather data
    train = train.merge(weather, on='Date')
    test = test.merge(weather, on='Date')
    train = train.drop(['Date'], axis = 1)
    test = test.drop(['Date'], axis = 1)

    # Convert categorical data to numbers
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train['Species'].values) + list(test['Species'].values))
    train['Species'] = lbl.transform(train['Species'].values)
    test['Species'] = lbl.transform(test['Species'].values)

    lbl.fit(list(train['Street'].values) + list(test['Street'].values))
    train['Street'] = lbl.transform(train['Street'].values)
    test['Street'] = lbl.transform(test['Street'].values)

    lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
    train['Trap'] = lbl.transform(train['Trap'].values)
    test['Trap'] = lbl.transform(test['Trap'].values)

    # drop columns with -1s
    train = train.ix[:,(train != -1).any(axis=0)]
    test = test.ix[:,(test != -1).any(axis=0)]
    train = train.drop('Depth_x',axis=1)
    test = test.drop('Depth_x',axis=1)
    train = train.drop('SnowFall_x',axis=1)
    test = test.drop('SnowFall_x',axis=1)
    return train,labels,test

######################################
##### function for computing AUC #####
######################################
"""
function for computing AUC of given predictions 'preds' and given labels 'yTrue'
@yTrue: the true labels
@preds: probability that WNV present. If preds is a two-dimensional array, the second column is used!
"""
def compute_score(yTrue,preds):
	pred =preds
	if preds.ndim>2:
		return -1
	if preds.ndim==2:
		pred = preds[:,1]
	fpr,tpr,thresholds = roc_curve(yTrue,pred)
	return auc(fpr,tpr)

##### Try on a benchmark ######
###############################
train,labels,test = get_data()

### training & testing ####
#train_x, valid_x = train.values[:8000,:],train.values[8000:,:]
#train_y, valid_y = labels[:8000],labels[8000:]

train_x,valid_x,train_y,valid_y = train_test_split(train,labels,test_size=0.25,random_state=0)
clf = ensemble.AdaBoostClassifier(n_estimators=1000)
print ("training RF on %d samples "%len(train_y))
clf.fit(train_x,train_y)
preds = clf.predict_proba(valid_x)
print ("testing RF on %d samples "%len(valid_y))
score = compute_score(valid_y,preds) ### compute_score(valid_y,preds[:,1])
print ("valid score is %8f"%score)


