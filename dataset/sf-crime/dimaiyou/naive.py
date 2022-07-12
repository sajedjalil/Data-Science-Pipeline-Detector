import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
import csv
from sklearn import metrics
from sklearn import preprocessing

def loadTrainData(df):
    data = df.get(['X','Y']) #2
    data = data.join(pd.get_dummies(df['DayOfWeek'])) #7
    data = data.join(pd.get_dummies(df['PdDistrict'])) #10
    date_time = pd.to_datetime(df.Dates)
    year = date_time.dt.year
    year.name = 'Year'
    month = date_time.dt.month
    month.name = 'Month'
    day = date_time.dt.day
    day.name = 'Day'
    hour = date_time.dt.hour
    minute = date_time.dt.minute
    time = hour*60+minute
    time.name = 'Time'
    #data = data.join(pd.get_dummies(year)) #13
    #data = data.join(pd.get_dummies(month)) #12
    #data = data.join(time) #1
    data = data.join([year,month,day,time]) #4
    X = data.values
    Y = df.Category.values
    scaler = None
    scaler = preprocessing.StandardScaler().fit(data.values)
    X = scaler.transform(data.values)
    return X,Y,scaler

def loadTestData(df,scaler=None):
    data = df.get(['X','Y']) #2
    data = data.join(pd.get_dummies(df['DayOfWeek'])) #7
    data = data.join(pd.get_dummies(df['PdDistrict'])) #10
    date_time = pd.to_datetime(df.Dates)
    year = date_time.dt.year
    year.name = 'Year'
    month = date_time.dt.month
    month.name = 'Month'
    day = date_time.dt.day
    day.name = 'Day'
    hour = date_time.dt.hour
    minute = date_time.dt.minute
    time = hour*60+minute
    time.name = 'Time'
    #data = data.join(pd.get_dummies(year)) #13
    #data = data.join(pd.get_dummies(month)) #12
    #data = data.join(time) #1
    data = data.join([year,month,day,time]) #4
    
    Xhat = data.values
    if scaler is not None:
        Xhat = scaler.transform(Xhat)
    return Xhat
    
def RF(X,Y):
    clf = ensemble.RandomForestClassifier()
    #cross validation
    #n_estimators = [20]
    #clf.set_params(n_estimators=500)
    max_features = ['auto'] #['auto', 'sqrt',None,'log2']
    #clf.set_params(max_features=None)
    #bootstrap = [True,False]
    #clf.set_params(bootstrap=False)
    #criterion = ['gini','entropy']
    #clf.set_params(criterion='entropy')
    #min_samples_split = [9,11,12,13,14,15]
    #clf.set_params(min_samples_split=12)
    #min_samples_leaf = [1,2,3,4,5]
    scoreSum = 0
    kfold = cross_validation.KFold(len(X),n_folds=4)
    for train, test in kfold:
        clf.set_params(max_features=max_features[0])
        clf.fit(X[train],Y[train])
        Yhat = clf.predict_proba(X[test])
        score = metrics.log_loss(Y[test],Yhat)
        print(score)
        scoreSum += score
    print(scoreSum/4.0)
    '''for i in range(len(max_features)):#
        clf.set_params(max_features=max_features[i])##
        score = cross_validation.cross_val_score(clf,X,Y,cv=4,scoring='log_loss')
        print(score)
        avgScore = np.mean(score)
        #print(max_features[i],avgScore)#
        if minScore>avgScore:
            minScore = avgScore
            best_param=max_features[i]#
    print(best_param,minScore)'''
    return clf

def RFpredict(X,Y,Xhat):
    clf = ensemble.RandomForestClassifier()
    #clf.set_params(n_estimators=20)
    #clf.set_params(max_features=None)
    #clf.set_params(criterion='entropy')
    #clf.set_params(min_samples_split=12)
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf

import os
print(os.listdir('../input'))
mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")

'''train = pd.read_csv("../input/train.csv")
X,Y,scaler = loadTrainData(train)
clf = RF(X,Y)'''

'''test = pd.read_csv("../input/test.csv")
Xhat = loadTestData(test,scaler)
Yhat,clf = RFpredict(X,Y,Xhat)

with open('RF2.csv','w') as fw:
    writer = csv.writer(fw,lineterminator='\n')
    writer.writerow(['Id']+clf.classes_.tolist())
    
    for i in range(len(Yhat)):
        writer.writerow([i]+Yhat[i].tolist())'''










