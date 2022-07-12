# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:08:06 2016

@author: Gareth
"""

# Based on https://www.kaggle.com/jeffd23/predicting-red-hat-business-value/single-unified-table-0-94-sklearn
# But rewritten for Python-practice
# Trains RandomForrestClassifier and ExtraTreesClassifier with default paramemters
# RFC: ~ 0.949
# Ext: ~ 0.947
# ExT + RFC: ~ 0.954 (magic!)

## Import functions
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


## Import data
actTrain = pd.read_csv('../input/act_train.csv')
actTest = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')

IDTest = actTest['activity_id']
IDTrain = actTrain['activity_id']
IDPeople = actTrain['people_id']


## Def functions
def strSplit(string,splitChar,idx):
    # Split strings on character, return requested index
    string = string.split(splitChar)
    return string[idx]


def numericPeople(data):
    # Make people numeric
    data['people_id'] = data['people_id'].apply(strSplit, splitChar='_', idx=1)
    data['people_id'] = pd.to_numeric(data['people_id'])    
    return data


def ppActs(data):

    # Drop outcome and return in sperate vector
    if 'outcome' in data.columns:    
        outcome = data['outcome']
        data = data.drop('outcome', axis=1)
    else:
        outcome = 0
    # Drop activity ID
    data = data.drop(['date', 'activity_id'], axis=1)
        
    # Make people numeric
    data = numericPeople(data)
    
    # Convert rest to numeric
    for c in data.columns:
        data[c] = data[c].fillna('type 0')
        if type(data[c][1]) == str:        
           data[c] = data[c].apply(strSplit, splitChar=' ', idx=1)

    return data, outcome


def ppPeople(data):
    
    # Drop date    
    data = data.drop('date', axis=1)
    
    # Make people numeric
    data = numericPeople(data)      
    
    for c in data.columns:
        if type(data[c][1]) == np.bool_:
            data[c] = pd.to_numeric(data[c]).astype(int)
        elif type(data[c][1]) == str:
            data[c] = data[c].apply(strSplit, splitChar=' ', idx=1)
    
    return data
    

def skSplit(XTrain, YTrain, propTest, randState):
        # Split training-test set
    xtr, xte, ytr, yte = train_test_split(XTrain, YTrain, test_size = propTest,
                                      random_state = randState)

    return xtr, xte, ytr, yte
    
    
def fitSKRFC(XTrain, XTest, propTest, randState):
    # Fit and score RandomForestClassifier from SK Learn
    xtr, xte, ytr, yte = skSplit(XTrain, YTrain, propTest, randState)

    # Fitting
    fit = RandomForestClassifier()
    fit.fit(xtr, ytr)

    # Predict
    xtePred = fit.predict_proba(xte)
    auc = roc_auc_score(xtePred[:,1].round(), yte)
    print(auc)
    
    return fit


def predMod(fit, XTest):
    # Predict test set
    YTest = fit.predict_proba(XTest)
    yPred = YTest[:,1]
    
    return yPred


def writeSub(IDTest, yPred, fn):
    submission = pd.DataFrame({'activity_id': IDTest, 'outcome': yPred })
    submission.head()
    submission.to_csv(fn, index=False)
    
    print('Saved as', fn)
    
    
def fitSKExT(XTrain, YTrain, propTest, randState=123):  
    # Fit and score ExtraTreesClassifier from SK Learn
    # Split
    xtr, xte, ytr, yte = skSplit(XTrain, YTrain, propTest, randState)
    
    # Fit
    fit = ExtraTreesClassifier()
    fit.fit(xtr, ytr)
    # Score    
    xtePred = fit.predict_proba(xte)
    auc = roc_auc_score(xtePred[:,1].round(), yte)
    print(auc)
    
    return fit
    
    
## Preprocess 
XTrain, YTrain = ppActs(actTrain)
XTest, asd = ppActs(actTest)
proPeople = ppPeople(people)    

# Merge in people
XTrain = XTrain.merge(proPeople, how='left', on='people_id')
XTest = XTest.merge(proPeople, how='left', on='people_id')


## Do fitting
randState = 124
propTest = 0.25
# Fitting - RFC
fn = 'SubSKRFC.csv'
fitRFC = fitSKRFC(XTrain, XTest, propTest, randState)
yPredRFC = predMod(fitRFC, XTest)
# writeSub(IDTest, yPredRFC, fn)
    
# Fitting - extraTrees
fn = 'SubSKExT.csv'
fitExT = fitSKExT(XTrain, YTrain, propTest, randState)
yPredExT = predMod(fitExT, XTest)
# writeSub(IDTest, yPredExT, fn)

# Compare preidctions
sum(yPredExT.round() == yPredRFC.round()) / len(yPredExT)
bodged = (yPredExT+yPredRFC)/2
fn = 'bodgensamle.csv'

# Save
writeSub(IDTest, bodged, fn)
