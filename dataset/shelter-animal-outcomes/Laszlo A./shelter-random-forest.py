# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def age_to_days(d):
    age = 0
    
    if type(d) is str:
        num = int(d.split(' ')[0])
        
        if 'day' in d:
            age = num
        if 'week' in d:
            age = num * 7
        if 'month' in d:
            age = num * 30
        if 'year' in d:
            age = num * 365    

    return age

def add_features(d):
    d["SexuponOutcome"].fillna('', inplace=True)
    
    d["HasName"] = d["Name"].map(lambda x : '0' if x != x else '1')
    d["TypeId"] = d["AnimalType"].map(lambda x : '1' if "Dog" in x else '0')
    d["IsMix"] = d["Breed"].map(lambda x : '1' if "Mix" in x else '0')
    d["IsIntact"] = d["SexuponOutcome"].map(lambda x : '1' if "Intact" in x else '0')
    d["Year"] = d["DateTime"].map(lambda x: x[0:4])
    d["Month"] = d["DateTime"].map(lambda x: x[5:7])
    d["Day"] = d["DateTime"].map(lambda x: x[8:10])
    d["Hour"] = d["DateTime"].map(lambda x: x[11:13])
    
    d["Age"] = d["AgeuponOutcome"].map(lambda x: age_to_days(x))
    d["SColor"] = d["Color"].map(lambda x: re.split("( |/)", x)[0])
    
    #d = pd.get_dummies(d, columns=["SColor"])
    
    return d

train = add_features(train)
train_outcome = train["OutcomeType"]
#train.drop(["Name", "OutcomeSubtype", "AnimalID", "DateTime", "OutcomeType"], axis=1, inplace=True)
#train = pd.get_dummies(train, columns=train.columns)

test = add_features(test)
#test.drop(["Name", "DateTime"], axis=1, inplace=True)
#test = pd.get_dummies(test, columns=test.columns)

#print (train)

alg = RandomForestClassifier(n_estimators=500, n_jobs=2)

cols = ["Age", "IsIntact", "Hour", "Month", "HasName", "TypeId", "IsMix"]
scores = model_selection.cross_val_score(alg, train[cols], train_outcome, cv=3)
print(scores.mean())

#alg.fit(train[cols], train["OutcomeType"])
#pred = alg.predict_proba(test[cols])

#results = pd.read_csv("../input/sample_submission.csv")
#results['Adoption'], results['Died'], results['Euthanasia'], results['Return_to_owner'], results['Transfer'] \
#= pred[:,0], pred[:,1], pred[:,2], pred[:,3], pred[:,4]
#results.to_csv("submission.csv", index=False)