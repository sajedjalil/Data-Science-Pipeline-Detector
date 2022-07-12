#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:21:51 2017

@author: pegasus
"""

import pandas as pd
import numpy as np
#from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
from sklearn  import cross_validation 
import matplotlib.pyplot as plt

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train=train.drop('Id',axis=1)

ids=test['Id']
test=test.drop('Id',axis=1)

y=train['Cover_Type']
xtrain=train.drop('Cover_Type',axis=1)
"""

predictors=list(xtrain.axes[1])

selector = SelectKBest(f_classif, k=9)

selector.fit(xtrain[predictors], train["Cover_Type"])

scores = -np.log10(selector.pvalues_)

submission = pd.DataFrame({
        "Feature": xtrain.columns,
        "Value": scores
    })

jo=list(submission[submission["Value"]>60]["Feature"][1:])

print jo
"""

alg=RandomForestClassifier(max_features='sqrt')
"""
prange=np.arange(50,650,50)

train_scores, cv_scores=validation_curve(alg, xtrain, y, param_name='n_estimators', param_range=prange)

plt.xlabel('n_estimators')
plt.ylabel('mean CV score')

plt.plot(prange, np.mean(cv_scores, axis=1), label="cross-validation-score", color="b")
"""
from sklearn.preprocessing import Normalizer

size=10
def feat_eng(df):
    
    #absolute distance to water
    df['Distance_To_Hydrology']=(df['Vertical_Distance_To_Hydrology']**2.0+
                                 df['Horizontal_Distance_To_Hydrology']**2.0)**0.5
    
feat_eng(xtrain)
feat_eng(test)
#X_temp=Normalizer().fit_transform(xtrain[:,0:size])

#print cross_validation.cross_val_score(alg,xtrain,y).mean()

cols=list(xtrain.columns[:10])
muffin=list(xtrain.columns[10:-1])
cols.append('Distance_To_Hydrology')

norm=Normalizer().fit_transform(xtrain[cols])
norm_form=pd.DataFrame(norm, columns=cols)
print(norm_form.describe(),norm_form.shape)

new=pd.concat([norm_form[cols[:-1]],xtrain[muffin],norm_form[cols[-1]]],axis=1)
print(new.describe())

"""
alg.set_params(n_estimators=350)
alg.fit(xtrain,y)
ranks=list(pd.DataFrame(alg.feature_importances_,index=xtrain.columns).sort([0], ascending=False).axes[0])
alg.fit(xtrain[ranks[:38]],y)
print cross_validation.cross_val_score(alg, xtrain[ranks[:38]], y).mean()
predictions=alg.predict(test[ranks[:38]])
"""
"""
print predictions
sub=pd.DataFrame({ "Cover_Type": predictions,
                      "Id": ids
                     
                     })
#print sub
"""