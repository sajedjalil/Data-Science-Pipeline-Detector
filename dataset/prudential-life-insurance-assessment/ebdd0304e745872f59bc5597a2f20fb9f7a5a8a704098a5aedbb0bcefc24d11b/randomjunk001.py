# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:30:40 2015

@author: Z
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from ml_metrics import quadratic_weighted_kappa
import xgboost as xgb

train = pd.read_csv('../input/train.csv')

def do_treatment(df):
    for col in df:
        if df[col].dtype == np.dtype('O'):
            df[col] = df[col].apply(lambda x : hash(str(x)))
            
    df.fillna(-1, inplace = True)

do_treatment(train)
        
target = 'Response'

non_med_keywords = train.columns.difference([target])
                                             
samples = train.sample(10000 , random_state = 0)

params = {'objective' : 'reg:linear',
          'max_depth' : 8,
          'seed' : 0,
          'subsample' : 0.8,
          'colsample_bytree' : 1,
          'colsample_bylevel' : 0.7,
          'gamma' : 0,
          'silent' : 1,
          'n_estimators' : 300,
          'eta' : 0.05}
            
DMatrix_samples = xgb.DMatrix(samples[non_med_keywords].as_matrix(), label = samples[target])

DMatrix_train = xgb.DMatrix(train[non_med_keywords].as_matrix(), label = train[target].as_matrix())
    
clf = xgb.train(params, DMatrix_train,
                num_boost_round = params['n_estimators'],
                evals= [(DMatrix_samples, 'samples')],
                early_stopping_rounds = 50)
    
mainclf = LinearRegression()

clfpreds = pd.DataFrame(clf.predict(DMatrix_train), index = train.index)

mainclf.fit(clfpreds.as_matrix(), train[target])

endclf = IsotonicRegression()

mainclfpreds = mainclf.predict(clfpreds.as_matrix())

endclf.fit(mainclfpreds, train[target])

clfpreds = pd.DataFrame(clf.predict(DMatrix_samples), index = samples.index)

preds2 = mainclf.predict(clfpreds.as_matrix())

preds = endclf.predict(preds2)

preds[np.isnan(preds)] = preds2[np.isnan(preds)]

preds = np.round(preds)
preds[preds < 1] = 1
preds[preds > 8] = 8
preds = np.array([int(x) for x in preds])

print(quadratic_weighted_kappa(samples[target], preds))
    
test = pd.read_csv('../input/test.csv')

do_treatment(test)

DMatrix_test = xgb.DMatrix(test[non_med_keywords].as_matrix(), label = None)

clfpreds = pd.DataFrame(clf.predict(DMatrix_test), index = test.index)

preds2 = mainclf.predict(clfpreds.as_matrix())

preds = endclf.predict(preds2)

preds[np.isnan(preds)] = preds2[np.isnan(preds)]

preds = np.round(preds)
preds[preds < 1] = 1
preds[preds > 8] = 8
preds = np.array([int(x) for x in preds])

sub = pd.read_csv('../input/sample_submission.csv')

sub[target] = preds

sub.to_csv('sub49.csv', index = False)