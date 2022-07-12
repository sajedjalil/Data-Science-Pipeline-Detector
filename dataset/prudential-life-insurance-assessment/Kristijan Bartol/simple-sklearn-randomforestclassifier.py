# -*- coding: utf-8 -*-
"""

@author: Z-bra

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../input/train.csv')

def do_treatment(df):
    for col in df:
        if df[col].dtype == np.dtype('O'):
            df[col] = df[col].apply(lambda x : hash(str(x)))
            
    df.fillna(-1, inplace = True)

do_treatment(train)
        
target = 'Response'

features = list(train.columns.difference([target]))

clf = RandomForestClassifier(n_estimators = 200, max_features = 'sqrt',
                             max_depth = None, verbose = 1, n_jobs = -1)
                             
clf.fit(train[features], train[target])

test = pd.read_csv('../input/test.csv')

do_treatment(test)

preds = clf.predict(test[features])

sub = pd.read_csv('../input/sample_submission.csv')

sub[target] = preds

sub.to_csv('sub2.csv', index = False)