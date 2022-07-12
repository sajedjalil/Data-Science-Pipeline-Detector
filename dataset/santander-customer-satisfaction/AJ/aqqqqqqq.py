import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def compute_score(clf, x, y):
    xval = cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
    return np.mean(xval)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.set_index('ID', inplace=True, drop=True)
test.set_index('ID', inplace=True, drop=True)
target = train.TARGET
del train['TARGET']



rfc = RandomForestClassifier()
rfc.fit(train, target)
p = rfc.predict(test)
rfc_submit = pd.DataFrame({'ID': test.index, 'TARGET': p})
rfc_submit.to_csv('rfc_base.csv', index=False)
