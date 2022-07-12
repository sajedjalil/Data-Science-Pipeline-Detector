# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

n_folds = 5
rand_state = 115

# Load data
train_ds = pd.read_csv('../input/train.csv')
test_ds = pd.read_csv('../input/test.csv')

print('Train Dataset: {}'.format(train_ds.shape))
print('Test Dataset: {}'.format(test_ds.shape))

train_columns = [c for c in train_ds.columns if c not in ['ID_code', 'target']]

targets = train_ds['target']
sub = pd.DataFrame({'ID_code': test_ds['ID_code'].values})

sc_scaler = StandardScaler()
train_ds_scaled = pd.DataFrame(sc_scaler.fit_transform(train_ds[train_columns]), columns=train_columns)
test_ds_scaled = pd.DataFrame(sc_scaler.transform(test_ds[train_columns]), columns=train_columns)
del train_ds, test_ds

def feature_engineering(ds, features):
    for i in features:
        colname = 'p2_' + i
        ds[colname] = ds[i] ** 2
        p2 = ds[colname]

        colname = 'p3_' + i
        ds[colname] = ds[i] ** 3
        p3 = ds[colname]

        colname = 'p4_' + i
        ds[colname] = ds[i] ** 4
        p4 = ds[colname]

        colname = 'p5_' + i
        ds[colname] = ds[i] ** 5
        p5 = ds[colname]

        colname = 'g1_' + i
        ds[colname] = p2 * p3
        colname = 'g2_' + i
        ds[colname] = p2 * p4
        colname = 'g3_' + i
        ds[colname] = p2 * p5
        colname = 'g4_' + i
        ds[colname] = p3 * p4
        colname = 'g5_' + i
        ds[colname] = p3 * p5
        colname = 'g6_' + i
        ds[colname] = p4 * p5
    return ds

train_ds_scaled = feature_engineering(train_ds_scaled, train_columns)
test_ds_scaled = feature_engineering(test_ds_scaled, train_columns)

j = 1
r = np.zeros(train_ds_scaled.shape[0])
scores = []
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
for train_index, test_index in skf.split(train_ds_scaled, targets):
    x_train, x_valid = train_ds_scaled.iloc[train_index], train_ds_scaled.iloc[test_index]
    y_train, y_valid = targets[train_index], targets[test_index]
    
    clf = Ridge(alpha=1, random_state = rand_state)
    clf.fit(x_train, y_train)

    p = clf.predict(x_valid)
    p = np.exp(p) / np.sum(np.exp(p))
    score = roc_auc_score(y_valid, p)
    scores.append(score)

    print('Fold ({}) Score: {:.5f}'.format(j, score))
    
    p = clf.predict(test_ds_scaled)
    p = np.exp(p) / np.sum(np.exp(p))
    r += p
    j += 1
    
r /= n_folds

print('Total Average Score {:.5f}'.format(np.mean(scores)))
sub['target'] = r
sub.to_csv('submission.csv', index=False)
print('--->>> Done <<<---')
