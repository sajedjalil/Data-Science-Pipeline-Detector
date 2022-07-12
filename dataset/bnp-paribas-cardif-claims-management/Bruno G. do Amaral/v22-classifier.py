# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.metrics import log_loss

train = pd.read_csv('../input/train.csv', usecols=['ID', 'target', 'v22'])
test = pd.read_csv('../input/test.csv', usecols=['ID', 'v22'])

c = 'v22'
target_column = 'target'
col_with_target = train[[c, target_column]].copy()

# Keep only those categories that are both on train and test
ctrain = frozenset(train[c])
ctest = frozenset(test[c])
cboth = ctrain.intersection(ctest)
train.loc[~train[c].isin(cboth), c] = np.nan
test.loc[~test[c].isin(cboth), c] = np.nan

target_values = col_with_target[target_column].unique()
for t in target_values:
    col_with_target[c + '_targets_' + str(t)] = 1.0 * (col_with_target[target_column] == t)
col_with_target.drop([target_column], inplace=True, axis=1)

# Sum-up target values for each category and normalize
targets_sum = col_with_target.groupby(c).sum().apply(lambda r: r / r.sum(), axis=1, raw=True)

train = pd.merge(train, targets_sum, left_on=c, right_index=True, how='left')
test = pd.merge(test, targets_sum, left_on=c, right_index=True, how='left')

train_preds = train['v22_targets_1'].fillna(train['v22_targets_1'].mean())

train_score = log_loss(train[target_column], train_preds)

print("Score on train is %.5f" % train_score)

test_preds = test['v22_targets_1'].fillna(test['v22_targets_1'].mean())
ids = test['ID'].values

predictions_file = open("result.csv", "w", 2048)
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(ids, test_preds))
predictions_file.close()