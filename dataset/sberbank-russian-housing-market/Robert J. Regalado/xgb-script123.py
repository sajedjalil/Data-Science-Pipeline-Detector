# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import math
#from ml_metrics import rmsle

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


def rmsle(preds, dtrain):
	labels = dtrain.get_label()
	assert len(preds) == len(labels)
	labels = labels.tolist()
	preds = preds.tolist()
	terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0, preds[i]) + 1)) ** 2.0 for i, pred in enumerate(labels)]
	return 'rmsle', (sum(terms_to_sum) * (1.0 / len(preds))) ** 0.5


# We take all float/int columns except for ID, timestamp, and the target value
train_columns = list(
	set(df_train.select_dtypes(include=['float64', 'int64']).columns) - set(['id', 'timestamp', 'price_doc']))

y_train = df_train['price_doc'].values
x_train = df_train[train_columns].values
x_test = df_test[train_columns].values

# Train/Valid split
split = 25000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 800, watchlist, feval=rmsle, early_stopping_rounds=100)

p_test = clf.predict(xgb.DMatrix(x_test))

sub = pd.DataFrame()
sub['id'] = df_test['id'].values
sub['price_doc'] = p_test
sub.to_csv('xgb.csv', index=False)