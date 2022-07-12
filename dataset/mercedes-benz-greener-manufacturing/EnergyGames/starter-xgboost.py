# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

from sklearn.decomposition import PCA
pca2 = PCA(n_components=5)
pca2_results = pca2.fit_transform(train.drop(["y"], axis=1))
train['pca0']=pca2_results[:,0]
train['pca1']=pca2_results[:,1]
train['pca2']=pca2_results[:,2]
train['pca3']=pca2_results[:,3]
train['pca4']=pca2_results[:,4]
pca2_results = pca2.transform(test)
test['pca0']=pca2_results[:,0]
test['pca1']=pca2_results[:,1]
test['pca2']=pca2_results[:,2]
test['pca3']=pca2_results[:,3]
test['pca4']=pca2_results[:,4]

import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

usable_columns = list(set(train.columns) - set(['ID', 'y']))

y_train = train['y'].values
id_test = test['ID'].values
x_train = train[usable_columns]
x_test = test[usable_columns]

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(x_test)

params = {
    'eta': 0.02,
    'max_depth': 4,
    'subsample': 0.9,
    # 'colsample_bytree': 0.95,
    'objective': 'reg:linear',
}

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Uncomment to tune XGB `num_boost_rounds`

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=20, feval=xgb_r2_score, maximize=True, verbose_eval=10)

p_test = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = p_test
sub.to_csv('xgb.csv', index=False)