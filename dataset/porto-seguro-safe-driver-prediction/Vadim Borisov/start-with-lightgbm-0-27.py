# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Read in our input data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


y_train = df_train['target'].values
id_train = df_train['id'].values
id_test = df_test['id'].values

# We drop these variables as we don't want to train on them
# The other 57 columns are all numerical and can be trained on without preprocessing
x_train = df_train.drop(['target', 'id'], axis=1)
x_test = df_test.drop(['id'], axis=1)




print("Train")
params = {
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'learning_rate': 0.03,
        'metric': 'binary_logloss',
        'metric': 'auc',
        'min_data_in_bin': 3,
        'max_depth': 10,
        'objective': 'binary',
        'verbose': -1,
        'num_leaves': 108,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 223,
        'num_rounds': 1000,
         }
clf = lgb.LGBMClassifier(**params, n_estimators = 233)
clf.fit(x_train, y_train)


print("Predict")
y_pred = clf.predict_proba(x_test)[:,1]



# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('submit.csv', index=False, float_format='%.2f') 