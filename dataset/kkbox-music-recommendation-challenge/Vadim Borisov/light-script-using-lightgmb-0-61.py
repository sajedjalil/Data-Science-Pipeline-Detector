# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn import preprocessing
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#from datacleaner import autoclean

# Any results you write to the current directory are saved as output.


#members_df = pd.read_csv("../input/members.csv")
#songs_df = pd.read_csv("../input/songs.csv")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


y = train_df.target
train_df = train_df.drop('target',1)

full = train_df.append(test_df)
full.drop('id', 1, inplace=True)


print(full.shape[0], '=', train_df.shape[0] + test_df.shape[0])


print("Label Encoding")
full = pd.DataFrame({col: full[col].astype('category').cat.codes for col in full}, index=full.index)


print("Make train and test data")
X_train = full.iloc[:7377418]
X_test = full.iloc[7377418:]

print("Train")
params = {
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'learning_rate': 0.01,
        'metric': 'binary_logloss',
        'metric': 'auc',
        'min_data_in_bin': 3,
        'max_depth': 9,
        'objective': 'binary',
        'verbose': 1,
        'num_leaves': 108,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 228,
        'num_rounds': 1000,
         }
clf = lgb.LGBMClassifier(**params, n_estimators = 300)
clf.fit(X_train, y)


print("Predict")
y_pred = clf.predict_proba(X_test)[:,1]

print("Writing submission file")

# Save submission
sub = pd.DataFrame()
sub['id'] = test_df['id']
sub['target'] = y_pred
sub.to_csv('sub.csv', index=False, float_format='%.2f')  

# How to improve the score?
# Play with hyperparams and add members and songs data. 