# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
from sklearn.cross_validation import KFold

def get_oof(clf, x_train, y, x_test):
    NFOLDS=20
    SEED=71
    kf = KFold(len(x_train), n_folds=NFOLDS, shuffle=True, random_state=SEED)
    oof_train = np.zeros((len(x_train),))
    oof_test = np.zeros((len(x_test),))
    oof_test_skf = np.empty((NFOLDS, len(x_test)))
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        # 'max_depth': 15,
        'num_leaves': 30,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.75,
        'bagging_freq': 4,
        'learning_rate': 0.016*1.5,
        #'max_bin':1023,
        'verbose': 0
    }
    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        y_te = y[test_index]
        x_te = x_train[test_index]
        lgtrain = lgb.Dataset(x_tr, y_tr)
                    #feature_name=x_train.columns.tolist())
        lgvalid = lgb.Dataset(x_te, y_te)
                    #feature_name=x_train.columns.tolist())
                    #categorical_feature = categorical)
        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=20000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=50,
            verbose_eval=50
        )
        oof_train[test_index] = lgb_clf.predict(x_te)
        oof_test_skf[i, :]    = lgb_clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test

train = pd.read_csv('../input/train.csv')
trainX = train[train.columns[2:]].values
trainy = train['target'].values
test = pd.read_csv('../input/test.csv')
testX = test[test.columns[1:]].values

oof_train, oof_test = get_oof(None, trainX, np.log1p(trainy), testX)
rms = sqrt(mean_squared_error(trainy, oof_train))
print('LGB OOF RMSE: {}'.format(rms))
print("Modeling Stage")
preds = np.concatenate([oof_test])

sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = np.around(np.expm1(preds), 0)
sub.to_csv('sub_et.csv', index=False)
