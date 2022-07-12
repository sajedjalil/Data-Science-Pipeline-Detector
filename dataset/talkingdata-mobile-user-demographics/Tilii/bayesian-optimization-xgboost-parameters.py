from __future__ import print_function
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
__author__ = 'tilii: https://kaggle.com/tilii7'

# ZFTurbo defined first 3 features
# tilii added two new features and Bayesian Optimization
# Bayesian Optimization library credit to Fernando Nogueira https://www.kaggle.com/fnogueira
# Also see https://github.com/fmfn/BayesianOptimization
# Some ideas were taken from Mike Pearmain https://www.kaggle.com/mpearmain
# Also see https://github.com/mpearmain/BayesBoost

import pandas as pd
import numpy as np
import re
from sklearn.cross_validation import train_test_split
from sklearn import manifold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bayes_opt import BayesianOptimization
import xgboost as xgb

# Instead of running XGBXlassifier, we do xgbCV, so this takes longer
# We capture stderr and stdout using the function below

import contextlib


@contextlib.contextmanager
def capture():
    import sys
    from cStringIO import StringIO
    olderr, oldout = sys.stderr, sys.stdout
    try:
        out = [StringIO(), StringIO()]
        sys.stderr, sys.stdout = out
        yield out
    finally:
        sys.stderr, sys.stdout = olderr, oldout
        out[0] = out[0].getvalue().splitlines()
        out[1] = out[1].getvalue().splitlines()

# Define which xgbCV parameters are used for grid search
# and specify all xgbCV parameters


def XGBcv(max_depth, gamma, min_child_weight, max_delta_step, subsample,
          colsample_bytree):
    paramt = {
        'gamma': gamma,
        'booster': 'gbtree',
        'max_depth': max_depth.astype(int),
        'eta': 0.1,
        # Use the line below for classification
        #'objective' : 'binary:logistic',
        'objective': 'multi:softprob',
        #'nthread' : 8,
        # DO NOT use the line below when doing classification
        'num_class': 12,
        'silent': True,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'max_delta_step': max_delta_step.astype(int),
        'seed': 101
    }

    # Use 10-fold validation if you have time to spare
    #folds = 10
    folds = 5
    cv_score = 0

    print(" Search parameters (%d-fold validation):\n %s" % (folds, paramt),
          file=log_file)
    log_file.flush()

    # Do not optimize the number of boosting rounds, as early stopping will take care of that

    with capture() as result:
        xgb.cv(paramt,
               dtrain,
               num_boost_round=20000,
               stratified=True,
               nfold=folds,
               verbose_eval=1,
               early_stopping_rounds=50,
               metrics="mlogloss",
               show_stdv=True)

# All relevant things in XGboost output are in stdout, so we screen result[1]
# for a line with "cv-mean". This line signifies the end of output and contains CV values.
# Next we split the line to extract CV values. We also print the whole CV run into file
# In previous XGboost the output was in stderr, in which case we would need result[0]

    print('', file=log_file)
#    for line in result[0]:
    for line in result[1]:
        print(line, file=log_file)
        if str(line).find('cv-mean') != -1:
            cv_score = float(re.split('[|]| |\t|:', line)[2])
    log_file.flush()

    # The CV metrics function in XGboost can be lots of things. Some of them need to be maximized, like AUC.
    # If the metrics needs to be minimized, e.g, logloss, the return line below should be a negative number
    # as Bayesian Optimizer only knows how to maximize the function

    return (-1.0 * cv_score)
#    return cv_score


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table


def read_train_test():
    # App events
    print('\nReading app events...')
    ape = pd.read_csv('../input/app_events.csv')
    ape.drop_duplicates('event_id', keep='first', inplace=True)
    ape.drop(['app_id'], axis=1)

    # Events
    print('Reading events...')
    events = pd.read_csv('../input/events.csv', dtype={'device_id': np.str})
    events['counts'] = events.groupby(
        ['device_id'])['event_id'].transform('count')

    print('Making events features...')
    # The idea here is to count the number of installed apps using the data
    # from app_events.csv above. Also to count the number of active apps.
    events = pd.merge(events, ape, how='left', on='event_id', left_index=True)
    events['installed'] = events.groupby(
        ['device_id'])['is_installed'].transform('sum')
    events['active'] = events.groupby(
        ['device_id'])['is_active'].transform('sum')

    # Below is the original events_small table
    # events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
    # And this is the new events_small table with two extra features
    events_small = events[['device_id', 'counts', 'installed',
                           'active']].drop_duplicates('device_id',
                                                      keep='first')

    # Phone brand
    print('Reading phone brands...')
    pbd = pd.read_csv('../input/phone_brand_device_model.csv',
                      dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Train
    print('Reading train data...')
    train = pd.read_csv('../input/gender_age_train.csv',
                        dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    print('Merging features with train data...')
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train,
                     events_small,
                     how='left',
                     on='device_id',
                     left_index=True)
    train.fillna(-1, inplace=True)

    # Test
    print('Reading test data...')
    test = pd.read_csv('../input/gender_age_test.csv',
                       dtype={'device_id': np.str})
    print('Merging features with test data...\n')
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test,
                    events_small,
                    how='left',
                    on='device_id',
                    left_index=True)
    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')
    return train, test, features


train, test, features = read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}\n'.format(len(features), sorted(features)))
train_df = pd.DataFrame(data=train)
X = train_df.drop(['group', 'device_id'], axis=1).values
Y = train_df['group'].values
dtrain = xgb.DMatrix(X, label=Y)

# Create a file to store XGBoost output
# New lines are added to this file rather than overwriting it
log_file = open("XGBoost-output-from-BOpt.txt", 'a')

# Do hyperparameter search by Bayesian Optimization

# Below are real production parameters
#XGB_BOpt = BayesianOptimization(XGBcv, { 'max_depth': (4, 15),
#                                          'gamma': (0.0001, 2.0),
#                                          'min_child_weight': (1, 10),
#                                          'max_delta_step': (0, 5),
#                                          'subsample': (0.2, 1.0),
#                                          'colsample_bytree' :(0.2, 1.0)
#                                        })

# These parameters will allow for rapid grid search
XGB_BOpt = BayesianOptimization(XGBcv, {'max_depth': (4, 6),
                                        'gamma': (0.0001, 0.005),
                                        'min_child_weight': (1, 2),
                                        'max_delta_step': (0, 1),
                                        'subsample': (0.2, 0.4),
                                        'colsample_bytree': (0.2, 0.4)})

print('\n', file=log_file)
log_file.flush()

print('Running Bayesian Optimization ...\n')
XGB_BOpt.maximize(init_points=5, n_iter=5)
# Production parameters
#XG_BOpt.maximize(init_points=5, n_iter=50)

print('\nFinal Results')
print('XGBOOST: %f' % XGB_BOpt.res['max']['max_val'])
print('\nFinal Results', file=log_file)
print('XGBOOST: %f' % XGB_BOpt.res['max']['max_val'], file=log_file)
log_file.flush()
log_file.close()
