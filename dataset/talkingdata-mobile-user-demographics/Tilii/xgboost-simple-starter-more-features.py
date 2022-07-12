__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

# Added two new features in the read_train_test section
# My thanks to ZFTurbo for always providing a good starting point with his scripts

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from operator import itemgetter
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    
    # Tweaked these values a bit
    eta = 0.025
    max_depth = 7
    subsample = 0.75
    colsample_bytree = 0.75
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500
    early_stopping_rounds = 20
    test_size = 0.2

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid.tolist(), check)

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def read_train_test():
    # App events
    print('Read app events...')
    ape = pd.read_csv("../input/app_events.csv")
    ape['installed'] = ape.groupby(['event_id'])['is_installed'].transform('sum')
    ape['active'] = ape.groupby(['event_id'])['is_active'].transform('sum')
    ape.drop(['is_installed', 'is_active'], axis=1, inplace=True)
    ape.drop_duplicates('event_id', keep='first', inplace=True)
    ape.drop(['app_id'], axis=1, inplace=True)
    
    # Events
    print('Read events...')
    events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    
    # The idea here is to count the number of installed apps using the data
    # from app_events.csv above. Also to count the number of active apps.
    events = pd.merge(events, ape, how='left', on='event_id', left_index=True)

    # Below is the original events_small table
    # events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
    # And this is the new events_small table with two extra features
    events_small = events[['device_id', 'counts', 'installed', 'active']].drop_duplicates('device_id', keep='first')

    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("../input/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Train
    print('Read train...')
    train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)

    # Test
    print('Read test...')
    test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')

    return train, test, features


train, test, features = read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_xgb(train, test, features, 'group')
print("LS: {}".format(round(score, 5)))
create_submission(score, test, test_prediction)
