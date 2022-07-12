__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
from operator import itemgetter
import time
import copy

random.seed(1712)


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


def intersect(a, b):
    return list(set(a) & set(b))


def run_single(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 150
    early_stopping_rounds = 10
    test_size = 0.1

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
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def run_kfold(nfolds, train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round =100
    early_stopping_rounds = 10

    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['activity_id']].astype(object))
    kf = KFold(len(train.index), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, test_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
        y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]
        X_test = test[features].as_matrix()

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        
        print("Validating...")
        yhat = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.tolist(), yhat)
        print('Check error value: {:.6f}'.format(score))

        # Each time store portion of precicted data in train predicted values
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = yhat[i]

        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test set...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # Copy dict to list
    train_res = []
    for i in sorted(yfull_train.keys()):
        train_res.append(yfull_train[i])

    score = roc_auc_score(train[target], np.array(train_res))
    print('Check error value: {:.6f}'.format(score))

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds+1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return yfull_test['mean'].values, score


def create_submission(score, test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    for id in test['activity_id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    return sorted(output)


def read_test_train():

    print("Read people.csv...")
    people = pd.read_csv("../input/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("../input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv("../input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Process tables...")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop('date', axis=1, inplace=True)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop('date', axis=1, inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)

    features = get_features(train, test)
    return train, test, features


train, test, features = read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))

test_prediction, score = run_single(train, test, features, 'outcome')
# test_prediction, score = run_kfold(3, train, test, features, 'outcome')
create_submission(score, test, test_prediction)
