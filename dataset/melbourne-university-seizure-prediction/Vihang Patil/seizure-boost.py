# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from operator import itemgetter
import random
import os
import time
import glob
import re
from multiprocessing import Process
import copy

random.seed(2016)
np.random.seed(2016)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


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


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]), sequence


def create_simple_csv_train(patient_id):

    out = open("simple_train_" + str(patient_id) + ".csv", "w")
    out.write("Id,sequence_id,patient_id")
    for i in range(16):
        out.write(",avg_" + str(i))
    out.write(",file_size,result\n")

    # TRAIN (0)
    out_str = ''
    files = sorted(glob.glob("../input/train_" + str(patient_id) + "/*0.mat"), key=natural_key)
    sequence_id = 0
    total = 0
    for fl in files:
        total += 1
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(sequence_id) + "," + str(patient)
        for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out_str += "," + str(mean)
        out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        if total % 6 == 0:
            if int(sequence_from_mat) != 6:
                print('Check error! {}'.format(sequence_from_mat))
                exit()
            sequence_id += 1

    out.write(out_str)

    # TRAIN (1)
    out_str = ''
    files = sorted(glob.glob("../input/train_" + str(patient_id) + "/*1.mat"), key=natural_key)
    sequence_id += 1
    total = 0
    for fl in files:
        total += 1
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(sequence_id) + "," + str(patient)
        for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out_str += "," + str(mean)
        out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        if total % 6 == 0:
            if int(sequence_from_mat) != 6:
                print('Check error! {}'.format(sequence_from_mat))
                exit()
            sequence_id += 1

    out.write(out_str)
    out.close()
    print('Train CSV for patient {} has been completed...'.format(patient_id))


def create_simple_csv_test(patient_id):

    # TEST
    out_str = ''
    files = sorted(glob.glob("../input/test_" + str(patient_id) + "_new/*.mat"), key=natural_key)
    out = open("simple_test_" + str(patient_id) + ".csv", "w")
    out.write("Id,patient_id")
    for i in range(16):
        out.write(",avg_" + str(i))
    out.write(",file_size\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[1])
        id = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(patient)
        for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out_str += "," + str(mean)
        out_str += "," + str(os.path.getsize(fl)) + "\n"
        # break

    out.write(out_str)
    out.close()
    print('Test CSV for patient {} has been completed...'.format(patient_id))


def run_single(train, test, features, target, random_state=1):
    eta = 0.2
    max_depth = 3
    subsample = 0.9
    colsample_bytree = 0.9
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
    num_boost_round = 1000
    early_stopping_rounds = 50
    test_size = 0.2

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=int(round(1/test_size, 0)), shuffle=True, random_state=random_state)
    train_seq_index, test_seq_index = list(kf)[0]
    print('Length of sequence train: {}'.format(len(train_seq_index)))
    print('Length of sequence valid: {}'.format(len(test_seq_index)))
    train_seq = unique_sequences[train_seq_index]
    valid_seq = unique_sequences[test_seq_index]

    X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    X_test = test[features]

    print('Length train:', len(X_train))
    print('Length valid:', len(X_valid))

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def run_kfold(nfolds, train, test, features, target, random_state=2016):
    eta = 0.2
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7
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
    num_boost_round = 1000
    early_stopping_rounds = 50

    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['Id']].astype(object))

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_seq_index, test_seq_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        train_seq = unique_sequences[train_seq_index]
        valid_seq = unique_sequences[test_seq_index]
        print('Length of train people: {}'.format(len(train_seq)))
        print('Length of valid people: {}'.format(len(valid_seq)))

        X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
        y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
        X_test = test[features]

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=1000)

        yhat = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)

        # Each time store portion of precicted data in train predicted values
        for i in range(len(X_valid.index)):
            yfull_train[X_valid.index[i]] = yhat[i]

        print("Validating...")
        check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.tolist(), check)
        print('Check error value: {:.6f}'.format(score))

        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test set...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # Copy dict to list
    train_res = []
    for i in range(len(train.index)):
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
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')
    total = 0
    for id in test['Id']:
        patient = id // 100000
        fid = id % 100000
        str1 = 'new_' + str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('Id')
    # output.remove('file_size')
    return sorted(output)


def read_test_train():
    print("Load train.csv...")
    train1 = pd.read_csv("simple_train_1.csv")
    train2 = pd.read_csv("simple_train_2.csv")
    train3 = pd.read_csv("simple_train_3.csv")
    train = pd.concat([train1, train2, train3])
    # Remove all zeroes files
    train = train[train['file_size'] > 55000].copy()
    # Shuffle rows since they are ordered
    train = train.iloc[np.random.permutation(len(train))]
    # Reset broken index
    train = train.reset_index()
    print("Load test.csv...")
    test1 = pd.read_csv("simple_test_1.csv")
    test2 = pd.read_csv("simple_test_2.csv")
    test3 = pd.read_csv("simple_test_3.csv")
    test = pd.concat([test1, test2, test3])
    print("Process tables...")
    features = get_features(train, test)
    return train, test, features


if __name__ == '__main__':
    print('XGBoost: {}'.format(xgb.__version__))
    if 1:
        # Do reading and processing of MAT files in parallel
        p = dict()
        p[1] = Process(target=create_simple_csv_train, args=(1,))
        p[1].start()
        p[2] = Process(target=create_simple_csv_train, args=(2,))
        p[2].start()
        p[3] = Process(target=create_simple_csv_train, args=(3,))
        p[3].start()
        p[4] = Process(target=create_simple_csv_test, args=(1,))
        p[4].start()
        p[5] = Process(target=create_simple_csv_test, args=(2,))
        p[5].start()
        p[6] = Process(target=create_simple_csv_test, args=(3,))
        p[6].start()
        p[1].join()
        p[2].join()
        p[3].join()
        p[4].join()
        p[5].join()
        p[6].join()
    train, test, features = read_test_train()
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    # test_prediction, score = run_single(train, test, features, 'result')
    test_prediction, score = run_kfold(5, train, test, features, 'result')
    create_submission(score, test, test_prediction)
