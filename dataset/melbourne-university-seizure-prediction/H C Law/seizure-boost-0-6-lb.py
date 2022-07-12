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
import numba
from math import *
import random
import os
import time
import glob
import re
from multiprocessing import Process
import copy

random.seed(2015)
np.random.seed(2015)
fftlen = 2048
fftlenhalf = floor(fftlen / 2)
fftfreq = np.fft.fftfreq(fftlen, 1/80.0)
samp_per_batch = 60*80 
lvl = np.array([0.1, 2, 4, 8, 16, 20])
groups = np.digitize(fftfreq[:fftlenhalf], lvl)
numbins = np.unique(groups).size - 1   # 1...numbins

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
    
@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

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
    table = pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])
    table = table[(table.index%5 == 0)]
    table = table.loc[(table!=0).any(axis=1)]
    #cols = table.columns
    #table = table.loc[(table[cols].shift() != table[cols]).any(axis=1)]
    return table, sequence

def labels_out(out):
    numcol = 16
    for i in range(numcol):
        #out.write(",avg_" + str(i))
        out.write(",skw_" + str(i))
        out.write(",kur_" + str(i))
        for j in range(1, numbins):
            out.write(",freq"+str(j)+"_"+str(i))
        #for j in range(1, numbins-1):
        #    out.write(",freqcum"+str(j)+"_"+str(i))
        out.write(",pow_" + str(i))
    for i in range(numcol):
        for j in range(i+1, numcol):
            out.write(",corr_"+str(i)+"_"+str(j))
            
def features_out(tables):
    out_str = "";
    '''
    for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out_str += "," + str(mean)
            std = tables[f].std()
            out_str += "," + str(std)
    '''        
    cols = tables.columns
    #mean = tables.mean(axis=0)
    skw = tables.skew(axis=0)
    kur = tables.kurtosis(axis=0)
    colnames = sorted(list(tables.columns.values))
    fft_corr_tab = dict();
    for f in colnames:
        #out_str += "," + str(mean[f])
        out_str += "," + str(skw[f])
        out_str += "," + str(kur[f])
        #fftout = np.absolute(np.fft.fft(tables[f][:fftlen],n=fftlenhalf))
        fftout = abs2(np.fft.fft(tables[f][:fftlen],n=fftlenhalf))
        fft_corr_tab[f] = fftout
        freqdist = np.zeros(numbins-1)
        freqcum = np.zeros(numbins-1)
        for j in range (1, numbins):
            freqgroup = fftout[groups==j]
            mfreq = sqrt(freqgroup.mean())
            freqdist[j-1] = mfreq
            freqcum[j-1] = sqrt(freqgroup.sum()) 
        freqsum = freqdist.sum()
        freqdist /= freqsum
        freqcum = freqcum.cumsum()
        power = freqcum[-1]
        #freqcum /= power
        for j in range(1, numbins):
            out_str += "," + str(freqdist[j-1])
        #for j in range(1, numbins-1):
        #    out_str += "," + str(freqcum[j-1])
        out_str += "," + str(power)
    
    fft_corr_tab_frame = pd.DataFrame.from_dict(fft_corr_tab)
    fft_corr = fft_corr_tab_frame.corr('pearson')
    # Corr matrix
    numcol = 16
    for i in range(numcol):
        for j in range(i+1, numcol):
            out_str += "," + str(fft_corr[colnames[i]][colnames[j]])
    return out_str        

def create_simple_csv_train(patient_id):

    out = open("simple_train_" + str(patient_id) + ".csv", "w")
    out.write("Id,sequence_id,patient_id")
    labels_out(out)
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
        id = int(arr[1])*10
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        
        num_subbatch = tables.shape[0] // samp_per_batch
        if (num_subbatch > 0):
            for iter in range(0, num_subbatch, 2):
                out_str += str(new_id+iter) + "," + str(sequence_id) + "," + str(patient)
                out_str += features_out(tables[samp_per_batch*iter:samp_per_batch*(iter+1)])
                out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        '''
        else:
            out_str += str(new_id) + "," + str(sequence_id) + str(patient)
            out_str += features_out(tables)
            out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        '''    
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
        id = int(arr[1])*10
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        
        num_subbatch = tables.shape[0] // samp_per_batch
        if (num_subbatch > 0):
            for iter in range(num_subbatch):
                out_str += str(new_id+iter) + "," + str(sequence_id) + "," + str(patient)
                out_str += features_out(tables[samp_per_batch*iter:samp_per_batch*(iter+1)])
                out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        '''
        else:
            out_str += str(new_id) + "," + str(sequence_id) + str(patient)
            out_str += features_out(tables)
            out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        '''    
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
    labels_out(out)
    out.write(",file_size\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[1])
        id = int(arr[2])*10
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        
        num_subbatch = tables.shape[0] // samp_per_batch
        if (num_subbatch > 0):
            for iter in range(num_subbatch):
                out_str += str(new_id+iter) + "," + str(patient)
                out_str += features_out(tables[samp_per_batch*iter:samp_per_batch*(iter+1)])
                out_str += "," + str(os.path.getsize(fl)) +  "\n"
        else:
            out_str += str(new_id) + "," + str(patient)
            out_str += features_out(tables)
            out_str += "," + str(os.path.getsize(fl)) +  "\n"

    out.write(out_str)
    out.close()
    print('Test CSV for patient {} has been completed...'.format(patient_id))


def run_single(train, test, features, target, random_state=1):
    eta = 0.2
    max_depth = 4
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


def run_kfold(nfolds, train, test, features, target, random_state=2015):
    eta = 0.2
    max_depth = 3
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
        "lambda": 1.5,
        "alpha": 1.5,
        "seed": random_state,
    }
    num_boost_round = 1000
    early_stopping_rounds = 100

    train_res = []
    yfull_test = copy.deepcopy(test[['Id']].astype(object))

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    
    X_test_id = test[['Id']]
    X_test_id['Id'] = (X_test_id['Id'] // 10).astype(int)
    
    for train_seq_index, test_seq_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        train_seq = unique_sequences[train_seq_index]
        valid_seq = unique_sequences[test_seq_index]
        print('Length of train people: {}'.format(len(train_seq)))
        print('Length of valid people: {}'.format(len(valid_seq)))

        X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
        y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
        
        X_valid_id = train[train['sequence_id'].isin(valid_seq)][['Id']]
        X_valid_id ['Id'] = (X_valid_id ['Id'] // 10 + y_valid * 50000).astype(int)
        
        X_test = test[features]

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=1000)

        print("Validating...")
        check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        
        unique_id = X_valid_id['Id'].unique()
        y_valid_list=[]
        y_hat_mean=[]

        for id in unique_id:
            yval = y_valid[(X_valid_id['Id'] == id)].values[0]
            y_valid_list.append(yval)
            check = pd.Series(check)
            checkIdx = X_valid_id.reset_index()
            ypredvec = check[(checkIdx['Id'] == id)]
            ymean = ypredvec.mean()
            y_hat_mean.append(ymean)
            train_res.append({'id':int(id % 50000), 'result': int(id // 50000), 'mean': ymean})

        score = roc_auc_score(y_valid_list, y_hat_mean)
        print('Check error value: {:.6f} '.format(score))
        
        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test set...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds+1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)
    
    yfull_test.loc[test['skw_0'].isnull(), 'mean'] = 0.0
    
    unique_test_id = X_test_id['Id'].unique()
    y_test_mean = []
    for id in unique_test_id:
        test_predvec = yfull_test[(X_test_id['Id'] == id)]['mean']
        ymean = test_predvec.mean()
        y_test_mean.append({'Id': id, 'mean': ymean})
        
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return pd.DataFrame(y_test_mean), pd.DataFrame(train_res)


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

def create_submission_merged(score, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')
    for _, row in prediction.iterrows():
        id = int(row['Id'])
        patient = id // 10000
        fid = id % 10000
        str1 = 'new_' + str(patient) + '_' + str(fid) + '.mat' + ',' + str(row['mean'])
        str1 += '\n'
        f.write(str1)
    f.close()

def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('Id')
    output.remove('file_size')
    return sorted(output)


def read_test_train():
    print("Load train.csv...")
    train1 = pd.read_csv("simple_train_1.csv")
    train2 = pd.read_csv("simple_train_2.csv")
    train3 = pd.read_csv("simple_train_3.csv")
    train = pd.concat([train1, train2, train3])
    # Remove all zeroes files / rows with na
    train = train[(train['file_size'] > 55000) & (train['skw_0'].notnull())].copy()
    # Shuffle rows since they are ordered
    # train = train.iloc[np.random.permutation(len(train))]
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
        
    train_whole, test_whole, features = read_test_train()
    test_prediction, train_pred = run_kfold(3, train_whole, test_whole, features, 'result')
  
    score = roc_auc_score(train_pred['result'], train_pred['mean'])
    print ("Training score=", score)
    create_submission_merged (score, test_prediction)

