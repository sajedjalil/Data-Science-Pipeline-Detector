# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from operator import itemgetter
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
fftlen = 512
fftlenhalf = floor(fftlen / 2)
fftfreq = np.fft.fftfreq(fftlen, 1/80.0)
lvl = np.array([0.1, 2, 4, 6,  8, 12, 16, 20, 28, 36])
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
    for i in range(16):
        out.write(",avg_" + str(i))
        out.write(",skw_" + str(i))
        out.write(",kur_" + str(i))
        for j in range(1, numbins+1):
            out.write(",freq"+str(j)+"_"+str(i))
        
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
    mean = tables.mean(axis=0)
    skw = tables.skew(axis=0)
    kur = tables.kurtosis(axis=0)
    for f in sorted(list(tables.columns.values)):
        out_str += "," + str(mean[f])
        out_str += "," + str(skw[f])
        out_str += "," + str(kur[f])
        fftout = np.absolute(np.fft.fft(tables[f][:fftlen],n=fftlenhalf))
        for j in range (1, numbins+1):
            mfreq = fftout[groups==j].mean()
            out_str += "," + str(mfreq)
            
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
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(sequence_id) + "," + str(patient)
        out_str += features_out(tables)
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
        out_str += features_out(tables)
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
    labels_out(out)
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
        out_str += features_out(tables)
        out_str += "," + str(os.path.getsize(fl)) + "\n"
        # break

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


def Outputs(data):
    return 1./(1.+np.exp(-data))


def GP1(data):
    p = (1.000000*np.tanh(((31.006277 + (data["skw_12"] - (data["kur_6"] - 0.474747))) * (-((np.cos(((0.474747 * data["kur_8"]) - ((data["freq9_2"] > np.sin(np.minimum( (0.474747),  (np.cos((0.474747 * (((0.474747 * data["kur_8"]) * data["kur_8"]) + (1.0/(1.0 + np.exp(- (-(9.869604)))))))))))).astype(float)))) * 2.0))))) +
         1.000000*np.tanh((-3.0 - ((2.076920 + (-(data["freq8_0"]))) + (data["kur_4"] - ((data["kur_14"] - ((np.tanh(data["freq8_0"]) > ((np.cos(data["freq8_0"]) < data["freq6_2"]).astype(float))).astype(float))) - ((np.cos(np.cos(data["freq8_0"])) > (data["kur_14"] - ((data["freq8_0"] > np.cos((np.cos(data["kur_4"]) + data["freq8_0"]))).astype(float)))).astype(float))))))) +
         1.000000*np.tanh((-(np.maximum( (((data["freq2_1"] + (data["freq3_12"] + np.maximum( ((((data["skw_2"] > 1.414214).astype(float)) * 2.0)),  (((data["patient_id"] >= 0.0).astype(float)))))) + ((data["freq2_1"] + (data["freq3_12"] + np.cos(data["freq7_8"]))) + np.maximum( ((((data["skw_2"] > 1.414214).astype(float)) * 2.0)),  (data["freq4_14"]))))),  (((data["freq1_11"] - (-((((data["freq4_14"] >= 0.0).astype(float)) * 2.0)))) / 2.0)))))) +
         1.000000*np.tanh(np.minimum( (((data["kur_15"] > (((np.minimum( (data["kur_4"]),  (np.cos(data["kur_1"]))) < data["kur_7"]).astype(float)) * 2.0)).astype(float))),  ((np.maximum( (data["kur_1"]),  (np.maximum( (data["kur_1"]),  ((-(((np.maximum( (data["freq1_0"]),  (data["freq5_5"])) > np.minimum( (data["kur_13"]),  (((np.maximum( (data["freq1_0"]),  (data["freq5_5"])) > np.minimum( (2.718282),  ((((data["kur_9"] < data["kur_7"]).astype(float)) * 2.0)))).astype(float))))).astype(float)))))))) * (2.718282 * 2.0))))) +
         1.000000*np.tanh((data["freq7_2"] * (np.minimum( (2.302585),  ((((data["kur_3"] + (data["kur_0"] * (((data["freq7_14"] > (-(np.tanh(((np.maximum( (data["freq7_7"]),  (((data["freq9_2"] > data["freq10_2"]).astype(float)))) < ((data["kur_0"] * (data["freq7_2"] / 2.0)) + data["skw_10"])).astype(float)))))).astype(float)) / 2.0))) >= 0.0).astype(float)))) * 2.0))) +
         1.000000*np.tanh(((((-((np.maximum( (data["skw_10"]),  ((1.414214 + (data["avg_15"] - np.minimum( (1.414214),  (data["skw_14"])))))) / 2.0))) - data["skw_4"]) - data["skw_10"]) * ((data["freq1_14"] > (data["freq9_4"] * -1.0)).astype(float)))) +
         1.000000*np.tanh(((np.minimum( (data["freq9_9"]),  (np.minimum( (data["freq9_13"]),  ((data["freq1_10"] * (data["freq5_3"] - np.minimum( (np.minimum( (((data["freq3_8"] < data["freq8_0"]).astype(float))),  (((data["freq3_8"] < data["freq9_13"]).astype(float))))),  (((data["freq8_0"] + (-2.0 + np.maximum( (data["freq8_9"]),  (data["freq3_8"])))) + (data["file_size"] + data["freq8_0"])))))))))) >= 0.0).astype(float))) +
         1.000000*np.tanh(np.minimum( (data["kur_11"]),  ((-((((-(((data["freq6_11"] < np.minimum( (data["freq10_6"]),  (data["freq1_9"]))).astype(float)))) * np.minimum( ((-(np.minimum( (((data["freq7_5"] < data["skw_9"]).astype(float))),  (np.minimum( (data["skw_15"]),  (data["freq1_9"]))))))),  (((data["freq1_9"] < data["freq10_6"]).astype(float))))) * 2.0)))))) +
         1.000000*np.tanh((((((data["skw_3"] + 0.636620) > ((((((data["freq7_11"] > np.cos(data["freq7_8"])).astype(float)) > np.maximum( (np.maximum( ((data["skw_12"] + data["skw_12"])),  (((np.sin((data["freq2_6"] * 2.0)) >= 0.0).astype(float))))),  ((1.0/(1.0 + np.exp(- 0.301030)))))).astype(float)) < np.cos(np.minimum( (data["freq2_6"]),  (data["freq7_11"])))).astype(float))).astype(float)) > np.maximum( ((data["skw_12"] + data["skw_12"])),  (0.301030))).astype(float))) +
         1.000000*np.tanh(((0.474747 < (((data["avg_13"] > data["freq4_15"]).astype(float)) * np.minimum( (data["kur_1"]),  (np.maximum( (np.sin(np.maximum( (data["freq10_10"]),  ((data["freq4_14"] - data["kur_1"]))))),  ((((data["freq3_1"] < (data["freq4_14"] - ((0.474747 < (-(data["avg_1"]))).astype(float)))).astype(float)) * np.minimum( (data["kur_5"]),  (data["kur_5"]))))))))).astype(float))) +
         0.964800*np.tanh((data["skw_14"] * np.cos((data["freq4_4"] - np.cos(((data["avg_1"] * np.sin((data["skw_2"] * 2.718282))) - ((1.0/(1.0 + np.exp(- (np.tanh(np.sin(data["avg_5"])) + (data["kur_9"] * (data["skw_14"] * (2.718282 * ((data["freq9_13"] < (0.434294 - np.sin(data["avg_5"]))).astype(float))))))))) - data["kur_9"]))))))) +
         1.000000*np.tanh(((((data["kur_0"] * np.minimum( (((((((math.tanh((float(0.0 >= 0.0))) / 2.0) < np.minimum( ((1.0/(1.0 + np.exp(- data["freq3_3"])))),  (data["freq6_0"]))).astype(float)) * 2.0) < data["freq7_3"]).astype(float))),  ((4.04527997970581055)))) * (data["freq7_3"] - np.minimum( ((-(data["freq5_7"]))),  (0.272727)))) * ((data["freq5_7"] > data["skw_6"]).astype(float))) * (3.141593 - ((data["freq5_7"] > data["freq5_7"]).astype(float))))) +
         1.000000*np.tanh((-((((data["skw_2"] > ((((data["skw_6"] > 0.369565).astype(float)) > (data["freq5_10"] * (data["freq3_14"] * np.minimum( (((data["skw_2"] > (((-(((data["skw_2"] > ((data["skw_6"] > 0.369565).astype(float))).astype(float)))) > (0.369565 + np.minimum( (data["avg_0"]),  (data["kur_9"])))).astype(float))).astype(float))),  (((0.987342 > ((data["skw_6"] > np.maximum( (data["freq3_14"]),  (data["freq3_14"]))).astype(float))).astype(float))))))).astype(float))).astype(float)) * 2.0)))) +
         1.000000*np.tanh((1.0 - ((data["freq5_11"] > (data["freq4_11"] + (-(np.maximum( (data["freq10_11"]),  (np.cos((((data["freq3_8"] > np.maximum( (np.maximum( (0.369565),  (np.cos(data["freq4_11"])))),  ((-(1.0))))).astype(float)) * ((data["freq10_11"] - ((data["freq7_0"] > (0.434294 / 2.0)).astype(float))) * 2.0))))))))).astype(float)))) +
         1.000000*np.tanh((np.tanh(((data["freq4_14"] * (data["avg_0"] * (data["kur_15"] - (data["freq9_7"] + ((data["freq3_7"] + np.cos((((data["freq6_2"] < (1.0/(1.0 + np.exp(- data["avg_0"])))).astype(float)) - data["freq4_13"]))) + (np.maximum( ((data["avg_4"] * (data["freq9_7"] * 2.0))),  (((((data["freq7_8"] > data["avg_0"]).astype(float)) > data["freq7_8"]).astype(float)))) / 2.0)))))) * 2.0)) / 2.0)) +
         1.000000*np.tanh((((np.minimum( ((-(1.732051))),  (np.cos(data["skw_5"]))) < ((((data["skw_5"] > (np.minimum( ((-(1.732051))),  (1.722220)) * 2.0)).astype(float)) * (data["skw_5"] * 2.0)) + data["kur_9"])).astype(float)) - ((data["skw_5"] > (np.minimum( ((-(1.732051))),  ((-(1.722220)))) * 2.0)).astype(float)))))

    return Outputs(p)


def GP2(data):
    p = (1.000000*np.tanh((-(((9.869604 - ((-((9.869604 - data["kur_15"]))) * 2.0)) + (data["skw_12"] * (((data["freq4_8"] * 2.0) + np.cos(((data["skw_12"] * 2.0) * (2.685452 + (-((9.869604 - data["kur_15"]))))))) + np.cos(((data["freq4_8"] >= 0.0).astype(float))))))))) +
         1.000000*np.tanh(((data["skw_14"] - ((7.300000 + (-((data["kur_8"] + np.maximum( (data["kur_8"]),  ((data["freq7_2"] - ((7.300000 + (-((data["skw_14"] + np.maximum( (np.cos(data["skw_14"])),  (data["kur_14"])))))) * 2.0)))))))) * 2.0)) * (data["skw_11"] - (-3.0 + 0.915966)))) +
         1.000000*np.tanh((-((np.maximum( (np.maximum( (np.tanh((data["freq10_0"] * data["skw_6"]))),  (((np.maximum( (data["skw_2"]),  (data["freq1_9"])) + data["freq10_0"]) * 2.0)))),  (data["freq10_5"])) + (1.0/(1.0 + np.exp(- ((-((data["skw_6"] - (data["skw_10"] + data["freq3_12"])))) * 2.0)))))))) +
         1.000000*np.tanh(np.minimum( (((data["kur_11"] * 2.0) + np.sin(np.maximum( (np.sin((-(data["freq1_3"])))),  (data["kur_1"]))))),  (np.maximum( ((((((math.tanh(0.261497) > (data["kur_3"] - 0.735632)).astype(float)) < (data["kur_9"] + ((data["freq2_4"] >= 0.0).astype(float)))).astype(float)) * 2.0)),  (((((np.minimum( (np.maximum( (data["freq2_5"]),  ((-(data["kur_10"]))))),  ((1.0/(1.0 + np.exp(- data["kur_11"]))))) / 2.0) > data["kur_11"]).astype(float)) / 2.0)))))) +
         1.000000*np.tanh(((data["freq3_13"] * np.tanh((data["kur_4"] * (np.maximum( (data["skw_9"]),  (np.minimum( (1.464290),  (np.maximum( (np.maximum( (data["freq7_15"]),  (data["freq3_10"]))),  ((np.maximum( (np.cos(data["avg_9"])),  (data["freq3_13"])) / 2.0))))))) / 2.0)))) * 2.0)) +
         1.000000*np.tanh(((1.0/(1.0 + np.exp(- 1.142860))) * np.sin((-((-(np.minimum( ((np.maximum( (data["freq8_7"]),  ((((data["freq9_4"] * data["freq3_8"]) < (data["freq9_4"] * (1.0/(1.0 + np.exp(- 1.142860))))).astype(float)))) * (data["freq7_0"] - (1.0/(1.0 + np.exp(- data["freq8_7"])))))),  (((np.cos(data["avg_12"]) * 2.0) * 2.0)))))))))) +
         1.000000*np.tanh(((0.434294 < ((0.434294 < np.maximum( (((0.434294 < np.maximum( (data["skw_3"]),  ((np.minimum( (data["freq3_8"]),  ((((data["avg_11"] + (data["freq4_0"] - (((data["freq7_14"] < data["avg_11"]).astype(float)) * 2.0))) >= 0.0).astype(float)))) / 2.0)))).astype(float))),  ((data["freq7_14"] * (data["kur_3"] * ((np.cos(data["freq4_0"]) >= 0.0).astype(float))))))).astype(float))).astype(float))) +
         1.000000*np.tanh((np.cos((data["freq5_3"] * np.cos((((data["freq2_11"] >= 0.0).astype(float)) * 2.0)))) * np.minimum( (np.cos(data["freq7_13"])),  ((data["kur_4"] * ((data["kur_2"] < np.tanh(np.maximum( ((-(((data["kur_2"] < np.tanh(((data["freq4_4"] / 2.0) * ((np.minimum( ((data["freq7_7"] * 2.0)),  (data["freq4_12"])) >= 0.0).astype(float))))).astype(float))))),  (np.maximum( (-1.0),  (np.sin(data["skw_4"]))))))).astype(float))))))) +
         1.000000*np.tanh((((((1.0/(1.0 + np.exp(- data["freq10_3"]))) > np.cos((np.minimum( (np.minimum( (data["skw_15"]),  (((float(1.098360 >= 0.0)) / 2.0)))),  (np.minimum( (data["skw_15"]),  (np.minimum( (data["skw_15"]),  ((((np.cos((float(1.098360 >= 0.0))) >= 0.0).astype(float)) / 2.0))))))) / 2.0))).astype(float)) > np.cos((np.minimum( (data["avg_7"]),  (np.minimum( (data["avg_15"]),  ((data["freq7_15"] * np.minimum( ((data["skw_2"] * 2.0)),  (data["patient_id"]))))))) / 2.0))).astype(float))) +
         1.000000*np.tanh((-(np.tanh((np.sin(data["kur_10"]) * (np.maximum( (data["freq3_4"]),  (data["kur_9"])) * (np.maximum( (((data["kur_10"] < data["freq6_15"]).astype(float))),  ((data["kur_11"] * 2.0))) * ((np.minimum( (data["freq4_11"]),  (np.minimum( (data["kur_10"]),  ((-(np.maximum( (data["freq4_11"]),  ((np.sin((data["kur_9"] / 2.0)) / 2.0))))))))) * 2.0) * 2.0)))))))) +
         1.000000*np.tanh((data["freq3_2"] * (data["freq1_9"] * ((data["freq8_0"] > ((data["freq3_2"] * (data["freq1_9"] * (np.maximum( (data["skw_10"]),  ((data["freq3_2"] * (data["freq1_9"] * (data["freq1_9"] * (((data["freq4_10"] + np.minimum( (data["freq6_6"]),  ((1.0/(1.0 + np.exp(- data["freq3_2"])))))) * (data["skw_9"] * np.sin((data["freq9_1"] + data["freq3_2"])))) * 2.0)))))) * 2.0))) / 2.0)).astype(float))))) +
         1.000000*np.tanh(((((data["kur_0"] > (-((((0.318310 * ((3.714290 > (-(data["freq4_11"]))).astype(float))) * np.maximum( (data["skw_14"]),  (data["freq1_9"]))) * np.maximum( (data["freq10_3"]),  ((-(data["skw_0"])))))))).astype(float)) > np.maximum( (data["kur_0"]),  ((((-(0.318310)) > (-(((data["freq4_0"] > (-(((1.246580 > np.maximum( (data["freq4_11"]),  ((-(data["skw_7"]))))).astype(float))))).astype(float))))).astype(float))))).astype(float))) +
         1.000000*np.tanh((np.maximum( ((-(data["freq1_0"]))),  (((((1.142860 < data["freq5_12"]).astype(float)) > (data["skw_11"] * 0.367879)).astype(float)))) * (-(((data["skw_12"] > (1.0/(1.0 + np.exp(- (data["freq3_8"] * 0.367879))))).astype(float)))))) +
         1.000000*np.tanh((-(((1.414214 < np.maximum( (np.maximum( ((data["freq10_2"] * data["skw_2"])),  (data["avg_5"]))),  (np.maximum( (data["freq3_15"]),  (np.maximum( (np.maximum( (data["skw_2"]),  (data["avg_5"]))),  ((-(((-(((-((1.414214 * ((0.367879 * data["skw_13"]) + (1.0/(1.0 + np.exp(- 1.414214))))))) * 2.0))) * ((((data["freq10_2"] > 2.0).astype(float)) >= 0.0).astype(float)))))))))))).astype(float))))) +
         1.000000*np.tanh(np.sin(((data["skw_14"] > (np.maximum( (data["skw_11"]),  (((np.maximum( (((data["freq2_7"] < ((data["skw_11"] / 2.0) - np.maximum( (0.301030),  (np.maximum( ((data["skw_14"] - 0.301030)),  (np.minimum( (0.915966),  (((data["freq7_0"] >= 0.0).astype(float)))))))))).astype(float))),  ((((((data["kur_6"] - 2.680000) > (np.maximum( (data["skw_14"]),  (data["freq3_3"])) * 2.0)).astype(float)) < data["freq10_12"]).astype(float)))) * 2.0) * 2.0))) * 2.0)).astype(float)))) +
         0.993400*np.tanh(np.minimum( (data["kur_13"]),  (((0.367879 + (data["skw_12"] + ((0.367879 + (data["skw_12"] + ((((np.minimum( (np.minimum( (((data["freq6_12"] > data["avg_5"]).astype(float))),  ((data["freq9_1"] * (data["freq10_2"] + data["freq10_2"]))))),  (data["freq2_10"])) >= 0.0).astype(float)) / 2.0) * 2.0))) * 2.0))) * ((data["freq8_4"] < data["freq9_1"]).astype(float)))))))

    return Outputs(p)


def GP3(data):
    p = (1.000000*np.tanh(((data["skw_12"] + (31.006277 + data["skw_8"])) * (data["skw_8"] - np.maximum( (((data["skw_8"] - ((data["skw_12"] + (31.006277 + data["kur_2"])) * (data["skw_8"] - np.maximum( (data["kur_2"]),  (np.maximum( ((10.0)),  ((data["skw_8"] - data["freq3_4"])))))))) * 2.0)),  ((-(data["skw_8"]))))))) +
         1.000000*np.tanh((-(np.maximum( ((0.915966 - data["kur_11"])),  ((np.maximum( (((data["kur_11"] >= 0.0).astype(float))),  ((((np.maximum( (data["freq3_12"]),  (((((2.718282 + data["freq9_14"]) >= 0.0).astype(float)) - data["kur_11"]))) * 2.0) + np.maximum( (2.302585),  (data["freq6_15"]))) * 2.0))) - ((-(data["freq6_15"])) - (np.maximum( (data["kur_2"]),  (((0.915966 - data["freq9_14"]) - 3.400000))) * 2.0)))))))) +
         1.000000*np.tanh((-(((data["freq10_13"] + (np.tanh((data["freq1_4"] + (3.0 + ((data["freq9_3"] - ((data["skw_5"] > np.maximum( (np.sin((-(np.maximum( (data["freq8_7"]),  (data["freq10_6"])))))),  (data["freq1_15"]))).astype(float))) * 2.0)))) / 2.0)) + np.cos(((np.cos(np.minimum( (1.618034),  (data["freq7_7"]))) < ((data["kur_14"] >= 0.0).astype(float))).astype(float))))))) +
         1.000000*np.tanh(np.sin(np.tanh((data["kur_4"] * np.maximum( (((data["freq3_12"] + ((data["kur_11"] < np.cos(np.minimum( (data["freq9_9"]),  (np.minimum( (data["freq2_8"]),  ((data["kur_10"] * 2.0))))))).astype(float))) + np.maximum( ((data["freq2_12"] + (((1.0/(1.0 + np.exp(- data["kur_10"]))) < np.cos((np.sin(0.577216) * 2.0))).astype(float)))),  ((data["freq4_13"] * 2.0))))),  (np.sin((-(0.577216))))))))) +
         1.000000*np.tanh((0.052632 + (0.052632 + (0.052632 + np.minimum( (data["kur_0"]),  ((data["kur_0"] + (((-(data["freq6_5"])) + ((np.minimum( (data["freq8_11"]),  (((data["freq8_11"] >= 0.0).astype(float)))) + (((data["file_size"] < (((data["freq2_1"] * data["freq6_5"]) * 2.0) / 2.0)).astype(float)) * 2.0)) * 2.0)) * 2.0)))))))) +
         0.985200*np.tanh((-((np.sin(data["freq1_3"]) * np.minimum( (((data["skw_5"] > (-((data["freq4_2"] + (1.0/(1.0 + np.exp(- ((data["freq3_11"] > ((data["skw_5"] > (data["freq1_14"] + data["freq9_4"])).astype(float))).astype(float))))))))).astype(float))),  (((data["freq1_14"] + ((data["skw_5"] > (data["freq1_14"] + data["skw_2"])).astype(float))) + ((data["skw_5"] > (data["freq1_14"] + data["skw_2"])).astype(float))))))))) +
         1.000000*np.tanh(((data["skw_14"] > np.maximum( (data["freq1_14"]),  ((3.608700 * np.maximum( ((0.052632 - data["freq8_7"])),  (np.maximum( (data["freq3_15"]),  ((data["freq7_9"] * np.minimum( (data["kur_10"]),  ((np.minimum( (data["freq7_3"]),  (data["freq3_8"])) * ((data["avg_12"] < (np.tanh(data["skw_14"]) - data["freq8_7"])).astype(float)))))))))))))).astype(float))) +
         1.000000*np.tanh(np.minimum( (np.minimum( (data["kur_11"]),  (np.cos((np.cos(np.cos(np.minimum( (data["freq10_7"]),  (np.minimum( (np.minimum( (data["freq3_7"]),  (((data["avg_0"] - data["freq9_9"]) * 2.0)))),  (data["skw_2"])))))) * 2.0))))),  (np.cos((np.cos(np.cos(np.minimum( (data["skw_2"]),  (((data["freq9_9"] - np.maximum( (data["freq6_9"]),  (data["freq9_9"]))) * 2.0))))) * 2.0))))) +
         1.000000*np.tanh(np.sin((np.maximum( ((data["freq7_8"] - np.cos(((data["freq3_8"] > data["freq6_4"]).astype(float))))),  (np.maximum( (data["skw_6"]),  (((((data["kur_0"] > (np.cos((data["kur_0"] * ((data["freq2_1"] - data["skw_6"]) * 2.0))) / 2.0)).astype(float)) - np.cos(data["skw_6"])) * 3.0))))) / 2.0))) +
         1.000000*np.tanh((data["kur_0"] * ((-(((data["freq1_0"] * (data["freq4_6"] - np.maximum( (data["freq5_13"]),  (data["freq1_5"])))) - np.minimum( (data["freq8_3"]),  (np.maximum( (data["freq5_13"]),  (data["freq1_5"]))))))) - (data["freq4_8"] - data["freq1_5"])))) +
         1.000000*np.tanh(np.minimum( ((-(((np.cos(data["freq4_7"]) > ((np.cos(np.maximum( (data["kur_7"]),  (data["freq6_12"]))) > (data["skw_10"] / 2.0)).astype(float))).astype(float))))),  (np.minimum( ((-(((data["skw_2"] > np.maximum( ((4.066670 * np.sin(((-(data["freq10_11"])) / 2.0)))),  (np.cos(data["freq4_7"])))).astype(float))))),  (np.cos(np.cos(((np.sin((data["freq4_7"] + 31.006277)) * 2.0) * 2.0)))))))) +
         1.000000*np.tanh(np.sin(((-((data["skw_15"] * (data["skw_5"] * (((data["skw_15"] * (data["skw_5"] * (1.0/(1.0 + np.exp(- (-(np.cos(np.minimum( (np.tanh(data["skw_5"])),  ((np.cos((np.cos(data["skw_15"]) * 2.0)) + data["skw_15"]))))))))))) < data["freq2_9"]).astype(float)))))) / 2.0))) +
         1.000000*np.tanh((data["kur_10"] * (np.maximum( (data["freq4_11"]),  (data["skw_2"])) * (((((((((data["freq5_0"] + (data["freq4_2"] * 1.250000)) > data["kur_10"]).astype(float)) * 2.0) > ((data["kur_7"] > (np.maximum( (np.maximum( (data["freq4_11"]),  (data["skw_2"]))),  (data["freq10_1"])) * data["kur_10"])).astype(float))).astype(float)) > ((data["kur_7"] > (np.maximum( (data["freq4_11"]),  (data["freq10_1"])) * (data["freq5_2"] * 2.0))).astype(float))).astype(float)) * 2.0)))) +
         1.000000*np.tanh(((1.618034 < (((data["freq3_13"] - (((np.minimum( (data["freq10_0"]),  (np.sin(0.577216))) > np.minimum( (data["freq2_10"]),  (np.tanh((2.0 * data["freq8_1"]))))).astype(float)) * 2.0)) - ((((data["kur_12"] >= 0.0).astype(float)) * 2.0) * 2.0)) - (((np.minimum( (data["kur_12"]),  (np.minimum( (data["freq8_1"]),  (np.sin(np.minimum( (1.618034),  (data["freq10_0"]))))))) > (data["freq8_8"] / 2.0)).astype(float)) * 2.0))).astype(float))) +
         1.000000*np.tanh((-((((((np.minimum( (((data["freq5_0"] < ((data["freq5_0"] < (1.0/(1.0 + np.exp(- np.minimum( (data["freq4_1"]),  (data["freq4_15"])))))).astype(float))).astype(float))),  (((data["freq4_15"] > np.cos((1.0/(1.0 + np.exp(- data["freq4_15"]))))).astype(float)))) * 2.0) * 2.0) * 2.0) / 2.0) * np.maximum( ((data["freq2_13"] + 0.828283)),  (np.minimum( (data["freq4_15"]),  (np.minimum( (data["freq7_12"]),  ((((data["freq5_0"] < 0.828283).astype(float)) * 2.0))))))))))) +
         1.000000*np.tanh((np.sin((data["freq1_2"] - np.tanh((np.tanh((data["skw_0"] + ((data["freq1_2"] < data["skw_0"]).astype(float)))) + np.minimum( (((np.tanh(data["freq4_7"]) > ((data["freq4_8"] >= 0.0).astype(float))).astype(float))),  (data["skw_4"])))))) * ((data["freq8_3"] > ((np.sin(np.sin((data["freq4_8"] - np.tanh((data["skw_0"] + ((data["skw_4"] > ((data["freq4_8"] >= 0.0).astype(float))).astype(float))))))) >= 0.0).astype(float))).astype(float)))))
    return Outputs(p)


def GP4(data):
    p = (1.000000*np.tanh((data["kur_12"] - (-((data["kur_6"] - 31.006277))))) +
         1.000000*np.tanh((-3.0 - ((data["freq7_14"] - (-((1.0/(1.0 + np.exp(- np.maximum( (((1.0/(1.0 + np.exp(- data["freq5_3"]))) * 1.282427)),  ((-((-(((2.307690 * (np.minimum( (data["freq10_6"]),  (data["kur_15"])) + 1.282427)) * 2.0))))))))))))) * ((2.307690 * (data["kur_15"] + 1.282427)) * 2.0)))) +
         1.000000*np.tanh(((np.tanh((-((data["freq2_12"] + np.maximum( ((data["freq1_9"] + data["skw_13"])),  (np.minimum( (data["avg_10"]),  ((data["freq1_15"] + 0.915966))))))))) - (np.maximum( (data["freq1_9"]),  (np.maximum( ((data["freq10_5"] + (1.0/(1.0 + np.exp(- ((data["freq4_13"] - data["avg_3"]) + np.cos(np.tanh(np.minimum( (data["avg_10"]),  (data["freq10_5"])))))))))),  (data["skw_2"])))) * 2.0)) / 2.0)) +
         1.000000*np.tanh(np.minimum( (data["kur_11"]),  ((data["freq3_9"] * (data["kur_4"] - ((data["kur_6"] + np.cos(((1.0/(1.0 + np.exp(- (data["freq5_13"] - data["freq6_4"])))) * data["freq5_0"]))) * (data["freq1_4"] * (-(np.cos((data["freq3_9"] * (1.0/(1.0 + np.exp(- (data["kur_11"] - (np.cos(data["freq5_2"]) + np.cos(data["freq3_9"]))))))))))))))))) +
         1.000000*np.tanh(((data["kur_1"] > ((data["freq7_0"] < ((((data["freq10_2"] / 2.0) < (np.cos((((data["freq10_13"] > (-((((((np.tanh(2.685452) + data["freq8_2"]) * data["freq10_2"]) >= 0.0).astype(float)) * data["freq7_11"])))).astype(float)) - ((data["freq5_5"] > (data["freq4_12"] + data["freq5_5"])).astype(float)))) + data["skw_1"])).astype(float)) + data["skw_1"])).astype(float))).astype(float))) +
         1.000000*np.tanh((data["freq1_9"] * np.sin(np.minimum( (np.maximum( (data["freq4_13"]),  (np.minimum( (np.maximum( (data["kur_15"]),  (np.minimum( ((data["skw_4"] * 2.0)),  (data["freq6_3"]))))),  (np.sin(np.sin((data["freq9_6"] + data["skw_4"])))))))),  (np.sin(np.sin((np.maximum( (data["kur_15"]),  (np.minimum( (np.maximum( (data["kur_15"]),  (np.minimum( (data["freq8_6"]),  (data["freq6_3"]))))),  (data["freq6_3"])))) + data["skw_4"])))))))) +
         1.000000*np.tanh(((np.minimum( (data["kur_10"]),  (np.minimum( (((-(((1.0 < data["freq10_5"]).astype(float)))) + (data["freq3_7"] - np.maximum( (data["kur_6"]),  (np.sin(np.minimum( (data["freq6_11"]),  (((-(((1.0 < data["freq10_5"]).astype(float)))) + (data["freq9_13"] - data["skw_7"])))))))))),  (((-((data["freq3_13"] * 2.0))) + (data["freq10_5"] / 2.0)))))) >= 0.0).astype(float))) +
         1.000000*np.tanh((data["kur_0"] * ((data["freq4_6"] + np.tanh(np.minimum( (2.685452),  (np.tanh(((data["freq4_6"] + np.tanh(2.307690)) - data["freq9_8"])))))) + np.tanh(((data["freq6_9"] * (data["freq6_9"] - np.minimum( (data["avg_15"]),  (np.tanh((data["kur_0"] + (1.0/(1.0 + np.exp(- np.tanh(2.0)))))))))) * (2.307690 - data["freq4_6"])))))) +
         1.000000*np.tanh((-(((((-((((-(((data["freq4_10"] < (data["freq2_0"] - ((2.302585 > (-((data["freq2_0"] - ((2.302585 > data["kur_13"]).astype(float)))))).astype(float)))).astype(float)))) < np.minimum( (data["freq5_5"]),  ((np.tanh((data["freq2_10"] + np.minimum( (data["skw_11"]),  ((np.sin(data["freq3_3"]) + data["freq8_3"]))))) / 2.0)))).astype(float)))) * 2.0) < np.minimum( (data["freq5_5"]),  ((data["freq3_13"] / 2.0)))).astype(float))))) +
         1.000000*np.tanh((data["kur_3"] * (((data["freq8_5"] - np.sin(np.maximum( (data["kur_5"]),  ((data["skw_1"] * 2.0))))) - np.tanh(np.sin((data["kur_3"] + data["freq9_13"])))) - np.sin(data["freq8_7"])))) +
         1.000000*np.tanh((((data["skw_3"] > np.sin((1.0/(1.0 + np.exp(- (-((np.tanh((data["freq7_7"] * (0.369863 * np.maximum( (data["freq10_4"]),  (np.maximum( (((data["skw_11"] - data["freq8_2"]) - (-((data["freq7_7"] - 0.369863))))),  ((data["freq7_7"] - np.sin((1.0/(1.0 + np.exp(- 0.369863)))))))))))) * 2.0)))))))).astype(float)) * 2.0)) +
         1.000000*np.tanh(((data["freq3_8"] - np.tanh(((((data["freq10_14"] > data["skw_14"]).astype(float)) > data["skw_14"]).astype(float)))) * ((((data["freq9_4"] > data["freq10_11"]).astype(float)) > np.maximum( (((data["skw_11"] > data["skw_14"]).astype(float))),  (np.maximum( ((((data["skw_11"] < 0.301030).astype(float)) / 2.0)),  (np.cos((-(((data["freq8_15"] > np.tanh(data["skw_14"])).astype(float)))))))))).astype(float)))) +
         1.000000*np.tanh(((data["freq9_4"] > (np.maximum( (data["freq4_15"]),  ((((data["freq4_0"] * data["avg_6"]) >= 0.0).astype(float)))) - (-(np.maximum( ((data["freq7_12"] * (((data["freq4_0"] * np.cos(((data["freq7_12"] < data["freq4_10"]).astype(float)))) < ((data["freq4_15"] < np.sin(((data["freq5_5"] > data["freq7_12"]).astype(float)))).astype(float))).astype(float)))),  (data["avg_7"])))))).astype(float))) +
         1.000000*np.tanh(np.tanh(np.tanh((data["file_size"] * np.minimum( (((data["freq8_10"] > np.minimum( (np.minimum( (data["freq10_3"]),  (((-(data["avg_10"])) * 2.0)))),  ((-(data["freq10_3"]))))).astype(float))),  (((np.maximum( ((-(data["freq10_3"]))),  ((-(np.cos(np.minimum( (data["freq7_14"]),  (((1.0/(1.0 + np.exp(- data["freq10_3"]))) * 2.0)))))))) * 2.0) * 2.0))))))) +
         1.000000*np.tanh(np.minimum( ((data["avg_6"] * np.minimum( (data["kur_11"]),  (np.maximum( (((data["kur_1"] > data["freq5_6"]).astype(float))),  (np.cos(data["kur_11"]))))))),  (np.minimum( ((data["avg_3"] * (data["skw_10"] * ((data["kur_1"] > np.maximum( (data["freq5_6"]),  (data["avg_6"]))).astype(float))))),  (((-(data["kur_1"])) * (data["skw_10"] * ((data["kur_1"] > np.maximum( (data["freq5_6"]),  ((data["skw_10"] / 2.0)))).astype(float))))))))) +
         1.000000*np.tanh(((((np.maximum( (data["skw_2"]),  (np.maximum( ((1.0/(1.0 + np.exp(- (data["freq10_6"] - ((data["skw_2"] < data["skw_13"]).astype(float))))))),  ((data["freq4_11"] * ((0.434294 > (-((data["freq10_6"] + 0.127660)))).astype(float))))))) * ((data["skw_6"] > 0.434294).astype(float))) * np.cos(np.sin(data["freq10_6"]))) * ((data["skw_6"] > 0.434294).astype(float))) * np.cos(data["skw_2"]))))
    return Outputs(p)

def GP5(data):
    p = (1.000000*np.tanh((data["freq7_2"] - ((0.301030 * (15.333300 - (np.maximum( ((np.maximum( ((15.333300 - 15.333300)),  ((data["kur_6"] - (((((0.577216 * 15.333300) * 2.0) - data["kur_10"]) * (15.333300 - (np.maximum( (data["kur_6"]),  (data["kur_12"])) / 2.0))) * 2.0)))) - (-((data["skw_0"] / 2.0))))),  (data["kur_6"])) / 2.0))) * 2.0))) +
         1.000000*np.tanh(((data["kur_12"] - (((data["skw_5"] * (data["kur_12"] + 3.141593)) - ((-(((data["kur_12"] + 3.141593) * 2.0))) * 2.0)) - ((-(((data["skw_9"] + (data["skw_5"] * data["kur_12"])) * 2.0))) * 2.0))) * ((data["kur_12"] < (np.cos(np.cos(np.cos(0.313433))) / 2.0)).astype(float)))) +
         1.000000*np.tanh((data["kur_0"] - (data["freq10_12"] + np.cos(((data["avg_9"] < (-((data["freq10_0"] + (((data["freq10_0"] + (((-(((data["freq5_12"] < np.minimum( (((data["avg_3"] < data["freq2_3"]).astype(float))),  (((-((-((data["freq10_0"] + data["freq7_7"]))))) + ((data["freq10_8"] > data["freq10_0"]).astype(float)))))).astype(float)))) >= 0.0).astype(float))) > np.minimum( (data["freq8_11"]),  (data["avg_1"]))).astype(float)))))).astype(float)))))) +
         1.000000*np.tanh((-((((((data["freq2_0"] + np.maximum( (np.cos(data["skw_1"])),  ((-(np.sin(data["freq6_12"])))))) >= 0.0).astype(float)) > (((data["kur_11"] - np.minimum( (data["skw_1"]),  (np.tanh(np.minimum( (data["kur_11"]),  (np.sin(np.sin(data["freq6_12"])))))))) > (1.0/(1.0 + np.exp(- np.minimum( (-1.0),  (data["kur_11"])))))).astype(float))).astype(float))))) +
         1.000000*np.tanh((data["freq7_0"] * ((((data["avg_2"] > np.tanh((1.0/(1.0 + np.exp(- data["skw_2"]))))).astype(float)) > (np.cos(np.maximum( (np.maximum( (data["freq3_13"]),  (np.maximum( (data["skw_2"]),  (np.tanh(((data["kur_6"] < data["freq9_7"]).astype(float)))))))),  (((1.0/(1.0 + np.exp(- data["freq4_5"]))) - (-(np.maximum( (((data["freq3_12"] < np.cos(data["freq9_7"])).astype(float))),  (data["freq4_5"])))))))) * 2.0)).astype(float)))) +
         1.000000*np.tanh(((data["kur_4"] * 1.732051) * (data["freq4_13"] + ((data["skw_12"] < ((data["freq5_10"] * np.minimum( (data["freq4_13"]),  ((((data["freq10_2"] * (3.363640 - (-((data["kur_4"] * data["freq4_13"]))))) * np.minimum( (data["freq4_13"]),  (np.minimum( (1.732051),  (np.minimum( ((-(1.732051))),  (1.732051))))))) / 2.0)))) / 2.0)).astype(float))))) +
         1.000000*np.tanh((data["freq8_10"] * np.minimum( ((data["kur_3"] * data["freq8_10"])),  (np.maximum( ((np.cos(np.minimum( (data["freq8_10"]),  ((data["kur_3"] * np.minimum( (data["kur_3"]),  (np.maximum( ((data["kur_3"] * data["freq8_10"])),  (data["freq2_1"])))))))) * np.maximum( ((data["freq3_8"] + data["kur_3"])),  (data["avg_8"])))),  (np.minimum( (((data["freq8_10"] < (7.41375255584716797)).astype(float))),  (data["freq2_1"])))))))) +
         1.000000*np.tanh(np.maximum( ((np.maximum( ((data["freq4_11"] - ((1.0/(1.0 + np.exp(- np.minimum( (data["freq8_5"]),  (2.250000))))) * 2.0))),  (((((data["avg_2"] > (2.250000 - ((data["avg_2"] > (2.250000 - np.tanh(((0.567143 < data["freq4_10"]).astype(float))))).astype(float)))).astype(float)) * 2.0) * 2.0))) / 2.0)),  ((((0.567143 < np.minimum( (data["freq8_5"]),  (data["skw_7"]))).astype(float)) * 2.0)))) +
         1.000000*np.tanh(((2.718282 * np.minimum( (data["freq5_13"]),  (data["freq8_1"]))) * (data["kur_2"] * np.maximum( (data["freq5_13"]),  (((data["kur_11"] * ((np.cos((data["freq2_6"] * (data["kur_11"] * ((np.cos(data["kur_2"]) > np.cos(np.minimum( (data["freq8_1"]),  (data["kur_11"])))).astype(float))))) > np.tanh(np.cos(data["kur_11"]))).astype(float))) / 2.0)))))) +
         1.000000*np.tanh(np.sin(np.sin((data["kur_6"] * (data["skw_15"] * ((0.617647 + ((np.tanh(data["freq1_12"]) >= 0.0).astype(float))) + (np.tanh((data["freq1_9"] - np.tanh(np.sin((data["skw_15"] * (0.617647 + np.sin(data["skw_15"]))))))) / 2.0))))))) +
         1.000000*np.tanh(((data["file_size"] + ((data["freq4_5"] >= 0.0).astype(float))) * (-(((data["freq3_8"] < (data["freq2_3"] + ((data["freq4_9"] + data["kur_7"]) - (((data["file_size"] / 2.0) < (data["kur_7"] * (31.006277 * ((data["avg_10"] < np.cos(((data["file_size"] < ((np.sin(data["freq3_8"]) < (data["freq4_9"] + data["kur_7"])).astype(float))).astype(float)))).astype(float))))).astype(float))))).astype(float)))))) +
         1.000000*np.tanh(((((data["kur_13"] >= 0.0).astype(float)) + np.sin((((data["kur_13"] >= 0.0).astype(float)) + data["kur_10"]))) * ((np.minimum( (data["freq3_10"]),  (np.cos((np.minimum( (data["avg_7"]),  (((data["freq6_4"] - (data["freq3_10"] + ((data["avg_7"] > (10.0)).astype(float)))) / 2.0))) * 2.0)))) > np.tanh(np.minimum( ((np.minimum( (data["freq3_10"]),  (data["freq3_10"])) / 2.0)),  (data["skw_9"])))).astype(float)))) +
         1.000000*np.tanh(np.sin((data["freq1_8"] * ((np.sin(np.minimum( (np.minimum( (np.minimum( (data["freq6_3"]),  (np.sin(np.sin((((data["freq5_6"] >= 0.0).astype(float)) * (data["freq8_15"] + data["freq8_15"]))))))),  (np.sin(data["freq2_4"])))),  (np.sin((data["freq6_0"] * (data["freq5_6"] + np.tanh(np.tanh(data["freq6_3"])))))))) >= 0.0).astype(float))))) +
         1.000000*np.tanh((0.567143 - np.tanh(np.maximum( (np.maximum( ((data["freq7_9"] / 2.0)),  (((((data["skw_14"] > data["skw_14"]).astype(float)) > data["skw_14"]).astype(float))))),  ((((1.0/(1.0 + np.exp(- np.cos(((data["freq3_15"] / 2.0) * 2.0))))) < (1.0/(1.0 + np.exp(- ((((data["freq5_1"] >= 0.0).astype(float)) < np.maximum( (data["freq3_15"]),  (data["freq10_1"]))).astype(float)))))).astype(float))))))) +
         1.000000*np.tanh(np.tanh(((data["kur_2"] * 2.0) * (np.minimum( (data["skw_2"]),  ((-((((-(((data["skw_13"] < (np.tanh(data["kur_2"]) * data["avg_2"])).astype(float)))) < (data["skw_13"] * ((data["skw_13"] < (np.tanh(data["kur_2"]) * (-(((data["skw_2"] < (np.tanh(data["kur_2"]) * data["avg_2"])).astype(float)))))).astype(float)))).astype(float)))))) + (-(data["skw_0"])))))) +
         1.000000*np.tanh((-(np.maximum( (np.maximum( (((data["skw_2"] > 1.282427).astype(float))),  ((-((data["kur_1"] * (data["file_size"] + np.maximum( (data["freq1_1"]),  ((-(((np.cos((data["file_size"] + np.maximum( (data["skw_2"]),  (0.915966)))) / 2.0) / 2.0)))))))))))),  ((-(np.sin((1.282427 * (data["file_size"] + 1.282427)))))))))))
    return Outputs(p)


def GP6(data):
    p = (1.000000*np.tanh((((data["kur_12"] * np.sin((np.maximum( (4.615380),  (data["kur_12"])) * 2.0))) / 2.0) - ((data["avg_3"] + np.maximum( (5.900000),  (4.615380))) + (0.636620 - np.minimum( (data["avg_13"]),  (((data["kur_12"] + np.minimum( (data["avg_13"]),  ((((1.0/(1.0 + np.exp(- (data["kur_12"] + data["kur_12"])))) + ((data["freq5_0"] < data["freq9_2"]).astype(float))) * 2.0)))) * 2.0))))))) +
         1.000000*np.tanh(((data["skw_14"] - (5.0)) - np.minimum( (((((data["freq6_1"] > ((-1.0 < data["freq4_2"]).astype(float))).astype(float)) * (data["freq1_5"] - np.maximum( (data["skw_5"]),  (np.minimum( (data["skw_14"]),  (data["freq4_2"])))))) - (np.maximum( (((data["kur_14"] - (5.0)) - np.minimum( ((data["skw_14"] - (5.0))),  (np.maximum( (-1.0),  (0.567143)))))),  (data["skw_8"])) / 2.0))),  (data["skw_5"])))) +
         1.000000*np.tanh((-(np.maximum( (((-((-((data["freq8_9"] + data["freq4_14"]))))) - np.tanh(data["skw_6"]))),  ((data["freq10_0"] + ((data["freq7_11"] < ((data["freq5_10"] < np.sin(((data["freq7_8"] < np.sin(((data["freq1_0"] < (data["freq1_7"] + ((data["freq7_8"] < np.sin(data["freq5_10"])).astype(float)))).astype(float)))).astype(float)))).astype(float))).astype(float)))))))) +
         1.000000*np.tanh(np.minimum( ((-(((np.minimum( (data["kur_13"]),  (data["skw_10"])) < np.minimum( (data["freq10_12"]),  (data["skw_10"]))).astype(float))))),  ((-(((((-((-(np.minimum( (data["kur_11"]),  (0.426471)))))) * 2.0) < np.minimum( (np.minimum( (data["freq9_3"]),  ((-(data["freq8_8"]))))),  ((-((-(np.minimum( (data["kur_11"]),  (data["avg_7"]))))))))).astype(float))))))) +
         1.000000*np.tanh(((data["kur_1"] > ((np.tanh((data["kur_13"] + np.minimum( (data["freq7_14"]),  ((np.maximum( (data["skw_6"]),  (np.minimum( (np.maximum( ((np.maximum( (data["freq5_2"]),  (((data["skw_6"] >= 0.0).astype(float)))) / 2.0)),  (data["kur_0"]))),  (data["kur_0"])))) / 2.0))))) < np.minimum( ((np.tanh(data["kur_1"]) + data["freq4_7"])),  ((np.sin((data["skw_14"] * (data["freq4_7"] + data["kur_1"]))) * data["avg_13"])))).astype(float))).astype(float))) +
         1.000000*np.tanh(((data["kur_10"] * 2.0) * (((data["freq2_0"] / 2.0) > (np.minimum( (data["kur_4"]),  ((-3.0 * ((2.718282 < (data["skw_2"] * np.maximum( (data["avg_2"]),  ((np.cos((data["kur_10"] * (np.maximum( ((data["freq2_0"] / 2.0)),  (data["kur_4"])) * 2.0))) * 2.0))))).astype(float))))) / 2.0)).astype(float)))) +
         1.000000*np.tanh((((np.minimum( (0.360000),  (np.sin((data["freq9_9"] + (-(data["freq7_3"])))))) + (-(((((data["freq3_8"] >= 0.0).astype(float)) > (data["skw_6"] * 2.0)).astype(float))))) / 2.0) + (((1.0/(1.0 + np.exp(- (((data["freq9_9"] + data["freq7_3"]) >= 0.0).astype(float))))) < np.sin((data["freq4_11"] - (((data["freq10_9"] + (-(data["freq9_9"]))) < data["freq7_3"]).astype(float))))).astype(float)))) +
         1.000000*np.tanh((((((data["freq4_13"] * data["kur_4"]) + np.sin(data["kur_1"])) / 2.0) + ((data["freq7_0"] > (((((data["freq5_3"] < np.maximum( (data["kur_1"]),  (((data["freq5_10"] * 2.0) - (1.0/(1.0 + np.exp(- data["freq4_13"]))))))).astype(float)) > ((np.cos(data["freq5_0"]) < (-(data["skw_8"]))).astype(float))).astype(float)) * 2.0)).astype(float))) / 2.0)) +
         1.000000*np.tanh((-((data["freq3_4"] * (data["freq3_4"] * (data["freq3_4"] * (data["freq3_4"] * (data["freq3_4"] * (data["freq3_14"] * ((data["avg_9"] > (1.0/(1.0 + np.exp(- np.sin(np.sin(np.sin(((data["avg_9"] > (1.0/(1.0 + np.exp(- np.sin(np.maximum( ((1.0/(1.0 + np.exp(- np.maximum( (data["freq6_7"]),  (np.maximum( (data["freq6_7"]),  (data["freq6_7"])))))))),  (data["freq6_7"]))))))).astype(float))))))))).astype(float))))))))))) +
         1.000000*np.tanh((np.minimum( (data["freq8_0"]),  (np.maximum( (((data["freq10_9"] < np.minimum( (data["freq8_0"]),  (((-(((data["skw_2"] > data["freq8_0"]).astype(float)))) / 2.0)))).astype(float))),  (((-((-(((data["avg_1"] < data["freq1_8"]).astype(float)))))) / 2.0))))) * ((data["skw_2"] < np.minimum( (data["skw_7"]),  (np.tanh(np.minimum( (data["freq5_0"]),  (((-(((((data["skw_2"] > 1.414214).astype(float)) < ((data["freq8_14"] >= 0.0).astype(float))).astype(float)))) / 2.0))))))).astype(float)))) +
         1.000000*np.tanh((data["freq10_14"] * ((((2.0 < (data["freq1_3"] * np.tanh(np.cos(((data["freq4_14"] >= 0.0).astype(float)))))).astype(float)) > (-(((0.0 > (-(np.minimum( (data["kur_9"]),  ((((data["freq2_4"] > np.minimum( (np.cos((data["skw_2"] + np.tanh(((data["kur_9"] >= 0.0).astype(float)))))),  (0.567143))).astype(float)) - ((data["freq9_4"] < ((data["kur_5"] < 1.618034).astype(float))).astype(float)))))))).astype(float))))).astype(float)))) +
         1.000000*np.tanh((-(((((0.0 > (-(((((0.0 > data["skw_3"]).astype(float)) > np.maximum( ((data["freq4_4"] * np.sin(np.sin(((data["skw_2"] > ((data["skw_3"] >= 0.0).astype(float))).astype(float)))))),  (data["kur_9"]))).astype(float))))).astype(float)) > np.maximum( ((-(np.sin((((data["skw_12"] * 2.0) > ((((3.916670 > data["skw_2"]).astype(float)) > (data["skw_12"] * 2.0)).astype(float))).astype(float)))))),  (data["kur_10"]))).astype(float))))) +
         1.000000*np.tanh(((0.367879 > (data["kur_14"] + (((data["kur_13"] * np.tanh((-(np.minimum( (np.tanh((data["skw_3"] + ((0.367879 > (data["kur_14"] + ((float(0.577216 >= 0.0)) * data["skw_13"]))).astype(float))))),  (np.minimum( (0.577216),  (((data["freq10_1"] > ((((0.577216 * data["freq10_1"]) >= 0.0).astype(float)) - np.sin(((0.367879 > data["kur_13"]).astype(float))))).astype(float)))))))))) >= 0.0).astype(float)))).astype(float))) +
         1.000000*np.tanh((-(np.sin((np.minimum( ((data["freq6_0"] + np.minimum( (data["freq4_10"]),  ((np.sin(np.minimum( (data["freq4_10"]),  (data["freq4_14"]))) * 2.0))))),  (((data["freq6_0"] + np.minimum( (data["freq4_10"]),  (np.sin(data["freq4_13"])))) + np.minimum( (np.tanh(data["freq4_10"])),  ((np.sin(data["freq4_10"]) * 2.0)))))) * ((data["freq4_13"] > (1.0/(1.0 + np.exp(- (-(data["skw_6"])))))).astype(float))))))) +
         1.000000*np.tanh((((data["freq8_3"] > (0.261497 + (np.minimum( (data["freq5_3"]),  ((((data["kur_0"] * 2.0) > (-(data["kur_0"]))).astype(float)))) / 2.0))).astype(float)) * (data["kur_0"] * (2.718282 * ((data["freq5_3"] > (0.261497 + ((data["freq6_0"] + data["freq6_8"]) / 2.0))).astype(float)))))) +
         1.000000*np.tanh((data["skw_14"] * ((data["skw_14"] * (data["skw_14"] * data["freq1_9"])) * (-(((np.tanh(data["freq6_10"]) > (data["freq7_14"] + ((((data["freq6_10"] > np.maximum( (data["freq1_9"]),  (data["skw_14"]))).astype(float)) < data["freq4_3"]).astype(float)))).astype(float))))))))

    return Outputs(p)


def GP7(data):
    p = (1.000000*np.tanh((-((np.maximum( ((data["skw_13"] * (data["freq4_12"] * ((np.maximum( (data["skw_10"]),  ((data["kur_13"] + (data["skw_10"] - data["skw_10"])))) + ((data["kur_12"] * data["freq7_10"]) + 9.869604)) * 2.0)))),  ((data["skw_13"] * data["freq7_10"]))) + ((data["kur_12"] * data["freq7_10"]) + 9.869604))))) +
         1.000000*np.tanh((((data["kur_14"] * 0.434294) - (np.sin(np.minimum( ((data["skw_8"] * 2.0)),  (data["skw_8"]))) + 3.0)) * (np.cos((data["skw_8"] * np.sin(3.0))) * (np.minimum( (data["freq6_8"]),  (data["skw_15"])) + (0.693147 - np.minimum( (data["skw_8"]),  (np.minimum( (-2.0),  ((data["freq6_8"] + np.cos((data["skw_8"] * data["skw_8"])))))))))))) +
         1.000000*np.tanh((-((np.sin(((data["avg_5"] > data["kur_4"]).astype(float))) + (data["freq1_0"] + (((data["freq7_7"] * (((((((-(data["freq3_3"])) / 2.0) >= 0.0).astype(float)) / 2.0) > (data["freq3_4"] / 2.0)).astype(float))) < ((data["freq4_7"] >= 0.0).astype(float))).astype(float))))))) +
         1.000000*np.tanh(np.minimum( (np.minimum( ((((((data["freq9_12"] - np.minimum( (data["kur_15"]),  (np.maximum( (data["skw_0"]),  (data["freq3_8"]))))) >= 0.0).astype(float)) < (data["freq6_2"] * data["freq3_8"])).astype(float))),  (((data["skw_0"] < ((((data["freq8_6"] < data["skw_0"]).astype(float)) < data["skw_0"]).astype(float))).astype(float))))),  ((-((data["freq1_10"] + (data["freq9_12"] - np.minimum( (data["kur_15"]),  (np.maximum( ((-((data["skw_0"] + data["skw_0"])))),  (data["freq6_2"]))))))))))) +
         1.000000*np.tanh(((data["kur_1"] > ((((data["freq8_0"] > np.cos(((data["kur_3"] + np.maximum( ((float(-3.0 > 1.732051))),  (data["freq7_9"]))) / 2.0))).astype(float)) < ((((data["freq8_0"] > np.cos(np.minimum( (np.tanh(np.cos(data["freq7_9"]))),  ((data["freq1_13"] + data["freq7_8"]))))).astype(float)) < ((((data["kur_3"] / 2.0) / 2.0) + np.maximum( (np.cos(data["freq4_11"])),  (data["freq7_9"]))) / 2.0)).astype(float))).astype(float))).astype(float))) +
         1.000000*np.tanh(np.minimum( (np.minimum( (data["kur_11"]),  ((((data["freq2_13"] * (1.0/(1.0 + np.exp(- ((np.cos(data["kur_11"]) > (-((data["freq10_10"] * 2.0)))).astype(float)))))) > np.cos((data["freq10_10"] * 2.0))).astype(float))))),  ((((data["freq10_3"] * (1.0/(1.0 + np.exp(- np.cos((data["freq10_10"] * 2.0)))))) > np.cos(data["kur_11"])).astype(float))))) +
         1.000000*np.tanh(np.maximum( (np.maximum( ((np.maximum( ((-(((data["freq9_12"] < (float(0.101010 >= 0.0))).astype(float))))),  ((data["avg_7"] * 2.0))) * data["kur_0"])),  ((-(((0.567143 < ((((1.0/(1.0 + np.exp(- (data["freq8_9"] * 2.0)))) * 2.0) < ((data["skw_5"] < (1.0/(1.0 + np.exp(- np.maximum( (data["freq5_1"]),  (data["freq9_12"])))))).astype(float))).astype(float))).astype(float))))))),  ((-(np.sin((data["avg_7"] * data["freq5_3"]))))))) +
         1.000000*np.tanh(np.sin(np.minimum( ((((((data["skw_5"] >= 0.0).astype(float)) * np.cos((1.282427 - np.tanh(np.sin(data["skw_2"]))))) >= 0.0).astype(float))),  (np.cos((1.282427 - np.minimum( (((1.282427 * (data["freq3_7"] * ((data["skw_2"] > (((-(data["freq10_10"])) >= 0.0).astype(float))).astype(float)))) * 2.0)),  (data["skw_5"])))))))) +
         1.000000*np.tanh(((np.cos(data["freq4_11"]) + data["kur_5"]) * np.sin((((data["skw_3"] * 2.741940) > np.cos(((data["freq6_14"] > ((data["freq8_4"] > np.sin(np.sin((data["freq4_11"] + (data["freq4_11"] + np.cos(((2.685452 > np.maximum( (data["freq6_14"]),  (data["kur_5"]))).astype(float)))))))).astype(float))).astype(float)))).astype(float))))) +
         1.000000*np.tanh(((np.cos(data["skw_13"]) < np.minimum( (((np.cos(np.sin(np.sin(np.maximum( (data["freq2_0"]),  (np.maximum( (data["freq7_8"]),  (np.sin(np.cos(data["freq7_8"]))))))))) < np.minimum( (data["freq7_8"]),  ((-(((np.minimum( (np.sin(data["freq3_13"])),  (np.sin(np.tanh(np.sin(np.maximum( (data["kur_10"]),  (np.maximum( (data["freq2_0"]),  (data["freq2_0"]))))))))) * 2.0) * 2.0)))))).astype(float))),  (np.maximum( (data["freq2_0"]),  (data["freq3_13"]))))).astype(float))) +
         1.000000*np.tanh((data["kur_0"] * (data["freq6_15"] - (data["freq5_8"] * ((((data["freq8_8"] * (data["freq6_15"] - np.tanh(((data["freq5_8"] >= 0.0).astype(float))))) * ((np.tanh((data["freq5_8"] * np.tanh(((data["freq4_4"] > (data["freq6_15"] - (np.tanh(np.tanh(np.maximum( (np.cos(2.685452)),  (np.tanh((((1.0/(1.0 + np.exp(- data["freq6_15"]))) >= 0.0).astype(float))))))) / 2.0))).astype(float))))) >= 0.0).astype(float))) >= 0.0).astype(float)))))) +
         1.000000*np.tanh((-(((data["skw_2"] - np.minimum( (data["skw_2"]),  (np.minimum( (data["skw_2"]),  (np.cos((data["freq7_9"] * (((np.cos((data["freq1_11"] - ((np.minimum( (np.cos(data["freq6_7"])),  (data["freq7_9"])) < np.tanh(np.sin(data["skw_7"]))).astype(float)))) >= 0.0).astype(float)) * data["freq9_4"])))))))) * 2.0)))) +
         1.000000*np.tanh(((((((data["skw_0"] > data["kur_11"]).astype(float)) < ((((data["freq10_15"] > ((data["freq6_6"] > data["freq1_8"]).astype(float))).astype(float)) < np.minimum( ((data["skw_14"] * data["freq3_4"])),  (np.minimum( ((data["skw_14"] * data["freq6_8"])),  ((data["freq1_8"] + np.maximum( (data["freq6_8"]),  (((data["skw_14"] > (2.0 + data["freq3_3"])).astype(float)))))))))).astype(float))).astype(float)) * 2.0) * 2.0)) +
         1.000000*np.tanh((-(((((data["skw_12"] > (1.0/(1.0 + np.exp(- (data["kur_0"] / 2.0))))).astype(float)) > (-(((data["skw_12"] > (1.0/(1.0 + np.exp(- ((data["freq8_13"] * 2.0) * ((-(((data["freq7_1"] < (((data["freq8_13"] * 2.0) > (np.maximum( (data["freq1_3"]),  ((data["kur_0"] / 2.0))) + ((data["freq3_10"] * 2.0) + (data["freq10_15"] * 2.0)))).astype(float))).astype(float)))) / 2.0)))))).astype(float))))).astype(float))))) +
         1.000000*np.tanh(np.minimum( ((np.maximum( (np.cos(np.minimum( ((1.0/(1.0 + np.exp(- data["freq6_3"])))),  (data["freq10_0"])))),  ((1.0/(1.0 + np.exp(- (data["skw_4"] + (data["freq4_10"] + np.tanh(0.301030)))))))) / 2.0)),  ((data["freq10_0"] * (0.395062 * np.minimum( ((data["skw_4"] + (np.minimum( (data["freq7_2"]),  (data["freq7_2"])) + data["freq4_10"]))),  (np.minimum( ((data["skw_4"] + ((data["freq9_6"] >= 0.0).astype(float)))),  (data["freq6_3"]))))))))) +
         1.000000*np.tanh(np.minimum( ((((-((np.tanh(data["skw_7"]) - np.minimum( (((data["kur_4"] < (data["skw_15"] - (-(0.367879)))).astype(float))),  ((data["skw_7"] - (-(0.684932)))))))) < np.minimum( (data["freq3_3"]),  (np.sin(((np.maximum( (np.sin(data["avg_12"])),  (np.sin(data["freq2_8"]))) < 0.684932).astype(float)))))).astype(float))),  (((np.minimum( (data["kur_13"]),  (data["kur_13"])) >= 0.0).astype(float))))))

    return Outputs(p)


def GP8(data):
    p = (1.000000*np.tanh((((((data["skw_14"] + (-(np.maximum( ((data["skw_14"] + ((data["skw_14"] + (-(((((1.0/(1.0 + np.exp(- np.maximum( (0.577216),  (data["skw_14"]))))) * 2.0) * 2.0) * 2.0)))) * 2.0))),  (((((1.0/(1.0 + np.exp(- (np.cos(data["kur_7"]) + data["freq7_4"])))) * 2.0) * 2.0) * 2.0)))))) * 2.0) * 2.0) * 2.0) * 2.0)) +
         1.000000*np.tanh((np.maximum( (((np.maximum( (data["kur_8"]),  (np.maximum( (data["kur_8"]),  (((1.0/(1.0 + np.exp(- ((data["freq6_5"] < np.minimum( ((9.0)),  (np.tanh(np.maximum( (0.064516),  (data["kur_12"])))))).astype(float))))) / 2.0))))) * 2.0) + np.maximum( ((data["freq8_2"] + data["kur_14"])),  (np.maximum( (data["skw_14"]),  (data["kur_12"])))))),  (((data["freq2_0"] - np.tanh(np.cos(data["kur_14"]))) * 2.0))) - (9.0))) +
         1.000000*np.tanh(((-((((np.maximum( (data["freq1_9"]),  ((np.cos(data["freq5_0"]) / 2.0))) * 2.0) + (np.sin(data["freq3_15"]) + np.cos((data["freq7_11"] + np.maximum( (np.maximum( (data["freq6_1"]),  (data["freq7_11"]))),  (((np.maximum( (data["freq1_9"]),  (data["freq5_5"])) >= 0.0).astype(float)))))))) + (np.maximum( (data["freq1_9"]),  (np.maximum( (data["freq1_9"]),  (data["freq5_5"])))) * 2.0)))) / 2.0)) +
         1.000000*np.tanh(np.tanh(np.sin(((data["kur_1"] + data["kur_11"]) + np.minimum( (np.sin((data["kur_10"] * 2.0))),  ((np.maximum( (data["kur_11"]),  (np.sin(data["freq4_9"]))) - (data["skw_10"] + ((1.0/(1.0 + np.exp(- (((data["skw_10"] + np.sin((data["skw_10"] + np.sin((data["kur_1"] + data["kur_11"]))))) + data["kur_11"]) + np.sin(0.577216))))) * data["skw_0"]))))))))) +
         1.000000*np.tanh((-((data["freq9_12"] * ((np.minimum( (data["freq4_15"]),  (data["freq9_7"])) > np.sin((-(((data["freq9_7"] < (-(((data["kur_4"] < (-(((data["freq1_4"] < (-(((data["freq1_4"] < (data["freq10_12"] * np.tanh((((1.0/(1.0 + np.exp(- data["freq4_15"]))) < np.cos(np.tanh(data["freq2_14"]))).astype(float))))).astype(float))))).astype(float))))).astype(float))))).astype(float)))))).astype(float)))))) +
         1.000000*np.tanh((data["freq4_13"] * (1.595240 * np.minimum( (((data["freq9_10"] < ((1.202057 * (0.777778 + data["freq4_4"])) * 2.0)).astype(float))),  (((data["freq1_9"] > np.sin(((np.maximum( (data["freq6_6"]),  (data["freq10_15"])) < np.maximum( (data["freq9_5"]),  (np.maximum( (data["freq9_5"]),  (data["freq9_1"]))))).astype(float)))).astype(float))))))) +
         1.000000*np.tanh((data["freq7_7"] * (1.0/(1.0 + np.exp(- (-(((((np.tanh((data["kur_3"] * (1.0/(1.0 + np.exp(- data["skw_12"]))))) < ((((((data["kur_0"] * ((data["freq8_3"] > data["freq2_12"]).astype(float))) >= 0.0).astype(float)) * ((((data["freq6_11"] < ((np.cos((((2.302585 * np.sin(1.414214)) >= 0.0).astype(float))) >= 0.0).astype(float))).astype(float)) >= 0.0).astype(float))) / 2.0) * data["freq4_11"])).astype(float)) * 2.0) * 2.0)))))))) +
         1.000000*np.tanh(((np.minimum( (np.cos(data["kur_5"])),  ((data["kur_0"] * 2.0))) * 2.0) * np.minimum( (((data["freq3_14"] > (-((np.minimum( ((((-(((data["freq2_15"] < (data["kur_0"] / 2.0)).astype(float)))) / 2.0) + np.cos(np.cos(data["kur_0"])))),  (((data["freq3_14"] < data["kur_0"]).astype(float)))) + data["freq6_11"])))).astype(float))),  (((data["freq2_15"] < np.cos(data["kur_0"])).astype(float)))))) +
         1.000000*np.tanh(((data["skw_13"] * ((0.577216 > (((data["freq2_3"] - ((np.minimum( (data["freq9_8"]),  (np.cos(np.cos((1.0/(1.0 + np.exp(- data["freq7_7"]))))))) < 0.258065).astype(float))) * 2.0) / 2.0)).astype(float))) * ((data["freq7_7"] > ((data["freq2_3"] - (-(((data["freq5_12"] < (((data["freq7_7"] > data["freq5_5"]).astype(float)) + ((np.minimum( (data["freq9_8"]),  (data["freq7_15"])) < 0.258065).astype(float)))).astype(float))))) / 2.0)).astype(float)))) +
         1.000000*np.tanh((((data["freq10_6"] > (-(((data["freq4_7"] >= 0.0).astype(float))))).astype(float)) * (0.338028 - np.cos((0.777778 * ((0.777778 * (data["skw_4"] - (data["freq6_3"] * ((((data["skw_4"] > 0.338028).astype(float)) > data["kur_8"]).astype(float))))) - (data["freq6_3"] * ((data["freq3_0"] > data["kur_8"]).astype(float))))))))) +
         1.000000*np.tanh((np.sin((np.minimum( ((((0.367879 < data["skw_6"]).astype(float)) - (1.0/(1.0 + np.exp(- np.sin(((data["freq9_13"] > (((data["freq7_13"] - data["freq8_13"]) < data["freq3_7"]).astype(float))).astype(float)))))))),  (np.minimum( (np.minimum( (data["skw_6"]),  (((data["skw_6"] > np.cos((np.cos(data["freq7_2"]) + data["skw_6"]))).astype(float))))),  ((((data["skw_12"] < data["skw_6"]).astype(float)) - data["freq8_13"]))))) * 2.0)) / 2.0)) +
         1.000000*np.tanh(((np.minimum( (np.minimum( (data["kur_13"]),  (np.sin(np.minimum( (((data["freq10_0"] < data["avg_9"]).astype(float))),  ((((data["freq10_0"] < data["avg_9"]).astype(float)) - (((data["freq6_14"] * np.cos((data["freq9_5"] / 2.0))) >= 0.0).astype(float))))))))),  (np.minimum( ((data["freq6_14"] + ((data["freq6_14"] + ((((data["freq6_2"] >= 0.0).astype(float)) >= 0.0).astype(float))) / 2.0))),  (np.cos(data["freq3_2"]))))) >= 0.0).astype(float))) +
         0.993400*np.tanh(((data["kur_0"] * 2.0) * ((((np.sin(np.sin((data["freq9_15"] - data["freq5_8"]))) - ((data["avg_2"] > np.maximum( (data["kur_5"]),  (data["freq6_1"]))).astype(float))) >= 0.0).astype(float)) - np.cos(data["freq3_6"])))) +
         1.000000*np.tanh((-(np.maximum( (np.tanh(np.minimum( (0.338028),  (np.tanh((data["avg_8"] * np.cos((np.sin((data["skw_9"] * 2.0)) * 2.0)))))))),  (np.maximum( ((data["skw_2"] - np.minimum( (np.cos(((data["freq7_13"] - np.minimum( (data["skw_12"]),  (0.338028))) - np.minimum( (data["skw_2"]),  (0.258065))))),  (0.733333)))),  (np.minimum( (np.tanh((data["skw_11"] * 2.0))),  (0.338028))))))))) +
         1.000000*np.tanh((data["skw_3"] * ((data["freq2_12"] > np.minimum( ((((np.minimum( (2.703700),  (((data["avg_5"] - data["freq6_0"]) - data["freq6_0"]))) * 2.0) * 2.0) * data["avg_5"])),  ((((-((data["freq6_0"] * data["avg_12"]))) + ((data["kur_9"] + data["freq10_6"]) * data["freq8_6"])) + ((-((data["freq8_6"] + data["freq6_0"]))) + data["avg_5"]))))).astype(float)))) +
         1.000000*np.tanh(((data["freq7_2"] * ((data["kur_7"] < (-((1.0/(1.0 + np.exp(- ((data["skw_1"] > np.maximum( (data["freq5_1"]),  ((-(np.sin((data["freq7_2"] * np.cos(np.minimum( (data["skw_1"]),  (data["freq7_2"])))))))))).astype(float)))))))).astype(float))) * ((((data["kur_7"] < (-((1.0/(1.0 + np.exp(- ((data["skw_1"] > (np.cos(data["skw_1"]) / 2.0)).astype(float)))))))).astype(float)) * 2.0) * 2.0))))
    return Outputs(p)


def GP9(data):
    p = (1.000000*np.tanh(np.minimum( ((data["kur_14"] + ((data["freq1_14"] * data["freq1_14"]) - ((data["kur_14"] - data["freq10_8"]) - 9.869604)))),  (((np.minimum( ((data["kur_14"] + (data["skw_14"] - 9.869604))),  (np.cos(((data["skw_14"] - data["kur_14"]) + data["skw_14"])))) * 2.0) - np.maximum( ((data["skw_14"] - 9.869604)),  ((data["freq1_14"] * np.sin(data["freq7_10"])))))))) +
         1.000000*np.tanh((data["skw_14"] - ((4.187500 + (data["freq7_10"] + ((data["freq6_11"] < data["freq7_14"]).astype(float)))) + (data["freq3_9"] - (0.716667 * (data["kur_10"] - (4.187500 + ((data["freq3_12"] + ((np.minimum( (data["freq7_2"]),  ((data["freq1_3"] + np.cos(4.187500)))) < data["freq7_10"]).astype(float))) + ((data["freq6_11"] < data["freq7_14"]).astype(float)))))))))) +
         1.000000*np.tanh((data["skw_6"] - np.maximum( (data["skw_12"]),  (((data["freq2_0"] + ((1.0/(1.0 + np.exp(- ((data["freq2_0"] + data["freq2_0"]) + data["skw_10"])))) + (((np.minimum( (data["skw_4"]),  (np.tanh(data["freq2_0"]))) / 2.0) + (data["freq3_12"] + ((((data["freq10_5"] >= 0.0).astype(float)) > data["kur_14"]).astype(float)))) + (-(data["freq8_15"]))))) + data["skw_10"]))))) +
         1.000000*np.tanh(np.minimum( (data["kur_11"]),  ((((data["kur_11"] * np.maximum( (data["freq4_15"]),  (np.maximum( (data["avg_5"]),  ((data["avg_13"] * np.maximum( (data["freq4_15"]),  (np.maximum( (data["avg_5"]),  (np.maximum( (data["freq4_15"]),  (np.maximum( (data["freq4_15"]),  ((data["avg_13"] * (np.minimum( (((data["freq2_4"] < (data["freq7_7"] * 2.0)).astype(float))),  ((-(np.cos(data["kur_11"]))))) / 2.0)))))))))))))))) * 2.0) * 2.0)))) +
         1.000000*np.tanh(np.maximum( (np.maximum( (np.maximum( (data["kur_0"]),  ((data["kur_3"] * np.maximum( (data["freq7_14"]),  ((1.0/(1.0 + np.exp(- (data["freq5_0"] * 2.0)))))))))),  ((-(np.cos(((data["freq5_0"] + (data["kur_1"] * np.maximum( (((data["kur_1"] >= 0.0).astype(float))),  (data["freq5_0"])))) + (1.0/(1.0 + np.exp(- data["freq5_0"])))))))))),  ((-(np.cos((((data["freq9_14"] * 2.0) * data["kur_0"]) + 0.915966))))))) +
         1.000000*np.tanh(np.tanh((-((((data["skw_12"] * (2.0 * data["freq8_5"])) > (data["freq7_7"] + (np.cos((np.maximum( (np.maximum( (data["freq1_3"]),  (((data["skw_2"] > (2.0 + data["freq1_3"])).astype(float))))),  (((np.minimum( (data["freq3_12"]),  (data["freq8_5"])) > (1.0/(1.0 + np.exp(- np.tanh(data["avg_12"]))))).astype(float)))) * 2.0)) * 2.0))).astype(float)))))) +
         1.000000*np.tanh((data["freq7_10"] * ((np.tanh((data["freq3_8"] - np.maximum( (np.sin(np.sin(data["freq9_9"]))),  (np.maximum( (np.tanh((np.maximum( (data["freq9_9"]),  (np.maximum( (data["freq2_8"]),  (data["freq2_8"])))) / 2.0))),  (np.maximum( (data["freq2_8"]),  (np.maximum( ((np.tanh(data["freq4_8"]) / 2.0)),  (np.maximum( (np.tanh((data["freq3_14"] / 2.0))),  (np.maximum( (data["freq2_8"]),  (data["skw_6"])))))))))))))) >= 0.0).astype(float)))) +
         1.000000*np.tanh((((((data["skw_15"] < (-3.0 + np.minimum( (((data["freq3_1"] < data["freq2_8"]).astype(float))),  (np.sin((np.maximum( (-3.0),  ((data["skw_15"] - (((-(np.maximum( (np.maximum( ((data["freq2_1"] + ((data["freq9_13"] > data["freq10_9"]).astype(float)))),  (np.minimum( (data["freq10_9"]),  (data["freq8_4"]))))),  (np.minimum( (data["freq2_0"]),  ((1.0/(1.0 + np.exp(- np.sin(data["freq3_4"])))))))))) >= 0.0).astype(float))))) * 2.0)))))).astype(float)) * 2.0) * 2.0) * 2.0)) +
         1.000000*np.tanh(np.minimum( (((np.maximum( (data["skw_12"]),  ((float((2.685452 / 2.0) >= 0.0)))) > data["freq10_9"]).astype(float))),  (((np.minimum( (data["kur_13"]),  (np.minimum( (np.minimum( (data["kur_13"]),  ((((data["skw_12"] > np.cos((data["freq9_11"] - np.cos((data["skw_9"] * 2.0))))).astype(float)) * data["freq6_4"])))),  ((((data["skw_12"] > ((float(2.685452 > 0.915966)) / 2.0)).astype(float)) * data["skw_10"]))))) >= 0.0).astype(float))))) +
         1.000000*np.tanh((data["freq4_9"] * ((data["freq10_4"] > (np.cos(((data["freq7_7"] < data["freq6_0"]).astype(float))) + np.cos((data["freq5_9"] * ((data["freq6_4"] > (((data["freq7_7"] < (((data["freq4_9"] * np.tanh(((data["freq6_4"] > (np.cos(np.cos(np.sin(((data["freq6_0"] > data["freq6_4"]).astype(float))))) + np.cos(data["freq6_0"]))).astype(float)))) >= 0.0).astype(float))).astype(float)) + np.cos(data["freq9_4"]))).astype(float)))))).astype(float)))) +
         1.000000*np.tanh((np.sin((np.maximum( (data["freq3_13"]),  ((1.0/(1.0 + np.exp(- (((data["freq3_13"] > data["avg_13"]).astype(float)) + (data["kur_1"] / 2.0))))))) + (data["kur_1"] - (((data["freq7_5"] + ((np.tanh((-(((data["freq3_13"] > data["avg_13"]).astype(float))))) >= 0.0).astype(float))) >= 0.0).astype(float))))) * ((data["freq1_9"] > np.maximum( (np.tanh((((data["freq7_5"] >= 0.0).astype(float)) / 2.0))),  (np.tanh(data["kur_1"])))).astype(float)))) +
         1.000000*np.tanh((-(((((1.0/(1.0 + np.exp(- data["avg_2"]))) * 2.0) < (data["freq3_15"] - (data["freq4_11"] - ((((data["freq4_11"] > data["freq2_15"]).astype(float)) < (((data["freq7_8"] < (data["avg_2"] - (1.0/(1.0 + np.exp(- ((data["avg_2"] > (data["freq6_13"] + np.maximum( (data["freq2_11"]),  (data["skw_12"])))).astype(float))))))).astype(float)) / 2.0)).astype(float))))).astype(float))))) +
         1.000000*np.tanh(np.maximum( (np.tanh((((1.202057 * (data["skw_14"] / 2.0)) > np.maximum( (((data["skw_1"] > ((data["skw_14"] > ((data["freq8_11"] >= 0.0).astype(float))).astype(float))).astype(float))),  (((data["avg_15"] >= 0.0).astype(float))))).astype(float)))),  (((((data["freq5_9"] > np.sin(data["freq1_6"])).astype(float)) * data["freq7_7"]) - (np.maximum( (data["freq8_9"]),  (data["skw_1"])) + np.tanh((((data["freq6_1"] * 2.0) >= 0.0).astype(float)))))))) +
         1.000000*np.tanh((data["freq1_1"] * ((((data["skw_2"] < ((data["kur_6"] < (((-(data["kur_4"])) > (((-(data["kur_9"])) < data["freq5_12"]).astype(float))).astype(float))).astype(float))).astype(float)) < ((data["freq10_3"] < ((-((((((data["freq5_12"] * np.tanh((((data["skw_4"] / 2.0) > data["freq1_1"]).astype(float)))) + (data["skw_9"] - data["avg_4"])) < (data["skw_4"] / 2.0)).astype(float)) / 2.0))) / 2.0)).astype(float))).astype(float)))) +
         1.000000*np.tanh(((((((data["kur_0"] * ((data["freq9_15"] > (data["kur_5"] * data["freq5_8"])).astype(float))) * 2.0) * ((data["freq9_15"] > (-2.0 * ((data["freq9_15"] > data["freq5_8"]).astype(float)))).astype(float))) * 2.0) * ((data["freq9_15"] > (data["freq7_12"] * np.sin((-2.0 * 2.0)))).astype(float))) * ((data["freq3_6"] > (data["kur_5"] * -2.0)).astype(float)))) +
         1.000000*np.tanh(((np.minimum( (0.301030),  (data["kur_2"])) * 2.0) * (data["skw_2"] + np.minimum( ((-((data["skw_0"] * np.maximum( (data["freq3_10"]),  ((((data["skw_10"] * data["freq8_10"]) >= 0.0).astype(float)))))))),  ((np.maximum( (data["freq2_2"]),  ((-(data["skw_10"])))) - data["skw_10"])))))))

    return Outputs(p)


def GP10(data):
    p = (1.000000*np.tanh((data["freq7_2"] - (data["freq10_2"] + np.maximum( (np.maximum( ((data["freq10_2"] + np.maximum( (np.maximum( (((9.0) - data["kur_14"])),  (np.sin(((data["freq7_0"] > 0.181818).astype(float)))))),  (np.maximum( (((9.0) - data["kur_14"])),  (0.181818)))))),  (np.sin(((data["freq7_0"] > 0.181818).astype(float)))))),  ((np.sin(np.minimum( (data["freq7_0"]),  (data["freq7_0"]))) - data["skw_8"])))))) +
         1.000000*np.tanh((np.sin(data["skw_5"]) - (4.363640 + (data["skw_9"] + (((np.minimum( (data["skw_5"]),  ((data["freq3_13"] - np.tanh((data["freq8_2"] + (((((float(4.363640 >= 0.0)) * 2.0) * (np.tanh(4.363640) / 2.0)) < 4.363640).astype(float))))))) + (data["freq3_13"] - np.tanh((data["freq8_2"] + np.sin(data["skw_5"]))))) / 2.0) + data["freq10_3"]))))) +
         1.000000*np.tanh((-((data["freq8_7"] + ((((((-(((np.sin(((data["kur_12"] >= 0.0).astype(float))) < (((np.sin(((data["kur_12"] >= 0.0).astype(float))) < (data["freq3_15"] + data["freq3_15"])).astype(float)) + data["skw_10"])).astype(float)))) >= 0.0).astype(float)) < np.maximum( (data["freq1_9"]),  (((data["skw_6"] < (1.0/(1.0 + np.exp(- data["freq1_9"])))).astype(float))))).astype(float)) - (data["freq8_7"] - ((data["skw_6"] < (1.0/(1.0 + np.exp(- data["freq1_9"])))).astype(float)))))))) +
         1.000000*np.tanh(np.minimum( (np.tanh(np.tanh((1.0/(1.0 + np.exp(- data["freq4_11"])))))),  ((data["kur_10"] * np.maximum( ((np.maximum( ((data["freq8_11"] + ((np.tanh(data["freq3_5"]) + ((data["freq3_5"] + data["skw_9"]) * 2.0)) * 2.0))),  ((1.375000 + (((data["freq2_0"] * 2.0) + ((data["freq2_0"] * 2.0) + 0.693147)) * 2.0)))) + data["freq3_5"])),  (data["kur_10"])))))) +
         1.000000*np.tanh(((data["kur_1"] > (((np.cos((((data["freq7_14"] > (((-(((data["freq10_4"] < np.sin(np.minimum( ((((data["kur_1"] >= 0.0).astype(float)) - data["freq9_15"])),  (data["freq1_8"])))).astype(float)))) >= 0.0).astype(float))).astype(float)) * 2.0)) * np.maximum( (np.minimum( (data["kur_1"]),  ((-(((data["kur_1"] > ((np.tanh(data["freq9_12"]) >= 0.0).astype(float))).astype(float))))))),  ((-(data["kur_13"]))))) >= 0.0).astype(float))).astype(float))) +
         1.000000*np.tanh((-(((np.maximum( (data["freq5_10"]),  ((np.tanh(np.tanh((data["freq3_15"] - (((data["freq4_10"] - ((data["freq5_10"] < (data["kur_2"] + (((data["freq8_8"] < (-(data["freq4_10"]))).astype(float)) / 2.0))).astype(float))) < data["freq1_2"]).astype(float))))) * 2.0))) < (data["freq3_15"] - (1.0/(1.0 + np.exp(- data["kur_2"]))))).astype(float))))) +
         1.000000*np.tanh((-((data["avg_8"] * (data["freq9_13"] * ((np.maximum( (data["freq9_7"]),  (data["freq9_8"])) > ((data["freq6_5"] > ((np.maximum( (0.159091),  (data["freq9_13"])) > data["freq5_9"]).astype(float))).astype(float))).astype(float))))))) +
         1.000000*np.tanh(((((-(((data["skw_2"] > ((data["freq1_6"] < np.maximum( ((data["freq4_13"] + (-(np.minimum( (data["freq5_12"]),  (data["freq9_12"])))))),  ((data["freq4_13"] + np.maximum( (data["freq3_0"]),  (((data["kur_13"] > (((((data["freq9_12"] < 0.159091).astype(float)) < (-(np.minimum( (np.sin(data["freq4_1"])),  (data["freq4_13"]))))).astype(float)) * 2.0)).astype(float)))))))).astype(float))).astype(float)))) * 2.0) * 2.0) * 2.0)) +
         1.000000*np.tanh(((1.414214 < ((np.tanh(data["kur_0"]) * np.maximum( (data["skw_9"]),  (((np.maximum( (data["skw_14"]),  (np.maximum( (np.maximum( ((data["freq2_6"] * 2.0)),  ((((((np.sin(np.sin(np.sin(data["skw_14"]))) > (((((data["freq2_6"] > ((data["skw_9"] < data["skw_9"]).astype(float))).astype(float)) * 2.0) < data["skw_15"]).astype(float))).astype(float)) * 2.0) * 2.0) * 2.0)))),  ((data["freq2_6"] * 2.0))))) / 2.0) * 2.0)))) * 2.0)).astype(float))) +
         1.000000*np.tanh(((((data["kur_3"] * data["freq8_10"]) * np.maximum( (data["freq8_14"]),  ((-((1.0/(1.0 + np.exp(- (data["freq7_10"] + (-((data["kur_3"] * np.maximum( (data["freq8_14"]),  ((1.0/(1.0 + np.exp(- data["freq6_13"]))))))))))))))))) * np.maximum( (data["freq8_4"]),  ((data["freq6_13"] * np.minimum( (data["skw_11"]),  ((1.0/(1.0 + np.exp(- data["freq8_14"]))))))))) * np.maximum( (data["freq8_4"]),  (data["freq6_13"])))) +
         1.000000*np.tanh((((((data["skw_4"] >= 0.0).astype(float)) > (((((np.minimum( (data["freq2_12"]),  (data["avg_9"])) > data["kur_11"]).astype(float)) * np.minimum( (data["freq7_12"]),  (np.cos(np.maximum( (data["freq6_10"]),  (np.maximum( ((2.0 / 2.0)),  ((((data["freq2_12"] >= 0.0).astype(float)) - (((data["freq1_15"] / 2.0) >= 0.0).astype(float))))))))))) >= 0.0).astype(float))).astype(float)) * ((((data["kur_15"] * 2.0) * 2.0) * 2.0) * 2.0))) +
         1.000000*np.tanh((data["patient_id"] * ((data["skw_11"] < np.minimum( ((np.maximum( (data["freq7_5"]),  (np.minimum( (data["skw_0"]),  (np.maximum( (data["freq7_5"]),  (np.maximum( (data["freq7_5"]),  (np.maximum( (np.minimum( (data["skw_2"]),  (data["freq1_14"]))),  (np.minimum( ((data["skw_2"] * 2.0)),  (data["freq1_14"])))))))))))) - ((((data["freq3_8"] - data["skw_11"]) > np.maximum( (data["freq7_5"]),  (1.282427))).astype(float)) / 2.0))),  (np.minimum( ((1.0/(1.0 + np.exp(- data["freq2_3"])))),  (data["skw_0"]))))).astype(float)))) +
         1.000000*np.tanh((data["freq7_5"] * (((np.sin(np.maximum( (np.maximum( (data["freq8_13"]),  (1.732051))),  (data["freq5_8"]))) / 2.0) < ((data["freq5_8"] * ((data["freq3_9"] > (((np.sin(data["freq9_7"]) > np.tanh(((np.maximum( ((3.88284182548522949)),  (data["freq3_0"])) >= 0.0).astype(float)))).astype(float)) * data["freq3_9"])).astype(float))) / 2.0)).astype(float)))) +
         1.000000*np.tanh(((np.minimum( (data["freq2_9"]),  (np.minimum( (data["skw_11"]),  ((-(((data["freq3_13"] < np.maximum( (data["freq3_5"]),  (((((data["freq3_13"] > np.maximum( (data["freq9_10"]),  (np.maximum( (data["freq5_15"]),  (data["file_size"]))))).astype(float)) + (4.090910 * np.maximum( (data["freq9_10"]),  (np.maximum( (data["freq9_10"]),  (np.maximum( (data["freq9_10"]),  (np.maximum( (data["freq5_15"]),  (data["file_size"])))))))))) / 2.0)))).astype(float)))))))) >= 0.0).astype(float))) +
         1.000000*np.tanh((3.809520 * np.tanh(((-((((((1.0/(1.0 + np.exp(- data["freq6_10"]))) < np.cos(((((((data["freq8_14"] < (float(3.809520 >= 0.0))).astype(float)) < data["freq3_9"]).astype(float)) < (((1.0/(1.0 + np.exp(- data["freq9_12"]))) >= 0.0).astype(float))).astype(float)))).astype(float)) < (data["skw_1"] - (1.0/(1.0 + np.exp(- data["freq3_9"]))))).astype(float)))) * 2.0)))) +
         1.000000*np.tanh((data["kur_0"] * ((data["avg_7"] * ((data["freq1_8"] < ((math.sin(1.021740) < np.sin(data["avg_7"])).astype(float))).astype(float))) * ((((data["freq1_11"] + np.cos(((-1.0 < (data["kur_0"] + np.cos(((-1.0 < ((31.006277 / 2.0) + data["kur_0"])).astype(float))))).astype(float)))) / 2.0) < np.cos(np.maximum( (data["freq9_4"]),  (((data["freq1_4"] > -1.0).astype(float)))))).astype(float))))))

    return Outputs(p)


def GP(data):
    return (GP1(data) +
            GP2(data) +
            GP3(data) +
            GP4(data) +
            GP5(data) +
            GP6(data) +
            GP7(data) +
            GP8(data) +
            GP9(data) +
            GP10(data))/10.0



if __name__ == '__main__':
    if 1:
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
    origtrain, origtest, features = read_test_train()
    print('Length of train: ', len(origtrain))
    print('Length of test: ', len(origtest))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    test = origtest.copy()
    test['result'] = -1
    ss = StandardScaler()
    x = pd.concat([origtrain.copy(), test.copy()])
    x.file_size = np.log(x.file_size)
    x.fillna(0, inplace=True)
    x[features] = ss.fit_transform(x[features])
    train = x[x.result != -1].copy()
    test = x[x.result == -1].copy()
    test.drop(['result'], inplace=True, axis=1)
    score = roc_auc_score(train.result, GP(train))
    print('ROC:', score)
    predictions = GP(test)
    create_submission(score, origtest, predictions.values)
