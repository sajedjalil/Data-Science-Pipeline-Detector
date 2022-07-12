__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import math
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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


def Outputs(data):
    return 1./(1.+np.exp(-data))


def GP(data):
    predictions = (np.tanh(((data["avg_2"] * ((((1.0 - data["file_size"]) * 2.0) >= 0.0).astype(float))) - (9.869604 - (((np.cos(((-(data["avg_2"])) * 2.125000)) * data["avg_2"]) - (-(data["avg_2"]))) * ((((((((data["avg_3"] - (1.0 - data["file_size"])) * 2.0) >= 0.0).astype(float)) - data["file_size"]) * 2.0) >= 0.0).astype(float)))))) +
                   np.tanh((data["avg_3"] - (((6.0) + ((data["patient_id"] + (((data["avg_8"] < ((data["patient_id"] + (((1.0/(1.0 + np.exp(- ((data["patient_id"] + (((6.0) < data["avg_15"]).astype(float))) + data["patient_id"])))) < 3.141593).astype(float))) + (((0.909091 + data["avg_10"]) >= 0.0).astype(float)))).astype(float)) * 2.0)) + data["avg_15"]))/2.0))) +
                   np.tanh((data["avg_9"] - (data["avg_5"] + (1.0/(1.0 + np.exp(- ((((data["avg_9"] > (((1.0/(1.0 + np.exp(- ((((data["avg_5"] * (((data["avg_9"] - ((0.944444 + data["avg_11"])/2.0)) > data["avg_9"]).astype(float))) >= ((data["avg_15"] > 0.129032).astype(float))).astype(float)) * 2.0)))) + (data["avg_11"] * np.maximum( (data["avg_2"]),  (((data["avg_15"] > 0.129032).astype(float))))))/2.0)).astype(float)) * 2.0) * 2.0))))))) +
                   np.tanh(((-(((data["avg_12"] > ((1.0/(1.0 + np.exp(- 0.405405))) + (data["avg_11"] + 0.925926))).astype(float)))) - (((-(((data["avg_12"] > ((1.0/(1.0 + np.exp(- 0.405405))) + np.maximum( (0.405405),  ((data["avg_14"] + np.tanh(np.cos(np.cos(data["avg_11"])))))))).astype(float)))) < ((data["avg_5"] >= (0.324324 + ((0.367879 > data["avg_14"]).astype(float)))).astype(float))).astype(float)))) +
                   np.tanh((((np.cos((((data["patient_id"] + data["avg_15"]) * data["avg_0"]) * data["avg_0"])) <= ((np.cos(data["avg_2"]) <= np.sin((((1.0/(1.0 + np.exp(- data["avg_0"]))) <= ((np.cos(data["avg_2"]) <= ((np.sin((((data["patient_id"] + data["avg_2"]) >= 0.0).astype(float))) >= (3.0 * (1.0/(1.0 + np.exp(- ((data["patient_id"] + data["avg_15"]) * data["avg_0"])))))).astype(float))).astype(float))).astype(float)))).astype(float))).astype(float)) * 2.0)) +
                   np.tanh((((-(data["avg_13"])) >= ((((2.466670 + np.maximum( ((data["avg_7"] * 0.129032)),  (np.maximum( ((data["avg_7"] * (data["avg_7"] * ((data["avg_3"] < (data["avg_7"] * ((data["avg_3"] * 2.0) * ((0.367879 >= data["file_size"]).astype(float))))).astype(float))))),  (0.129032)))))/2.0) + (float((float(0.543210 >= 0.129032)) >= 0.0)))/2.0)).astype(float))) +
                   np.tanh((np.minimum( (np.minimum( (np.maximum( (data["avg_0"]),  (1.732051))),  ((np.minimum( (((data["avg_11"] + ((data["avg_1"] >= (0.543210 / 2.0)).astype(float)))/2.0)),  ((((1.570796 >= data["avg_5"]).astype(float)) - np.tanh(2.125000)))) * 2.0)))),  (np.minimum( (((data["avg_11"] + ((data["avg_1"] >= (data["avg_12"] + 0.543210)).astype(float)))/2.0)),  ((-((((data["avg_11"] + 1.732051) <= data["avg_12"]).astype(float)))))))) * 2.0)) +
                   np.tanh((((data["avg_1"] > ((2.421050 - (((data["avg_1"] + (-(np.cos(np.tanh(((data["avg_1"] <= ((data["avg_1"] > (((data["avg_1"] + np.cos(np.tanh(data["avg_4"])))/2.0) - ((data["avg_4"] + (1.0/(1.0 + np.exp(- 1.570796))))/2.0))).astype(float))).astype(float))))))) > ((data["avg_12"] / 2.0) / 2.0)).astype(float))) - data["avg_1"])).astype(float)) * ((data["avg_7"] <= np.cos(-2.0)).astype(float)))) +
                   np.tanh((((data["avg_8"] * (1.0/(1.0 + np.exp(- data["avg_15"])))) > (((((1.0/(1.0 + np.exp(- (1.0/(1.0 + np.exp(- (1.0/(1.0 + np.exp(- np.tanh(data["file_size"])))))))))) >= 0.0).astype(float)) > np.minimum( (np.minimum( ((((data["avg_5"] * data["avg_3"]) > np.sin(np.sin((((-((1.0/(1.0 + np.exp(- np.tanh(data["file_size"])))))) + ((0.909091 > data["avg_3"]).astype(float)))/2.0)))).astype(float))),  (data["avg_8"]))),  (data["avg_8"]))).astype(float))).astype(float))) +
                   np.tanh((((-((((-(((data["avg_11"] <= (-((((0.318310 - math.tanh(math.tanh(2.125000))) > np.tanh(data["avg_6"])).astype(float))))).astype(float)))) <= (-(np.tanh((data["avg_6"] + math.sin(0.944444)))))).astype(float)))) + (-(((data["avg_9"] / 2.0) / 2.0))))/2.0)) +
                   np.tanh((((np.cos((((1.732051 + data["avg_15"]) * 0.041096) * (data["avg_0"] - ((((2.466670 < data["avg_1"]).astype(float)) <= (data["avg_15"] * 2.0)).astype(float))))) <= np.cos((data["avg_4"] - ((np.cos(data["avg_1"]) < 0.636620).astype(float))))).astype(float)) * (data["avg_1"] + data["avg_11"]))) +
                   np.tanh((((0.416667 <= data["avg_4"]).astype(float)) * (np.minimum( ((((-(data["avg_13"])) + data["avg_1"]) + 0.509804)),  ((((np.maximum( ((((((1.0/(1.0 + np.exp(- data["avg_13"]))) <= data["avg_3"]).astype(float)) * ((0.416667 <= data["avg_3"]).astype(float))) + 0.416667)),  (data["avg_10"])) >= data["avg_7"]).astype(float)) + ((-(data["avg_13"])) + data["avg_1"])))) * 1.903850))) +
                   np.tanh(np.minimum( (((np.maximum( ((((data["file_size"] < 0.543210).astype(float)) / 2.0)),  (((data["avg_5"] <= ((data["avg_9"] > data["avg_0"]).astype(float))).astype(float)))) > ((data["avg_5"] <= ((data["avg_9"] > (0.324324 * 2.0)).astype(float))).astype(float))).astype(float))),  (((data["avg_0"] <= (-(((np.maximum( ((1.0/(1.0 + np.exp(- 1.903850)))),  (np.sin(data["avg_14"]))) >= np.sin(np.maximum( ((0.324324 / 2.0)),  (data["avg_1"])))).astype(float))))).astype(float))))) +
                   np.tanh(((data["avg_3"] >= ((np.cos(12.0) + np.tanh(((-3.0 <= (1.0/(1.0 + np.exp(- 5.750000)))).astype(float)))) * (((((np.maximum( (data["avg_3"]),  (data["avg_5"])) <= np.tanh(np.tanh(np.cos((-((1.0/(1.0 + np.exp(- data["avg_7"]))))))))).astype(float)) >= ((data["avg_7"] <= data["avg_8"]).astype(float))).astype(float)) + (((5.750000 + -3.0)/2.0) + (-(data["avg_8"])))))).astype(float))) +
                   np.tanh((-(((data["file_size"] >= np.maximum( (np.maximum( (((data["avg_4"] > ((data["avg_14"] <= (((((data["patient_id"] + np.tanh((data["avg_10"] * data["avg_4"]))) >= 1.903850).astype(float)) + 2.466670)/2.0)).astype(float))).astype(float))),  (((1.903850 + data["patient_id"])/2.0)))),  (((((data["avg_10"] - (((data["avg_11"] < data["file_size"]).astype(float)) * 2.0)) * 2.0) + data["patient_id"])/2.0)))).astype(float))))) +
                   np.tanh((0.900000 - np.tanh((((((data["avg_15"] >= ((1.570796 - data["avg_1"]) + ((np.maximum( (data["avg_14"]),  (data["avg_13"])) / 2.0) / 2.0))).astype(float)) <= (1.0/(1.0 + np.exp(- np.maximum( (data["avg_15"]),  (((np.maximum( (data["avg_13"]),  (data["avg_15"])) / 2.0) / 2.0))))))).astype(float)) - np.minimum( (np.minimum( (0.318310),  ((data["avg_1"] - 0.324324)))),  ((data["avg_3"] - 0.318310))))))) +
                   np.tanh((-((((0.636620 + data["avg_0"]) <= (-(((((((data["avg_6"] + ((data["avg_8"] <= data["avg_6"]).astype(float))) >= data["avg_3"]).astype(float)) * 2.0) + (((0.925926 >= ((-1.0 >= data["avg_8"]).astype(float))).astype(float)) * np.sin(((data["avg_8"] <= data["avg_6"]).astype(float)))))/2.0)))).astype(float))))) +
                   np.tanh(((0.428571 <= (((0.324324 + data["avg_0"])/2.0) * (((data["avg_11"] >= (((-((1.0/(1.0 + math.exp(- (2.0 * 0.909091)))))) < data["avg_12"]).astype(float))).astype(float)) - ((((((data["avg_4"] < data["avg_11"]).astype(float)) < ((data["avg_4"] + 0.909091)/2.0)).astype(float)) < (((data["avg_0"] * 0.041096) < data["avg_12"]).astype(float))).astype(float))))).astype(float))) +
                   np.tanh((((((data["avg_3"] + (0.900000 * 2.0))/2.0) * (((data["avg_5"] * np.sin(((data["avg_3"] + ((1.0/(1.0 + np.exp(- 2.421050))) * 2.0))/2.0))) > np.sin(((((data["avg_3"] + (np.sin((1.0)) * 2.0))/2.0) >= 0.0).astype(float)))).astype(float))) > np.sin(((np.sin(((data["avg_3"] + data["avg_5"])/2.0)) >= 0.0).astype(float)))).astype(float))) +
                   np.tanh((-(((((((3.0 + (data["avg_12"] * 2.0)) <= ((0.405405 < (3.0 + (data["avg_12"] * 2.0))).astype(float))).astype(float)) + (((((data["avg_12"] >= ((3.0 < ((data["avg_12"] * 2.0) + (data["avg_12"] * 2.0))).astype(float))).astype(float)) * 2.0) < data["avg_12"]).astype(float))) > ((data["avg_14"] < np.minimum( (data["avg_12"]),  (data["avg_12"]))).astype(float))).astype(float))))) +
                   np.tanh((-(((5.0) * ((math.sin(-1.0) > (data["avg_9"] * np.minimum( (np.minimum( (0.367879),  ((data["avg_8"] + (data["avg_8"] + -1.0))))),  ((-(((0.367879 > (data["file_size"] * 2.0)).astype(float)))))))).astype(float)))))) +
                   np.tanh(((((((1.0/(1.0 + np.exp(- data["avg_5"]))) <= np.minimum( (data["avg_4"]),  (((data["avg_4"] < np.maximum( (0.416667),  (((np.sin(1.903850) + np.maximum( (0.075000),  (data["avg_5"])))/2.0)))).astype(float))))).astype(float)) / 2.0) * ((data["avg_10"] <= ((np.sin(1.903850) + np.maximum( (data["avg_5"]),  (data["avg_5"])))/2.0)).astype(float))) * 3.0)) +
                   np.tanh(((((((((np.minimum( (data["avg_0"]),  (data["avg_8"])) > ((data["patient_id"] < ((((data["patient_id"] < np.sin(data["file_size"])).astype(float)) < ((data["avg_0"] > np.minimum( (data["file_size"]),  (data["file_size"]))).astype(float))).astype(float))).astype(float))).astype(float)) >= (((np.sin((((1.0/(1.0 + np.exp(- ((data["avg_0"] >= data["avg_8"]).astype(float))))) >= data["avg_2"]).astype(float))) / 2.0) + ((data["file_size"] >= 0.0).astype(float)))/2.0)).astype(float)) > ((data["patient_id"] < data["file_size"]).astype(float))).astype(float)) * 2.0) * 2.0)) +
                   np.tanh(((((data["avg_8"] > (-(np.maximum( (data["avg_1"]),  (-1.0))))).astype(float)) - np.maximum( ((-((float(0.944444 < -1.0))))),  (np.maximum( (((np.minimum( (data["avg_13"]),  ((data["avg_6"] / 2.0))) < data["avg_7"]).astype(float))),  (np.maximum( (-3.0),  (((np.minimum( (data["avg_1"]),  (((0.636620 >= data["avg_2"]).astype(float)))) < np.maximum( ((data["avg_12"] + -1.0)),  (0.909091))).astype(float))))))))) / 2.0)) +
                   np.tanh((((((data["avg_1"] >= ((np.maximum( (data["avg_13"]),  (0.900000)) + (-(data["avg_15"]))) * 2.0)).astype(float)) >= ((((np.maximum( (data["avg_13"]),  (0.900000)) + 1.414214)/2.0) < (((data["avg_5"] < (((np.maximum( (data["avg_13"]),  (data["avg_13"])) <= np.cos(data["avg_13"])).astype(float)) * 2.0)).astype(float)) * 2.0)).astype(float))).astype(float)) / 2.0)) +
                   np.tanh((-(((0.909091 <= np.cos(np.cos(np.minimum( ((data["avg_4"] - ((np.minimum( (data["avg_4"]),  (data["avg_4"])) > np.minimum( (np.minimum( (np.minimum( (np.minimum( (np.minimum( (data["avg_12"]),  (np.minimum( (data["avg_12"]),  ((data["avg_0"] * data["avg_12"])))))),  (data["avg_13"]))),  (data["avg_4"]))),  ((-(0.324324))))),  ((-(((0.909091 <= np.cos(np.cos(data["avg_13"]))).astype(float))))))).astype(float)))),  (0.909091))))).astype(float))))) +
                   np.tanh((((((((data["avg_13"] >= data["avg_12"]).astype(float)) < data["avg_2"]).astype(float)) * ((((((0.509804 >= data["avg_12"]).astype(float)) * 2.0) * 2.0) * data["avg_0"]) * ((data["avg_12"] <= (((data["avg_8"] > data["avg_0"]).astype(float)) + ((data["avg_14"] <= ((0.509804 >= data["avg_0"]).astype(float))).astype(float)))).astype(float)))) / 2.0) * 2.0)) +
                   np.tanh((((((math.sin(2.421050) < (0.416667 * (np.maximum( ((data["avg_11"] * np.maximum( (data["avg_0"]),  ((data["avg_0"] * 2.0))))),  ((data["avg_0"] * 2.0))) * np.minimum( (((1.545450 < (2.421050 * data["avg_12"])).astype(float))),  (np.cos((data["avg_11"] * ((data["avg_2"] <= 2.421050).astype(float))))))))).astype(float)) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((-(((np.sin(data["avg_9"]) >= ((data["avg_9"] < ((0.367879 * (data["avg_8"] + (float((float(3.0 >= 0.0)) >= 0.0)))) + (((0.367879 * (data["avg_8"] + (float(3.0 >= 0.0)))) > ((np.sin(data["avg_9"]) > ((data["avg_0"] < (data["avg_8"] + ((np.cos((data["avg_8"] + ((data["avg_8"] >= 0.0).astype(float)))) >= 0.0).astype(float)))).astype(float))).astype(float))).astype(float)))).astype(float))).astype(float))))) +
                   np.tanh((-(((((data["avg_8"] >= ((0.075000 - 1.570796) + ((1.0/(1.0 + np.exp(- (-(data["file_size"]))))) / 2.0))).astype(float)) <= ((data["avg_6"] < (-((((-(((((data["avg_10"] + 2.125000)/2.0) <= ((data["avg_8"] < (-(((((data["avg_10"] + (2.125000 + ((data["avg_6"] + np.tanh(data["file_size"]))/2.0)))/2.0) >= 0.0).astype(float))))).astype(float))).astype(float)))) >= 0.0).astype(float))))).astype(float))).astype(float))))) +
                   np.tanh(np.tanh((-((data["avg_7"] * (((((((1.0/(1.0 + np.exp(- ((data["avg_1"] <= ((data["avg_4"] >= 0.0).astype(float))).astype(float))))) >= data["avg_1"]).astype(float)) >= (((((data["avg_4"] * 2.0) >= ((((-(((((data["avg_10"] <= 0.075000).astype(float)) <= data["avg_3"]).astype(float)))) >= np.minimum( (data["avg_7"]),  (((data["avg_9"] / 2.0) / 2.0)))).astype(float)) + (data["avg_9"] / 2.0))).astype(float)) + data["avg_15"])/2.0)).astype(float)) <= data["avg_1"]).astype(float))))))) +
                   np.tanh(((((1.414214 >= np.maximum( (((data["avg_11"] + 0.428571)/2.0)),  (data["avg_1"]))).astype(float)) - ((((float(1.414214 >= 0.0)) / 2.0) > np.minimum( (((np.cos(np.maximum( (data["avg_11"]),  (data["avg_1"]))) < data["avg_12"]).astype(float))),  (np.minimum( (((data["avg_11"] > (-(np.maximum( (0.944444),  (((data["avg_9"] + 0.318310)/2.0)))))).astype(float))),  (((data["avg_9"] >= 0.0).astype(float))))))).astype(float))) * 2.0)))

    return Outputs(predictions)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


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
    files = sorted(glob.glob("../input/test_" + str(patient_id) + "/*.mat"), key=natural_key)
    out = open("simple_test_" + str(patient_id) + ".csv", "w")
    out.write("Id,patient_id")
    for i in range(16):
        out.write(",avg_" + str(i))
    out.write(",file_size\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
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
        str1 = str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


if __name__ == '__main__':
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
    origtrain, origtest, features = read_test_train()
    print('Length of train: ', len(origtrain))
    print('Length of test: ', len(origtest))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    test = origtest.copy()
    test['result'] = -1
    ss = StandardScaler()
    x = pd.concat([origtrain.copy(), test.copy()])
    x[features] = ss.fit_transform(x[features])
    train = x[x.result != -1].copy()
    test = x[x.result == -1].copy()
    test.drop(['result'], inplace=True, axis=1)
    score = roc_auc_score(train.result, GP(train))
    print('ROC:', score)
    predictions = GP(test)
    create_submission(score, origtest, predictions.values)
