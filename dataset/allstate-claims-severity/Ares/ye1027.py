# -*- coding: utf-8 -*-
"""
Based on Faron' script 
https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
"""
## loading packages 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import gc
import operator


### setup
ID = 'id'
TARGET = 'loss'
NFOLDS = 5
SEED = 1
NROWS = 10
DATA_DIR = "../input"
OUT_DIR ="../output"


### reading data 
TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)

train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
test = pd.read_csv(TEST_FILE, nrows=NROWS)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = np.log(train[TARGET]).ravel()
id_train= train[ID]
id_test= test[ID]

train_test = pd.concat((train, test)).reset_index(drop=True)

### factorize
cats=[]
for feature in train.columns:
    if 'cat' in feature:
        cats.append(feature)
    else:
        continue


for cat in cats:
    edict=train_test[cat].value_counts().to_dict()
    sorted_edict=sorted(edict.items(), key=operator.itemgetter(1))
    number=(len(sorted_edict))
    sorted_edict2=list()
    for a in sorted_edict:
        a1=list(a)
        sorted_edict2.append(a1)
    for number1 in range(number):
        sorted_edict2[number1][0]=number-number1-1
    for key,value in edict.items():
        for b in range(number):
            if value==sorted_edict2[b][1]:
                edict[key]=sorted_edict2[b][0]
            else:
                continue
    for key,value in edict.items():
        train_test.loc[train_test[cat] == key,cat]=value




### reorder? 
train_test=train_test.sort_values(ID)
gc.collect()



#### preprocessing
train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"])+0000.1)
train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"])+0000.1)
train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"])+0000.1)
train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"])+0000.1)
train_test["cont14"]=(np.maximum(train_test["cont14"]-0.179722,0)/0.665122)**0.25


### define x_train, x_test
train_test.drop([ID, TARGET], axis=1, inplace=True)
x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])
print("{},{}".format(x_train.shape, x_test.shape))
