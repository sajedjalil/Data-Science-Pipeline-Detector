# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold,StratifiedShuffleSplit,train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def target_convert(target):
    if target == 'Class_1':
        return 1
    elif target == 'Class_2':
        return 2
    elif target == 'Class_3':
        return 3
    elif target == 'Class_4':
        return 4
    elif target == 'Class_5':
        return 5
    elif target == 'Class_6':
        return 6
    elif target == 'Class_7':
        return 7
    elif target == 'Class_8':
        return 8
    else:
        return 9
    
# Data Preprocess, scale data
def data_preprocessing(train_fname, test_fname):
    train_data = pd.read_csv(train_fname)
    test_data = pd.read_csv(test_fname)
    target = train_data['target'].apply(target_convert)
    del train_data['target']
    del train_data['id']
    del test_data['id']
    train = pd.DataFrame(preprocessing.scale(train_data))
    test = pd.DataFrame(preprocessing.scale(test_data))
    return train, target, test
    
# This come from the Competition Method
def evaluation(label,pred_label):
    num = len(label)
    logloss = 0.0
    label_array = np.array(label)
    for i in range(num):
        p = max(min(pred_label[i][label_array[i]-1],1-10**(-15)),10**(-15))
        logloss += np.log(p)
    logloss = -1*logloss/num
    return logloss
    
FILE_TRAIN = '../input/train.csv'
FILE_TEST = '../input/test.csv'

# Runing RF Model
rf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
train_data, train_target, test_data = data_preprocessing(FILE_TRAIN,FILE_TEST)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.02, random_state=0)
rf.fit(X_train, y_train)
pre_label = rf.predict_proba(X_test)
print (pre_label)
print (y_test)
score = evaluation(y_test,pre_label)
print (score)
submission = pd.DataFrame(rf.predict_proba(test_data))