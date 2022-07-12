# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#!/usr/bin/python
# -*- coding: utf-8 -*-

from pandas import Series,DataFrame
from numpy import *  
import csv  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn import tree
from sklearn import svm
import time
from functools import wraps
from sklearn.metrics import classification_report

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# from keras.models import Sequential 
# from keras.layers.core import Dense, Activation
# from keras.utils import np_utils

def monitor_time(func):

    @wraps(func)
    def calculate_time(*args, **kwargs ):
        start_time = time.time()
        result=func(*args, **kwargs)
        end_time=time.time()
        cost_time=end_time-start_time
        print(cost_time)
        return result

    return calculate_time


def save_result(id, results,file):  
    this_file=open(file,'w')
    this_file.write("id,type\n")
    for i, v in zip(id, results):
        this_file.write(str(i)+","+str(v)+"\n")

    this_file.close()

@monitor_time
def logistic_regression(train_data,train_results,test_data):

    lr = LogisticRegression(penalty='l2',C=1000000)
    lr.fit(train_data,train_results)
    test_results= lr.predict(test_data) 
    return test_results


@monitor_time
def support_vector_machine(train_data,train_results,test_data):

    pca = PCA(n_components=0.8, whiten=True) 

    train_x = pca.fit_transform(train_data) 
    test_x = pca.transform(test_data) 

    svm_dr = svm.SVC(kernel='rbf', C=10)
    svm_dr.fit(train_x, train_results) 

    test_results=svm_dr.predict(test_x)
    return test_results 


@monitor_time
def random_forest_classify(train_data,train_results,test_data):

    rf = RandomForestClassifier(n_estimators=100,min_samples_split=5)
    rf.fit(train_data, train_results)
    test_label=rf.predict(test_data)
    return test_results


def run_classifier(train_data, train_results, test_data, model):

    if  model=='rf':
        test_results=logistic_regression(train_data,train_results,test_data)
    elif model=='svm':
        test_results=support_vector_machine(train_data,train_results,test_data)
    elif model=='rf':
        test_results=random_forest_classify(train_data,train_results,test_data)

    return test_results



def main():

    df_train=pd.read_csv('../input/train.csv')
    df_test=pd.read_csv('../input/test.csv')

    sns.set()
    sns.pairplot(df_train[["bone_length", "rotting_flesh", "hair_length", "has_soul", "type"]], hue="type")
    # sns.plt.show()

    df_train['hair_soul'] = df_train['hair_length'] * df_train['has_soul']
    df_train['hair_bone'] = df_train['hair_length'] * df_train['bone_length']
    df_train['hair_soul_bone'] = df_train['hair_length'] * df_train['has_soul'] *df_train['bone_length']


    df_test['hair_soul'] = df_test['hair_length'] * df_test['has_soul']
    df_test['hair_bone'] = df_test['hair_length'] * df_test['bone_length']
    df_test['hair_soul_bone'] = df_test['hair_length'] * df_test['has_soul'] * df_test['bone_length']

    test_id = df_test['id']

    df_train.drop(['id'], axis=1, inplace=True)
    df_test.drop(['id'], axis=1, inplace=True)

    df_train.drop(['color'], axis=1, inplace=True)
    df_test.drop(['color'], axis=1, inplace=True)

    df_train_data = df_train.drop('type', axis=1)
    df_train_results=df_train['type']

    df_train_data = pd.get_dummies(df_train_data)
    df_test_data = pd.get_dummies(df_test)

    # test_results=run_classifier(df_train_data,df_train_results,df_test_data, 'rf')
    # save_result(test_id, test_results,'results_logistic_regression.csv')

    # test_results=run_classifier(df_train_data,df_train_results,df_test_data, 'svm')
    # save_result(test_id, test_results,'results_svm.csv')

    test_results=run_classifier(df_train_data,df_train_results,df_test_data, 'rf')
    save_result(test_id, test_results,'results_rf.csv')

if __name__=='__main__':
    main()