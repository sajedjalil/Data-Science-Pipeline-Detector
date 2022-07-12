# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm, grid_search, cross_validation, preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


def load_data():
    
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
    X_train = train.loc[:, 'margin1':]
    y_train = train.pop('species')
    X_test = test.loc[:,'margin1':]
    

    return X_train, y_train, X_test

#normalization / standardization / binarization
def pre_processing(method_PP = "standardization"):
    
    X_train, y_train, X_test = load_data()
    
    if method_PP == "standardization":
        # standardize the data by setting the mean to 0 and std to 1
        X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
        X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
    
    elif method_PP == "normalization":
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)
        
    elif method_PP == "binarization":
        X_train = preprocessing.Binarizer().fit(X_train)
        X_train.transform(X_train)
        X_test = preprocessing.Binarizer().fit(X_test)
        X_test.transform(X_test)
    
    return X_train, X_test, y_train

#normalization / standardization / binarization
#univariance / variance
def filtering(method_PP = "standardization", method_F = "univariance"):
    
    X_train, X_test, y_train = pre_processing(method_PP)
    
    if method_F == "univariance":
        filt_kb_method = SelectKBest(f_classif, k=179)
        X_train = filt_kb_method.fit_transform(X_train, y_train)
    
    elif method_F == "variance":
        sel = VarianceThreshold()
        X_train = sel.fit_transform(X_train)
    
    return X_train, X_test, y_train
    

#svm / random_forest / neural_networks / True or False / True or False /True or False (if last perform grid)
def algorithm(method_A = "svm", OneVsRest = True, OneVsOne = False, randomized = True):
    
    if method_A == "svm":
        parameters_svm = {'kernel':('linear', 'rbf'), 'C':[1, 3, 10, 100],'gamma':[0.01, 0.001]}
        model = svm.SVC()
        model = search_par(randomized, model, parameters_svm)
    
    elif method_A == "random_forest":
        parameters_random = {"max_depth": [2, 3, None], "max_features": [2,4,6], "min_samples_split": [2,4,6], "min_samples_leaf": [2,4,6], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
        model = RandomForestClassifier(n_estimators = 100)
        model = search_par(randomized, model, parameters_random)
    
    elif method_A == "neural_networks":
        model = MLPClassifier()
    
    if OneVsRest:
        return OneVsRestClassifier(model)
    
    elif OneVsOne:
        return OneVsOneClassifier(model)
    
    return model

#True or False / model / parameters
def search_par(randomized, model, parameters):
    
    if randomized:
        return RandomizedSearchCV(model, param_distributions=parameters, n_iter=10)
    
    else:
        return grid_search.GridSearchCV(model, parameters)
        
#normalization / standardization / binarization
#univariance / variance    
#random / grid  
#calIbrated true or false
def accuracies(method_PP = "standardization", method_F = "univariance", method_A = "svm", OneVsRest = True, OneVsOne = False, randomized = True, calibrated = True):
    
    X_train, X_test, y_train = filtering(method_PP, method_F)
    
    model = algorithm(method_A, OneVsRest, OneVsOne, randomized)
    
    model.fit(X_train, y_train)
    
    if calibrated:
        pass
        
    else:
        score = cross_validation.cross_val_score(model, X_train,y_train, cv = 10)
        print(score.mean())
        return score.mean()


accuracies(method_PP = "standardization", method_F = "univariance", method_A = "svm", OneVsRest = False, OneVsOne = False, randomized = True, calibrated = False)
#oneVSrest()
#calibrated_neural_networks()        
#write_2()
#calibrated_random()