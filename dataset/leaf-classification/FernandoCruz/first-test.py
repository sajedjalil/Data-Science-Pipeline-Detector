# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train_o = train.loc[:, 'margin1':]
y_train_o = train["species"]

X_test = test.loc[:,'margin1':]

def Univ():
    #svm_model = svm.SVC(kernel = "linear", C=100.)
    filt_kb_method = SelectKBest(f_classif, k=179)
    filt_kb = filt_kb_method.fit_transform(X_train_o, y_train_o)
    #scores_kb = cross_validation.cross_val_score(svm_model, filt_kb, y_train_o, cv = 10)
    #print (scores_kb.mean())
    #print(filt_kb.loc[:,"margin1":])
    #print(filt_kb_method.get_support(True))
    return filt_kb,filt_kb_method.get_support()

def par_random():
    filt_data,ind = Univ()
    rf_model = RandomForestClassifier(n_estimators=100)
    param_dist = {"max_depth": [2, 3, None], "max_features": [2,4,6], "min_samples_split": [2,4,6], "min_samples_leaf": [2,4,6], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
    rand_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10)
    rand_search.fit(filt_data, y_train_o)
    #print (rand_search.best_estimator_)
    #scores_rs = cross_validation.cross_val_score(rand_search, filt_data, y_train_o, cv = 10)
    #print (scores_rs.mean())
    #return scores_rs.mean()
    return rand_search.predict(X_test.loc[:,ind])


def validation(reps=0):
    predictions = par_random()
    ids = list(test.loc[:,"id"])
    species = list(np.unique(y_train_o))
    predictions_counts = np.zeros([len(ids),len(species)])
    lin = 0
    for pred in predictions:
        col = species.index(pred)
        predictions_counts[lin,col] += 1
        lin += 1
    if reps > 0:
        for rep in range(reps):
            predictions = calibratedCV()
            lin = 0
            for pred in predictions:
                col = species.index(pred)
                predictions_counts[lin,col] += 1
                lin += 1
    return predictions_counts, species, ids

def write():
    probs, species, ids = validation()
    file = open("test_logistic_regression.csv", "w")
    file.write("id")
    for specie in species:
        file.write(","+str(specie))
    file.write("\n")
    for i in range(len(probs)):
        file.write(str(ids[i]))
        for j in probs[i]:
            file.write(","+str(j))
        file.write("\n")

write()


    

    