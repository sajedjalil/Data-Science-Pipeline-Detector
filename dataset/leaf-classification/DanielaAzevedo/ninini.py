# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm, grid_search, cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

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


#cv, 1.0


    
def par_svm():
    filt_data,ind = Univ()
    svm_model_d = svm.SVC()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 3, 10, 100],'gamma':[0.01, 0.001]}
    opt_model_d = grid_search.GridSearchCV(svm_model_d, parameters)
    opt_model_d.fit(filt_data, y_train_o)
    #print (opt_model_d.best_estimator_)
    #scores_gs = cross_validation.cross_val_score(opt_model_d, filt_data,y_train_o, cv = 10)
    #print (scores_gs.mean())
    #return scores_gs.mean()
    return opt_model_d.predict(X_test.loc[:,ind])
#Best parameters    
    #SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#  kernel='linear', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False)  
#CV 0.960606...


#RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#            max_depth=None, max_features=4, max_leaf_nodes=None,
#            min_samples_leaf=2, min_samples_split=4,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
#0.965656565657


def validation(reps=0):
    predictions = par_svm()
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