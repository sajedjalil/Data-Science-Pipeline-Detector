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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train = train.loc[:, 'margin1':]
y_train = train["species"]

X_test = test.loc[:,'margin1':]

def Univ_filt():
    #svm_model = svm.SVC(kernel = "linear", C=100.)
    filt_kb_method = SelectKBest(f_classif, k=179)
    filt_kb = filt_kb_method.fit_transform(X_train, y_train)
    #scores_kb = cross_validation.cross_val_score(svm_model, filt_kb, y_train, cv = 10)
    #print (scores_kb.mean())
    #print(filt_kb.loc[:,"margin1":])
    #print(filt_kb_method.get_support(True))
    return filt_kb,filt_kb_method.get_support()

def calibrated_random():
    filt_data,ind = Univ_filt()
    rf_model = RandomForestClassifier(n_estimators=100)
    calibrated_model = CalibratedClassifierCV(rf_model, method='sigmoid', cv=10)
    calibrated_model.fit(filt_data,y_train)
    #print(calibrated_model.score(filt_data, y_train))
    #return calibrated_model.score(filt_data, y_train)
    return calibrated_model.predict_proba(X_test.loc[:,ind])
#cv, 1.0
#position 815, score 0.518
   
def grid_par_search_svm():
    filt_data,ind = Univ_filt()
    svm_model_d = svm.SVC()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 3, 10, 100],'gamma':[0.01, 0.001]}
    opt_model_d = grid_search.GridSearchCV(svm_model_d, parameters)
    opt_model_d.fit(filt_data, y_train)
    #print (opt_model_d.best_estimator_)
    #scores_gs = cross_validation.cross_val_score(opt_model_d, filt_data,y_train, cv = 10)
    #print (scores_gs.mean())
    #return scores_gs.mean()
    return opt_model_d.predict(X_test.loc[:,ind])
#Best parameters    
    #SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#  kernel='linear', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False)  
#CV 0.960606...

def randomized_par_search_random():
    filt_data,ind = Univ_filt()
    rf_model = RandomForestClassifier(n_estimators=100)
    param_dist = {"max_depth": [2, 3, None], "max_features": [2,4,6], "min_samples_split": [2,4,6], "min_samples_leaf": [2,4,6], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
    rand_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10)
    rand_search.fit(filt_data, y_train)
    #print (rand_search.best_estimator_)
    #scores_rs = cross_validation.cross_val_score(rand_search, filt_data, y_train, cv = 10)
    #print (scores_rs.mean())
    #return scores_rs.mean()
    return rand_search.predict(X_test.loc[:,ind])
#RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#            max_depth=None, max_features=4, max_leaf_nodes=None,
#            min_samples_leaf=2, min_samples_split=4,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
#0.965656565657
#position unknown. score 1.400

def calibrated_neural_networks():
    filt_data,ind = Univ_filt()
    clf = MLPClassifier()
    calibrated_model = CalibratedClassifierCV(clf, method='sigmoid', cv=10)
    calibrated_model.fit(filt_data,y_train)                       
    print(calibrated_model.score(filt_data, y_train))
    #return calibrated_model.score(filt_data, y_train)

#MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)

def oneVSrest():
    filt_data, ind = Univ_filt()
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=10)
    calibrated_model = model.fit(filt_data, y_train)
    #print(model.score(filt_data, y_train))
    return calibrated_model.predict(X_test.loc[:,ind])
    
def validation(reps=10):
    ids = list(test.loc[:,"id"])
    species = list(np.unique(y_train))
    predictions_counts = np.zeros([len(ids),len(species)])
    if reps == 0:
        predictions = oneVSrest()
        lin = 0
        for pred in predictions:
            col = species.index(pred)
            predictions_counts[lin,col] += 1
            lin += 1
    elif reps > 0:
        for rep in range(reps):
            predictions = oneVSrest()
            lin = 0
            for pred in predictions:
                col = species.index(pred)
                predictions_counts[lin,col] += 1
                lin += 1
    for linha in range(len(predictions_counts)):
        soma = sum(predictions_counts[linha])
        for coluna in range(len(predictions_counts[linha])):
            predictions_counts[linha,coluna] = predictions_counts[linha,coluna]/soma
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

def write_2():
    ids = test.pop('id')
    species = list(np.unique(y_train))
    probs = calibrated_random()
    yPred = pd.DataFrame(probs,index=ids,columns=species)
    print('Creating and writing submission...')
    fp = open('random_probs.csv', 'w')
    fp.write(yPred.to_csv())

#oneVSrest()
#calibrated_neural_networks()        
write_2()
#calibrated_random()