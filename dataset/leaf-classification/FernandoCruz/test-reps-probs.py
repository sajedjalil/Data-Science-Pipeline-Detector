# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train = train.loc[:, 'margin1':]
y_train = train["species"]

X_test = test.loc[:,'margin1':]

def Univ():
    #svm_model = svm.SVC(kernel = "linear", C=100.)
    filt_kb_method = SelectKBest(f_classif, k=179)
    filt_kb = filt_kb_method.fit_transform(X_train, y_train)
    #scores_kb = cross_validation.cross_val_score(svm_model, filt_kb, y_train, cv = 10)
    #print (scores_kb.mean())
    #print(filt_kb.loc[:,"margin1":])
    #print(filt_kb_method.get_support(True))
    return filt_kb,filt_kb_method.get_support()

def calibratedCV():
    filt_data,ind = Univ()
    rf_model = RandomForestClassifier(n_estimators=100)
    calibrated_model = CalibratedClassifierCV(rf_model, method='sigmoid', cv=10)
    calibrated_model.fit(filt_data,y_train)
    #print(calibrated_model.score(filt_data, y_train))
    #return calibrated_model.score(filt_data, y_train)
    return calibrated_model.predict(X_test.loc[:,ind])


def validation(reps=20):
    ids = list(test.loc[:,"id"])
    species = list(np.unique(y_train))
    predictions_counts = np.zeros([len(ids),len(species)])
    if reps == 0:
        predictions = calibratedCV()
        lin = 0
        for pred in predictions:
            col = species.index(pred)
            predictions_counts[lin,col] += 1
            lin += 1
    elif reps > 0:
        for rep in range(reps):
            predictions = calibratedCV()
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

write()
