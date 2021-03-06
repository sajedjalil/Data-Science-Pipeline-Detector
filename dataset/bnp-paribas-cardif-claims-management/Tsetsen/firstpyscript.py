import pandas as pd
import numpy as np
import csv
import scipy as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from pandas import DataFrame
import copy
#############################################################
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
#############################################################
def splitData(train):
    count_0 = 27300 
    count_1 = 87021
    
    # comment this 
    # for line in train.values: 
    #     if(line[1] == 0):
    #         count_0 += 1
    #     else:
    #         count_1 += 1
    # print (count_0,count_1)
    
    # then split data
    headers = ['ID', 'target', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v46', 'v47', 'v48', 'v49', 'v50', 'v51', 'v52', 'v53', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v64', 'v65', 'v66', 'v67', 'v68', 'v69', 'v70', 
    'v71', 'v72', 'v73', 'v74', 'v75', 'v76', 'v77', 'v78', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v105', 'v106', 'v107', 'v108', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v123', 'v124', 'v125', 'v126', 'v127', 'v128', 'v129', 'v130', 'v131']

    newTrainData = []
    newTest = []
    act = []
    c_0 = 0
    c_1 = 0
    
    for line in train.values:
        if line[1] == 0:
            if c_0 < count_0/2:
                newTrainData.append(line)
                c_0 +=1
            else:
                newTest.append(line)
                act.append(line[1])
        else:
            if c_1 < count_1/2:
                newTrainData.append(line)
                c_1 +=1
            else:
                newTest.append(line)
                act.append(line[1])
    
    dfTrain = DataFrame(newTrainData, columns=headers)
    dfTest = DataFrame(newTest, columns=headers)
    dfTest2 = DataFrame(newTest, columns=headers)
    del dfTest['target']

    return dfTrain, dfTest, act, dfTest2
#############################################################
def calculateScoreDecisionTree(max_fs, target, train, id_test, test):
    
    for (train_col, train_type), (test_col, test_type) in zip(train.iteritems(),test.iteritems()):
        if train_type.dtype == 'O':
            #for objects: factorize
            train[train_col], tmp_indexer = pd.factorize(train[train_col])
            test[test_col] = tmp_indexer.get_indexer(test[test_col])
            #but now we have -1 values (NaN)
        else:
            #for int or float: fill NaN
            tmp_len = len(train[train_type.isnull()])
            if tmp_len>0:
                #print "mean", train_series.mean()
                train.loc[train_type.isnull(), train_col] = -999 
            #and Test
            tmp_len = len(test[test_type.isnull()])
            if tmp_len>0:
                test.loc[test_type.isnull(), test_col] = -999
    
    X_train = train
    X_test = test
    #print('Training...')
    extc = ExtraTreesClassifier(n_estimators=1200,max_features=max_fs,criterion= 'entropy',min_samples_split= 2,
                                max_depth= 30, min_samples_leaf= 2, n_jobs = -1)
    extc.fit(X_train,target)
    
    y_pred = extc.predict_proba(X_test)
    y_pred = y_pred[:,1]
    
    return y_pred
#############################################################
def findRange(value,range_value):
	if((value >= range_value) and (value < range_value+0.025)):
		return True
	return False
#############################################################
print('Load data...',end="")
train = pd.read_csv("../input/train.csv")
#train, test, act, range_test = splitData(train) 
test = pd.read_csv("../input/test.csv")
print('Done!')
backup_train = train.copy(deep=True)
backup_test = test.copy(deep=True)
target = backup_train['target'].values
id_test = backup_test['ID'].values
#train.to_csv("trainData.csv")

columns = ['v3','v10','v12','v14','v21','v22','v24','v31','v34','v38','v40','v47','v50','v52','v56','v62','v66','v71','v72','v74','v75','v79','v91','v107','v110','v112','v114','v125','v129']

train = backup_train[columns].copy(deep=True)
test = backup_test[columns].copy(deep=True)

# First DecisionTreeClassifier
print('Calculate Predictions(First Tree)...', end="")
y = calculateScoreDecisionTree(len(columns), target, train, id_test, test)
print('Done!')
pd.DataFrame({"ID": id_test, "PredictedProb": y}).to_csv('decisionTree.csv',index=False)
#GiveError()