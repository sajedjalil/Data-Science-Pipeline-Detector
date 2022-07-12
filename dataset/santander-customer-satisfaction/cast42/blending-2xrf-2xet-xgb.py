# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Based on Blending code from Emanuele Olivetti

"""
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

if __name__ == '__main__':

    np.random.seed(1301) # seed to shuffle the train set

    n_folds = 3
    verbose = True
    shuffle = False

    training = pd.read_csv("../input/train.csv", index_col=0)
    test = pd.read_csv("../input/test.csv", index_col=0)
    print(training.shape)
    print(test.shape)

    # Replace -999999 in var3 column with most common value 2 
    # See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
    # for details
    training = training.replace(-999999,2)

    X = training.iloc[:,:-1]
    y = training.TARGET

# Feature selection from 
# https://www.kaggle.com/cast42/santander-customer-satisfaction/xgb-for-features-selection-0-837

    features = ['var38', 'var15', 'saldo_medio_var5_hace2', 'saldo_var30',
'saldo_medio_var5_ult1', 'saldo_medio_var5_hace3', 'num_var45_hace3',
'saldo_medio_var5_ult3', 'num_var22_ult3', 'num_var22_hace3', 'num_var45_hace2',
'saldo_var5', 'imp_op_var41_ult1', 'imp_op_var41_efect_ult3', 'num_var45_ult3', 
'num_var22_ult1', 'imp_op_var41_efect_ult1', 'num_meses_var39_vig_ult3',
'num_var22_hace2', 'num_var45_ult1', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1',
'var3', 'imp_op_var39_comer_ult3', 'saldo_var37', 'imp_op_var39_ult1', 'var36',
'num_med_var45_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_ult3',
'imp_trans_var37_ult1', 'imp_var43_emit_ult1', 'num_var42_0', 'saldo_var8',
'ind_var8_0', 'num_op_var41_efect_ult3', 'num_ent_var16_ult1', 'saldo_var26',
'num_op_var41_hace2', 'num_meses_var5_ult3', 'imp_op_var41_comer_ult1',
'saldo_medio_var8_ult1', 'imp_sal_var16_ult1', 'num_op_var41_hace3', 'ind_var5_0',
'num_op_var41_ult1', 'num_var5_0', 'num_var37_0', 'num_op_var39_comer_ult1', 'num_var4', 
'num_var43_emit_ult1', 'num_op_var41_comer_ult1', 'num_var37_med_ult2', 'saldo_var31',
'ind_var39_0', 'num_op_var39_comer_ult3', 'saldo_var42', 'imp_op_var41_comer_ult3',
'num_op_var39_efect_ult1', 'ind_var30', 'num_op_var39_ult1', 'ind_var1_0',
'saldo_medio_var8_hace3', 'num_op_var40_comer_ult3', 'num_op_var41_comer_ult3',
'num_var43_recib_ult1']
    print (features)
    X_sel = X[features]
    sel_test = test[features]
    X, y, X_submission = np.array(X_sel), np.array(y.astype(int)).ravel(), np.array(sel_test)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    ratio = float(np.sum(y == 1)) / np.sum(y==0)
    clfs = [RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced'),
            GradientBoostingClassifier(n_estimators=200,max_depth=5),
            AdaBoostClassifier(n_estimators=200),
            xgb.XGBClassifier(missing=9999999999,
                    max_depth = 6,
                    n_estimators=200,
                    learning_rate=0.1, 
                    nthread=4,
                    subsample=1.0,
                    colsample_bytree=0.5,
                    min_child_weight = 3,
                    scale_pos_weight = ratio,
                    reg_alpha=0.01,
                    seed=1301)]

    print ("Creating train and test sets for blending.")
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    skf = cross_validation.StratifiedKFold(y, n_folds, shuffle=True)
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
        for i, (train, testidx) in enumerate(skf):
            print ("Fold", i)
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[testidx], y[testidx]
#            clf.fit(X_train, y_train)
            if j < len(clfs)-1:
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
                    eval_set=[(X_test, y_test)])
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[testidx, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(axis=1)

    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    submission = pd.DataFrame({"ID":test.index, "TARGET":y_submission})
    submission.to_csv("submission_RF_GBT_ABC_XGB.csv", index=False)
    