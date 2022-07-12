# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Loading the training and test data
train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

#importing the required modules
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

train = train.replace(-999999,2)
print('There are {} features.'.format(train.shape[1]))

y = train.TARGET
print(train.columns)
train = train.drop(['TARGET'], axis=1)
# Add #0 per row as extra feature
train['n0'] = (train==0).sum(axis=1)
test['n0'] = (test == 0).sum(axis=1)

def var36_99(var):
    if var == 99:
        return 1
    else:
        return 0
def var36_0123(var):
    if var != 99:
        return 1
    else:
        return 0


# Remove Constant Columns 
columnsToRemove = []
for col in train.columns:
    if train[col].std() == 0:
        columnsToRemove.append(col)
    
train.drop(columnsToRemove, axis=1, inplace=True)

# Remove duplicate Columns 

columnsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    for j in range(i+1, len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            columnsToRemove.append(columns[j])
            

train.drop(columnsToRemove, axis=1, inplace=True)

train['var36_99'] = train.var36.apply(var36_99)
train['var36_0123'] = train.var36.apply(var36_0123)

train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0

test['var36_99'] = test.var36.apply(var36_99)
test['var36_0123'] = test.var36.apply(var36_0123)

test['var38mc'] = np.isclose(test.var38, 117310.979016)
test['logvar38'] = test.loc[~test['var38mc'], 'var38'].map(np.log)
test.loc[test['var38mc'], 'logvar38'] = 0

train.drop(['var38','var36'],axis=1, inplace=True)
col = [x for x in train.columns if x not in ['TARGET']]
X = train[col]

def feature_selection(InputDf,output):
    
    # First select features based on chi2 and f_classif
    p = 3
    X = InputDf
    y = output
    X_bin = Binarizer().fit_transform(scale(X))
    selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
    selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
    print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),chi2_selected_features))
    f_classif_selected = selectF_classif.get_support()
    f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
    print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),f_classif_selected_features))
    selected = chi2_selected & f_classif_selected
    print('Chi2 & F_classif selected {} features'.format(selected.sum()))
    features = [ f for f,s in zip(X.columns, selected) if s]
    return features
    

features = feature_selection(X,y)
finalInput = train[features]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(finalInput, y, random_state=1301, stratify=y, test_size=0.3)

dtrain = xgb.DMatrix(X_train, label=y_train, missing=9999999999)
dtest = xgb.DMatrix(X_test, label=y_test, missing=9999999999)

param = {'bst:max_depth':5, 'bst:eta':0.0202048, 'silent':1, 'objective':'binary:logistic','bst:subsample':0.6815, 'bst:colsample_bytree':0.7}
param['nthread'] = 4
param['eval_metric'] = 'auc'

num_round = 560

evallist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train( param, dtrain, num_round, evallist )

#AUCTotal = roc_auc_score(y, clf.predict_proba(X_train, ntree_limit=clf.best_iteration)[:,1])   
#print('Overall AUC:', AUCTotal)

sel_test = test[features]
xgmat = xgb.DMatrix(sel_test)
y_pred = bst.predict(xgmat,ntree_limit=bst.best_ntree_limit)
test['TARGET'] = y_pred
test.loc[test['var15'] <23, 'TARGET'] = 0

submission = pd.DataFrame({"ID":test.index, "TARGET":test['TARGET']})
submission.to_csv("submission.csv", index=False)