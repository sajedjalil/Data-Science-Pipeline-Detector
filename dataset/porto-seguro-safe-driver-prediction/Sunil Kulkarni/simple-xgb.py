# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns


import sklearn
import xgboost as xgb 

from scipy import stats
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import binarize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.tools.plotting import scatter_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import csv as csv

#import missingno as msno
import matplotlib.pyplot as plt

#%matplotlib inline
p = sns.color_palette("hls", 8)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#system("ls ../input")

print("begin of part 1 ::")

#path_train = "D:/CBA/Practice/Safe driver prediction/train.csv"
#path_test = "D:/CBA/Practice/Safe driver prediction/test.csv"

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#train = pd.read_csv(path_train)
#test = pd.read_csv(path_test)

#missing values
train_copy = train
train = train.replace(-1, np.NaN)
test = test.replace(-1, np.NaN)

# deleted ps_car_03_cat, ps_car_05_cat - more than 50% of values are missing 
train = train.drop(['ps_car_03_cat','ps_car_05_cat'], axis = 1)
test = test.drop(['ps_car_03_cat','ps_car_05_cat'], axis = 1)

# separate categorical variables from others
############################################

bins = [1,10,20,30,40,50,60,70,80,90,100,110]
lab = [1,10,20,30,40,50,60,70,80,90,100]

train['ps_car_11_new_cat'] = pd.cut(train['ps_car_11_cat'], bins, labels=lab).astype('object')
test['ps_car_11_new_cat'] = pd.cut(test['ps_car_11_cat'], bins, labels=lab).astype('object')

train = train.drop('ps_car_11_cat', axis=1)
test = test.drop('ps_car_11_cat', axis=1)

# missing value analysis 
############################
#mis_col = train.isnull().sum().reset_index()
#mis_col.columns = ["Feature", "No_of_mis"]
#mis_col["Perc"] = (mis_col.No_of_mis / train.shape[0]) * 100
#mis_col = mis_col[mis_col.Perc > 0]
#
#mis_col = mis_col.sort_values(by="Perc", ascending = False).reset_index(drop=True)


# imputing with mean values for numeric variables identified by above code 

mis_num = ['ps_reg_03','ps_car_14','ps_car_11','ps_car_12','ps_car_14']

imp = Imputer(strategy = 'mean')
train[mis_num] = imp.fit_transform(train[mis_num])
test[mis_num] = imp.fit_transform(test[mis_num])

# distribution of missing categorical variables to check skewness
mis_cat = ['ps_car_07_cat','ps_ind_05_cat','ps_car_11_new_cat','ps_car_09_cat','ps_ind_02_cat',
           'ps_car_01_cat','ps_ind_04_cat','ps_car_02_cat']

# for below cat vars - replace by respective most frequent values 

imp = Imputer(strategy = 'most_frequent')
train[mis_cat] = imp.fit_transform(train[mis_cat])
test[mis_cat] = imp.fit_transform(test[mis_cat])

cat_features = train.columns[train.columns.str.endswith(('cat'))]

train[cat_features] = train[cat_features].astype(object)
test[cat_features] = test[cat_features].astype(object)

train_dum = pd.get_dummies(train, drop_first=True)
test_dum = pd.get_dummies(test, drop_first=True)

## Dummy variable trap 
#tr_dum_drop = [] 
#tst_dum_drop = [] 
#
#for i in cat_features:
#    tr_dum_drop += [i+'_'+str(train[i].unique()[-1])]
#    tst_dum_drop += [i+'_'+str(test[i].unique()[-1])]
#
#print("dummy drop train :: \n ", tr_dum_drop)
#print("dummy drop test :: \n ", tst_dum_drop)
#    
#print(train_dum.dtypes)
#print(test_dum.dtypes)

#train_dum.drop(tr_dum_drop, inplace=True)
#test_dum.drop(tst_dum_drop, inplace=True)

#print(train_dum.dtypes)
#print("train shape :: ", train_dum.shape)
#print("test shape :: ",test_dum.shape)


# Dropping target variable:
train_out = train_dum['target']
train_inp = train_dum.drop(['id','target'], axis = 1)
test_inp = test_dum.drop(['id'], axis = 1)

cat_features_dum = train_inp.columns[train_inp.columns.str.contains(('_cat'))]
num_features = train_inp.columns.difference(train_inp[cat_features_dum].columns)

train_x, test_x, train_y, test_y = train_test_split(train_inp, train_out, test_size=0.20, random_state = 6)

# standardizing the dataset 

std = preprocessing.StandardScaler().fit(train_x[num_features])
train_x[num_features] = std.transform(train_x[num_features])
test_x[num_features] = std.transform(test_x[num_features])

print('end of part 1 ::')



###############################################
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# Create an XGBoost-compatible metric from Gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

# scale_pos = 
params = {'eta': 0.09, 'max_depth': 6, 'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.6,
          'min_child_weight': 0.77, 'scale_pos_weight': 1.6, 'gamma': 10, 'reg_alpha': 8, 'reg_lambda': 1.3, 
          'eval_metric': 'auc', 'seed': 16, 'silent': True}

x1 = train_x
y1 = train_y

x2 = test_x
y2 = test_y

dtrain = xgb.DMatrix(x1, y1)
dvalid = xgb.DMatrix(x2, y2)
dtest = xgb.DMatrix(test_inp)

watchlist = [(dtrain,'train'),(dvalid,'test')]

print('before xgb train :: \n')

model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, feval=gini_xgb,
                  maximize=True, verbose_eval=50, early_stopping_rounds=100)

print('after xgb train :: \n')

#################################
print('begin of part 3 :: ')

#xgb.plot_importance(booster=model,max_num_features=15)

pred = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
pred_test = model.predict(dtest, ntree_limit=model.best_ntree_limit)

# Create a submission file
sub = pd.DataFrame()
sub['id'] = test_dum['id'].values
sub['target'] = pred_test

sub.to_csv('xgb1.csv', index=False, float_format='%.5f')
print(sub.head())