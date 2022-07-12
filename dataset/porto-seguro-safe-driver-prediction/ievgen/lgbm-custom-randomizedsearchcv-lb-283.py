# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import gc

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from time import time
from random import choice
from scipy.stats import randint as sp_randint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

np.random.seed(17)

df_tn = pd.read_csv('../input/train.csv')
df_tt = pd.read_csv('../input/test.csv')

# replace -1 for NaN

# train set
df_tn_z = df_tn.copy()
df_tn_z.replace(-1, np.NaN, inplace = True)

# test set
df_tt_z = df_tt.copy()
df_tt_z.replace(-1, np.NaN, inplace = True)

# -1 can be changed to 0 for features where there is no category "0",
# and features that have numerical values. Scrip below identifies such 
# features as well as those where -1 shouldn't be changed.

# list with features
zero_list = ['ps_ind_02_cat', 'ps_reg_03', 'ps_car_12', 'ps_car_12', 'ps_car_14',] # -1 can be changed for 0 in this features

minus_one = ['ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat',
             'ps_car_03_cat', 'ps_car_07_cat', 'ps_car_05_cat', 'ps_car_09_cat',
             'ps_car_11'] # these features already have 0 as value, thus -1 shouldn't be changed


# fill in missing values with 0 or -1

# train set
df_tn_z[minus_one] = df_tn_z[minus_one].fillna(-1)
df_tn_z[zero_list] = df_tn_z[zero_list].fillna(0)

# test set
df_tt_z[minus_one] = df_tt_z[minus_one].fillna(-1)
df_tt_z[zero_list] = df_tt_z[zero_list].fillna(0)


# group features by nature
cat_f = ['ps_ind_02_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 
         'ps_car_03_cat',  'ps_car_04_cat','ps_car_05_cat', 'ps_car_06_cat',
         'ps_car_07_cat', 'ps_car_08_cat','ps_car_09_cat', 'ps_car_10_cat', 
         'ps_car_11_cat']
bin_f = ['ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
         'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
         'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
         'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',
         'ps_calc_19_bin', 'ps_calc_20_bin']
ord_f = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11']

cont_f = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 
          'ps_car_14', 'ps_car_15',  'ps_calc_01', 'ps_calc_02', 'ps_calc_03',
          'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
          'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',
          'ps_calc_14']

# transform categorical values to dummies
df_tn_proc = df_tn_z.copy().drop(['id', 'target'], axis = 1)
df_tt_proc = df_tt_z.copy().drop(['id'], axis = 1)
df_all_proc = pd.concat((df_tn_proc, df_tt_proc), axis=0, ignore_index=True)

for i in cat_f:
    d = pd.get_dummies(df_all_proc[i], prefix = i, prefix_sep='_')
    df_all_proc.drop(i, axis = 1, inplace = True)
    df_all_proc = df_all_proc.merge(d, right_index=True, left_index=True)

# prepare X and Y
X = df_all_proc[:df_tn.shape[0]].copy()
X_tt = df_all_proc[df_tn.shape[0]:].copy()
Y = df_tn['target'].copy()
print ("X shape", X.shape)
print ("X_tt shape", X_tt.shape)
print ("Y shape", Y.shape)
print ("")

del df_all_proc, df_tn_proc, df_tt_proc

# formula for Gini Coefficient (https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703)
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


# Custom LightGBM random search --------------------------------------------------------------

n_iterations = 7 # number of iterations for random search
top_n = 5 # select top n parameter sets

gini_mean = []
gini_std = []
roc_auc_mean = []
roc_auc_std = []
dict_list = []

# prepare indexes for stratified cross validation
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, Y)


# loop for random search

print ("Random search start...")
print ("")

for i in range(0, n_iterations):
    skf_split = skf.split(X, Y)
    param_dist = {'num_leaves': choice([27, 31, 61, 81, 127, 197, 231, 275, 302]),
              'bagging_fraction': choice([0.5, 0.7, 0.8, 0.9]),
              'learning_rate': choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]),
              'min_data': choice([300, 400, 450, 500, 550, 650]),
              'is_unbalance': choice([True, False]),
              'max_bin': choice([3, 5, 10, 12, 18, 20, 22]),
              'boosting_type' : choice(['gbdt', 'dart']),
              'bagging_freq': choice([3, 9, 11, 15, 17, 23, 31]),
              'max_depth': choice([3, 4, 5, 6, 7, 9, 11]),       
              'feature_fraction': choice([0.5, 0.7, 0.8, 0.9]),
              'lambda_l1': choice([0, 10, 20, 30, 40]),
              'objective': 'binary', 
              'metric': 'auc'} 
    
    gini_norm = []
    roc_l = []
    
    print ("Cycle {}...".format(i+1))
    for train_index, test_index in skf_split:
    
        X_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
    
        X_val = X.iloc[test_index]
        y_val = Y.iloc[test_index]
    
        # training
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=True)
    
        gbm = lgb.train(param_dist,
                        lgb_train,
                        num_boost_round = 10,
                        valid_sets = lgb_val,
                        early_stopping_rounds=5,
                        verbose_eval=5)
        # predicting
        y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        gn = gini_normalized(y_val, y_pred)
        gini_norm.append(gn)
    
        roc = roc_auc_score(y_val, y_pred)
        roc_l.append(roc)

    gini_norm_array = np.asarray(gini_norm)
    roc_array = np.asarray(roc_l)
    
    gini_mean.append(gini_norm_array.mean())
    gini_std.append(gini_norm_array.std())
    roc_auc_mean.append(roc_array.mean())
    roc_auc_std.append(roc_array.std())
    dict_list.append(param_dist)
    gc.collect()

results_pd = pd.DataFrame({"gini_mean": gini_mean,
                           "gini_std": gini_std,
                           "roc_auc_mean": roc_auc_mean,
                           "roc_auc_std": roc_auc_std,
                           "parameters": dict_list})    

results_pd.sort_values("gini_mean", ascending = False, axis = 0, inplace = True)
top_pd = results_pd.head(top_n)
for i in range(0, top_n):
    print ("Model with rank {}".format(i+1))
    print ("Mean gini score %.5f (std: %.5f)" % (top_pd['gini_mean'].values[i], top_pd['gini_std'].values[i]))
    print ("Mean roc_auc score %.5f (std: %.5f)" % (top_pd['roc_auc_mean'].values[i], top_pd['roc_auc_std'].values[i]))
    print ("Parameters:", top_pd['parameters'].values[i])
    print ("")


# train final lgbm model using best parameters

print ("Train 5 lgbm models...")
print ("")

prms_1 = top_pd['parameters'].values[0]
prms_2 = top_pd['parameters'].values[1]
prms_3 = top_pd['parameters'].values[2]
prms_4 = top_pd['parameters'].values[3]
prms_5 = top_pd['parameters'].values[4]

prms_list = [prms_1, prms_2, prms_3, prms_4, prms_5]
weights = [0.4, 0.4, 0.1, 0.05, 0.05]

pred_df = pd.DataFrame({"id": df_tt['id'].values})
target = np.zeros(df_tt.shape[0])

lgb_s = lgb.Dataset(X, Y)
for i in range(0, len(prms_list)):
    print ("Set {}".format(i))
    print ("training...")
    model = lgb.train(prms_list[i], 
                      lgb_s, 
                      num_boost_round = 10)
    print ("predicting...")
    y_pred = model.predict(X_tt)
    print ("arrays addition...")
    target = np.add(target, y_pred*weights[i])
    print ("done")
    print ("")
    gc.collect()
    
pred_df['target'] = target
pred_df.to_csv("lgbm_5m.csv", index = False)