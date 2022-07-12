
import sys
import getopt
from sklearn import feature_extraction
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.metrics import log_loss
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler #StandardScaler
from sklearn.cross_validation import StratifiedKFold



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

## feature remove from the train dataset
del df_train['ind_var2_0']
del df_train['ind_var27_0']
del df_train['ind_var2']
del df_train['ind_var28_0']
del df_train['ind_var28']
del df_train['ind_var27']
del df_train['ind_var41']
del df_train['ind_var46_0']
del df_train['ind_var46']
del df_train['num_var27_0']
del df_train['num_var28_0']
del df_train['num_var28']
del df_train['num_var27']
del df_train['num_var41']
del df_train['num_var46_0']
del df_train['num_var46']
del df_train['saldo_var28']
del df_train['saldo_var27']
del df_train['saldo_var41']
del df_train['saldo_var46']
del df_train['imp_amort_var34_hace3']
del df_train['imp_amort_var18_hace3']
del df_train['imp_reemb_var13_hace3']
del df_train['imp_reemb_var33_hace3']
del df_train['imp_trasp_var17_out_hace3']
del df_train['imp_trasp_var33_out_hace3']
del df_train['num_var2_0_ult1']
del df_train['num_var2_ult1']
del df_train['num_reemb_var13_hace3']
del df_train['num_reemb_var33_hace3']
del df_train['num_trasp_var17_out_hace3']
del df_train['num_trasp_var33_out_hace3']
del df_train['saldo_var2_ult1']
del df_train['saldo_medio_var13_medio_hace3']
del df_train['ind_var6_0']
del df_train['ind_var6']


## feature remove from the test dataet
del df_test['ind_var2_0']
del df_test['ind_var27_0']
del df_test['ind_var2']
del df_test['ind_var28_0']
del df_test['ind_var28']
del df_test['ind_var27']
del df_test['ind_var41']
del df_test['ind_var46_0']
del df_test['ind_var46']
del df_test['num_var27_0']
del df_test['num_var28_0']
del df_test['num_var28']
del df_test['num_var27']
del df_test['num_var41']
del df_test['num_var46_0']
del df_test['num_var46']
del df_test['saldo_var28']
del df_test['saldo_var27']
del df_test['saldo_var41']
del df_test['saldo_var46']
del df_test['imp_amort_var34_hace3']
del df_test['imp_amort_var18_hace3']
del df_test['imp_reemb_var13_hace3']
del df_test['imp_reemb_var33_hace3']
del df_test['imp_trasp_var17_out_hace3']
del df_test['imp_trasp_var33_out_hace3']
del df_test['num_var2_0_ult1']
del df_test['num_var2_ult1']
del df_test['num_reemb_var13_hace3']
del df_test['num_reemb_var33_hace3']
del df_test['num_trasp_var17_out_hace3']
del df_test['num_trasp_var33_out_hace3']
del df_test['saldo_var2_ult1']
del df_test['saldo_medio_var13_medio_hace3']
del df_test['ind_var6_0']
del df_test['ind_var6']

# first feature set to remove with idenditi feature from train
del df_train['ind_var13_medio_0']
del df_train['ind_var18_0']
del df_train['ind_var26_0']
del df_train['ind_var25_0']
del df_train['ind_var32_0']
del df_train['ind_var34_0']
del df_train['ind_var37_0']
del df_train['ind_var40']
del df_train['num_var6_0']
del df_train['num_var6']
del df_train['num_var13_medio_0']
del df_train['num_var18_0']
del df_train['num_var26_0']
del df_train['num_var25_0']
del df_train['num_var32_0']
del df_train['num_var34_0']
del df_train['num_var37_0']
del df_train['num_var40']
del df_train['saldo_var6']
del df_train['saldo_var13_medio']
del df_train['delta_num_reemb_var13_1y3']
del df_train['delta_num_reemb_var17_1y3']
del df_train['delta_imp_reemb_var33_1y3']
del df_train['delta_imp_trasp_var17_out_1y3']
del df_train['delta_imp_trasp_var17_in_1y3']
del df_train['delta_imp_trasp_var33_in_1y3']
del df_train['delta_imp_trasp_var33_out_1y3']

# first identity feature to be remove from test
del df_test['ind_var13_medio_0']
del df_test['ind_var18_0']
del df_test['ind_var26_0']
del df_test['ind_var25_0']
del df_test['ind_var32_0']
del df_test['ind_var34_0']
del df_test['ind_var37_0']
del df_test['ind_var40']
del df_test['num_var6_0']
del df_test['num_var6']
del df_test['num_var13_medio_0']
del df_test['num_var18_0']
del df_test['num_var26_0']
del df_test['num_var25_0']
del df_test['num_var32_0']
del df_test['num_var34_0']
del df_test['num_var37_0']
del df_test['num_var40']
del df_test['saldo_var6']
del df_test['saldo_var13_medio']
del df_test['delta_num_reemb_var13_1y3']
del df_test['delta_num_reemb_var17_1y3']
del df_test['delta_imp_reemb_var33_1y3']
del df_test['delta_imp_trasp_var17_out_1y3']
del df_test['delta_imp_trasp_var17_in_1y3']
del df_test['delta_imp_trasp_var33_in_1y3']
del df_test['delta_imp_trasp_var33_out_1y3']


y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)


xgboost_params = { 
   "bst:max_depth":8,
   "bst:eta":.1,
   "nthread":8,
   "objective": "binary:logistic",
   "booster": "gbtree",
   "eval_metric": "auc",
   "eta": 0.01, # 0.06, #0.01,
   "subsample": 0.65,
   "colsample_bytree": 0.9,
   "learning_rate":0.03,
   "n_estimators":250,
   "max_depth": 7,
   # "nthread":4
}

num_round = 900

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 


xgtrain = xgb.DMatrix( X_train, label =  y_train)
xgtest = xgb.DMatrix(X_test)


print('Fit the model...')
boost_round = 250 #CHANGE THIS VALUE
clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)


nCV = 7
rng = np.random.RandomState(337)
kf = StratifiedKFold(y_train, n_folds=nCV, shuffle=True, random_state=rng) 
cv_preds = np.array([0.0] * X_train.shape[0])
i = 0

nBagging = 5
bagging_preds = np.array([0.0] * X_test.shape[0])
for i in range(nBagging):
	y_pred = clf.predict(xgtest, ntree_limit=clf.best_iteration)
	bagging_preds = bagging_preds + y_pred

submission = pd.DataFrame({"ID":id_test, "TARGET":bagging_preds})
submission.to_csv("Submission.csv", index=False)
