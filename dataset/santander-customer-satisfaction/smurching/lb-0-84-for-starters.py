# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

remove = ["imp_op_var39_comer_ult1", "imp_op_var39_comer_ult3", "imp_op_var40_efect_ult1", "imp_op_var40_efect_ult3", "imp_op_var40_ult1", "ind_var1_0", "ind_var1", "ind_var6_0", "ind_var6", "ind_var8", "ind_var13_medio_0", "ind_var18_0", "ind_var20_0", "ind_var20", "ind_var25_cte", "ind_var34_0", "ind_var40", "ind_var44_0", "ind_var44", "num_var1_0", "num_var1", "num_var5_0", "num_var5", "num_var6", "num_var12_0", "num_var12", "num_var13_0", "num_var13_corto_0", "num_var13_corto", "num_var13_medio_0", "num_var17_0", "num_var17", "num_var18_0", "num_var26_0", "num_op_var40_hace2", "num_op_var40_hace3", "num_op_var40_ult1", "num_op_var40_ult3", "num_op_var41_hace2", "num_op_var39_hace2", "num_var34_0", "saldo_var1", "saldo_var5", "saldo_var6", "saldo_var12", "saldo_var13_corto", "saldo_var13_medio", "saldo_var13", "saldo_var17", "saldo_var18", "saldo_var26", "saldo_var34", "imp_reemb_var17_hace3", "imp_reemb_var33_ult1", "imp_trasp_var33_out_ult1", "ind_var7_emit_ult1", "num_var22_hace2", "num_meses_var13_medio_ult3", "num_op_var39_comer_ult1", "num_op_var39_comer_ult3", "num_op_var40_efect_ult1", "num_op_var40_efect_ult3", "num_reemb_var13_ult1", "num_reemb_var33_ult1", "num_trasp_var17_in_ult1", "num_trasp_var17_out_ult1", "num_trasp_var33_in_ult1", "num_trasp_var33_out_ult1", "num_var45_hace2", "ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28", "ind_var27", "ind_var41", "ind_var46_0", "ind_var46", "num_var27_0", "num_var28_0", "num_var28", "num_var27", "num_var41", "num_var46_0", "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46", "imp_amort_var18_hace3", "imp_amort_var34_hace3", "imp_reemb_var13_hace3", "imp_reemb_var33_hace3", "imp_trasp_var17_out_hace3", "imp_trasp_var33_out_hace3", "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3", "num_reemb_var33_hace3", "num_trasp_var17_out_hace3", "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3", "ind_var29_0", "ind_var29", "ind_var13_medio", "ind_var18", "ind_var26", "ind_var25", "ind_var32", "ind_var34", "ind_var37", "ind_var39", "num_var29_0", "num_var29", "num_var13_medio", "num_var18", "num_var26", "num_var25", "num_var32", "num_var34", "num_var37", "num_var39", "saldo_var29", "saldo_medio_var13_medio_ult1", "delta_num_reemb_var13_1y3", "delta_num_reemb_var17_1y3", "delta_num_reemb_var33_1y3", "delta_num_trasp_var17_in_1y3", "delta_num_trasp_var17_out_1y3", "delta_num_trasp_var33_in_1y3", "delta_num_trasp_var33_out_1y3"]


df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

print("Dataset has %d features"%(len(df_train.columns)))


y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# weight positive examples
# num_ones = sum(y_train)
# weight = float(len_train - num_ones) / num_ones
# print("Weight of positive examples: %s"%weight)

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=3, n_estimators=2000,
    learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)


# fitting
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

# predicting
y_pred= clf.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

print('Completed!')
                