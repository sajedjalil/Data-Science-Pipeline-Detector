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

########################################################################
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble

print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['TARGET'].values
train = train.drop(['ID', 'TARGET', 'ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var6_0', 'ind_var6', 'ind_var13_medio_0', 'ind_var18_0', 'ind_var26_0', 'ind_var25_0', 'ind_var32_0', 'ind_var34_0', 'ind_var37_0', 'ind_var40', 'num_var6_0', 'num_var6', 'num_var13_medio_0', 'num_var18_0', 'num_var26_0', 'num_var25_0', 'num_var32_0', 'num_var34_0', 'num_var37_0', 'num_var40', 'saldo_var6', 'saldo_var13_medio', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3', 'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3'],axis=1)
test = pd.read_csv("../input/test.csv")
id_test = test['ID'].values
test = test.drop(['ID', 'ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var6_0', 'ind_var6', 'ind_var13_medio_0', 'ind_var18_0', 'ind_var26_0', 'ind_var25_0', 'ind_var32_0', 'ind_var34_0', 'ind_var37_0', 'ind_var40', 'num_var6_0', 'num_var6', 'num_var13_medio_0', 'num_var18_0', 'num_var26_0', 'num_var25_0', 'num_var32_0', 'num_var34_0', 'num_var37_0', 'num_var40', 'saldo_var6', 'saldo_var13_medio', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3', 'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3'],axis=1)

train['n0'] = (train == 0).sum(axis=1)
test['n0'] = (test == 0).sum(axis=1)

X_train = train
X_test = test
print('Training...')
extc = ExtraTreesClassifier(n_estimators=2000,max_features= 60,criterion= 'entropy',min_samples_split= 4,
                            max_depth= 35, min_samples_leaf= 2, n_jobs = -1)      

extc.fit(X_train,target) 

print('Predict...')
y_pred = extc.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": id_test, "TARGET": y_pred[:,1]}).to_csv('extra_trees.csv',index=False)