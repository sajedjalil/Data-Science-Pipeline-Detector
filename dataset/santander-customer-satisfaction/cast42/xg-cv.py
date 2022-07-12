import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

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

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

# Select the features calculated in https://www.kaggle.com/cast42/santander-customer-satisfaction/select-features-rfecv/code
# 
features = \
['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 
'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 
'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var8_0', 'ind_var30', 'num_var4',
'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_ult1', 
'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var37', 
'num_var39_0', 'num_var42', 'saldo_var5', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 
'imp_trans_var37_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 
'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 
'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1',
'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult3', 
'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',
'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'var38', 'n0']

X_sel = X[features]

test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]   

xgtrain = xgb.DMatrix(X_sel, label=y)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 7,
                n_estimators=700,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                seed=1301)
xgb_param = clf.get_xgb_params()
#do cross validation
print ('Start cross validation')
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, nfold=15, metrics=['auc'],
     early_stopping_rounds=50, stratified=True, seed=1301)
print('Best number of trees = {}'.format(cvresult.shape[0]))
clf.set_params(n_estimators=cvresult.shape[0])
print('Fit on the trainingsdata')
clf.fit(X_sel, y, eval_metric='auc')
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))
print('Predict the probabilities based on features in the test set')
pred = clf.predict_proba(sel_test, ntree_limit=cvresult.shape[0])

submission = pd.DataFrame({"ID":test.index, "TARGET":pred[:,1]})
submission.to_csv("submission.csv", index=False)

       
       