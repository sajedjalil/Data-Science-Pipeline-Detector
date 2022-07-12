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

# features = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult3', 
# 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 
# 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_ult1', 'ind_var8_0', 
# 'ind_var30_0', 'ind_var30', 'num_op_var41_hace2', 'num_op_var41_ult3', 
# 'num_var37_med_ult2', 'saldo_var5', 'saldo_var8', 'saldo_var26', 'saldo_var30', 
# 'saldo_var37', 'saldo_var42', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 
# 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3',
# 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3',
# 'num_op_var39_comer_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1',
# 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1',
# 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 
# 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2',
# 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult3',
# 'saldo_medio_var13_corto_hace2', 'var38', 'n0']

features = ['var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 
'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 
'imp_op_var41_ult1', 'imp_op_var39_ult1', 'ind_var5', 'ind_var8_0', 'ind_var26_cte', 'ind_var30_0', 'ind_var31_0',
'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'saldo_var5', 'saldo_var8',
'saldo_var30', 'saldo_var37', 'saldo_var42', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'num_var22_hace2', 
'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 
'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_efect_ult1',
'num_op_var41_efect_ult3', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3',
'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1',
'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'var38', 'n0']


X_sel = X[features]

test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]   

y_pred_list = []
for fold in range(64):
    print ('Fold {}'.format(fold))
    X_train, X_test, y_train, y_test = \
      cross_validation.train_test_split(X_sel, y, random_state=1301+fold, stratify=y, test_size=0.2+fold*0.01)
    
    clf = xgb.XGBClassifier(missing=9999999999,
                    max_depth = 8,
                    n_estimators=1000,
                    learning_rate=0.05, 
                    nthread=4,
                    subsample=0.8,
                    colsample_bytree=0.5,
                    min_child_weight = 7,
                    seed=1301+fold)
    clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc",
            eval_set=[(X_train, y_train), (X_test, y_test)])
            
    print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))
    
     
    y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)
    y_pred_list.append(y_pred)
    
pred = np.mean(np.array(y_pred_list), axis = 0)

submission = pd.DataFrame({"ID":test.index, "TARGET":pred[:,1]})
submission.to_csv("submission.csv", index=False)

       
       