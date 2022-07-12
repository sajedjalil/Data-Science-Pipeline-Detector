import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
# read train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# remove constant columns
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

np.random.seed(10)
train = train.reindex(np.random.permutation(train.index),).reset_index(drop=True)

col = [
 'ID',
 'var3',
 'var15',
 'imp_ent_var16_ult1',
 'imp_op_var39_comer_ult1',
 'imp_op_var39_comer_ult3',
 'imp_op_var40_efect_ult1',
 'imp_op_var41_comer_ult1',
 'imp_op_var41_comer_ult3',
 'imp_op_var41_efect_ult1',
 'imp_op_var41_efect_ult3',
 'imp_op_var41_ult1',
 'imp_op_var39_efect_ult1',
 'imp_op_var39_efect_ult3',
 'imp_op_var39_ult1',
 'imp_sal_var16_ult1',
 'ind_var5_0',
 'ind_var5',
 'ind_var8_0',
 'ind_var12_0',
 'ind_var13',
 'ind_var30',
 'ind_var39_0',
 'ind_var41_0',
 'num_var4',
 'num_var5_0',
 'num_var5',
 'num_var8_0',
 'num_op_var41_hace2',
 'num_op_var41_hace3',
 'num_op_var41_ult1',
 'num_op_var41_ult3',
 'num_op_var39_hace2',
 'num_op_var39_hace3',
 'num_op_var39_ult1',
 'num_op_var39_ult3',
 'num_var30_0',
 'num_var30',
 'num_var35',
 'num_var37_med_ult2',
 'num_var37_0',
 'num_var39_0',
 'num_var41_0',
 'num_var42_0',
 'num_var42',
 'saldo_var5',
 'saldo_var8',
 'saldo_var12',
 'saldo_var26',
 'saldo_var25',
 'saldo_var30',
 'saldo_var37',
 'saldo_var42',
 'var36',
 'imp_var43_emit_ult1',
 'imp_trans_var37_ult1',
 'ind_var10cte_ult1',
 'ind_var9_cte_ult1',
 'num_ent_var16_ult1',
 'num_var22_hace2',
 'num_var22_hace3',
 'num_var22_ult1',
 'num_var22_ult3',
 'num_med_var22_ult3',
 'num_med_var45_ult3',
 'num_meses_var5_ult3',
 'num_meses_var39_vig_ult3',
 'num_op_var39_comer_ult1',
 'num_op_var39_comer_ult3',
 'num_op_var41_comer_ult1',
 'num_op_var41_comer_ult3',
 'num_op_var41_efect_ult1',
 'num_op_var41_efect_ult3',
 'num_op_var39_efect_ult1',
 'num_op_var39_efect_ult3',
 'num_var43_emit_ult1',
 'num_var43_recib_ult1',
 'num_var45_hace2',
 'num_var45_hace3',
 'num_var45_ult1',
 'num_var45_ult3',
 'saldo_medio_var5_hace2',
 'saldo_medio_var5_hace3',
 'saldo_medio_var5_ult1',
 'saldo_medio_var5_ult3',
 'saldo_medio_var8_hace2',
 'saldo_medio_var8_hace3',
 'saldo_medio_var8_ult1',
 'saldo_medio_var8_ult3',
 'saldo_medio_var12_ult1',
 'saldo_medio_var12_ult3',
 'saldo_medio_var13_corto_hace2',
 'saldo_medio_var13_corto_ult3',
 'var38'
 ]

test = test[col]
col.append('TARGET')
train = train[col]

features = col[1:-1]

unhappy = train.loc[train['TARGET']==1, features + ['TARGET']]
happy = train.loc[train['TARGET']==0, features+['TARGET']]
happy_index = 1500
unhappy_index = 30000
new_data = pd.concat([unhappy[0:happy_index],happy[0:unhappy_index]])
test_data = pd.concat([unhappy[happy_index:], happy[unhappy_index:]])

# model = xgb.XGBClassifier(
#     objective='binary:logistic', n_estimators=350, learning_rate=0.04, 
#     max_depth=5, nthread=4, subsample=0.7, colsample_bytree=0.5, 
#     reg_lambda=6, reg_alpha=5, seed=10, silent=True,
# )
# model.fit(train.iloc[:,1:-1], train['TARGET'])

new_model = xgb.XGBClassifier(
    objective='binary:logistic', n_estimators=321, learning_rate=0.04, 
    max_depth=5, nthread=4, subsample=0.7, colsample_bytree=0.5, 
    reg_lambda=6, reg_alpha=5, seed=10, silent=True
)
new_model.fit(
    new_data[features].values, new_data['TARGET'], eval_metric="auc",  early_stopping_rounds=30,  
    eval_set=[(test_data[features].values, test_data['TARGET'])], verbose=False
)

new_model2 = xgb.XGBClassifier(
    objective='binary:logistic', n_estimators=290, learning_rate=0.04, 
    max_depth=5, nthread=4, subsample=0.7, colsample_bytree=0.5, 
    reg_lambda=6, reg_alpha=5, seed=10, silent=True
)

new_model2.fit(
    test_data[features].values, test_data['TARGET'], eval_metric="auc",  early_stopping_rounds=30,  
    eval_set=[(new_data[features].values, new_data['TARGET'])], verbose=False)

predict_result = pd.DataFrame(test['ID'], columns=['ID'])
test.drop('ID', axis=1, inplace=True)
predict_y = (new_model.predict_proba(test)[:,1] + new_model2.predict_proba(test)[:,1])/2
test['TARGET'] = predict_y

# test.loc[test['var15'] < 23, 'TARGET'] = 0
# test.loc[test['saldo_medio_var5_hace2'] > 160000, 'TARGET'] = 0
# test.loc[test['var38'] > 3988596, 'TARGET'] = 0

predict_result['TARGET'] = test['TARGET'].copy()
predict_result.to_csv('submission.csv', index=False)