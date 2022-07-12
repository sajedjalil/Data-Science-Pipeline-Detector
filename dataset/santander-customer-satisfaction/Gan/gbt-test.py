import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
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

col = ['ID', 'var3',
 'var15',
 'imp_ent_var16_ult1',
 'imp_op_var39_comer_ult1',
 'imp_op_var39_comer_ult3',
 'imp_op_var41_comer_ult1',
 'imp_op_var41_comer_ult3',
 'imp_op_var41_efect_ult1',
 'imp_op_var41_efect_ult3',
 'imp_op_var41_ult1',
 'imp_op_var39_efect_ult1',
 'imp_op_var39_efect_ult3',
 'imp_op_var39_ult1',
 'ind_var5',
 'ind_var8_0',
 'ind_var30_0',
 'ind_var30',
 'ind_var41_0',
 'num_var4',
 'num_var5_0',
 'num_var5',
 'num_var13_0',
 'num_op_var41_hace2',
 'num_op_var41_ult1',
 'num_op_var41_ult3',
 'num_op_var39_hace2',
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
 'saldo_var1',
 'saldo_var5',
 'saldo_var8',
 'saldo_var26',
 'saldo_var25',
 'saldo_var30',
 'saldo_var37',
 'saldo_var42',
 'var36',
 'imp_var43_emit_ult1',
 'imp_trans_var37_ult1',
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
 'var38'
 ]
test = test[col]
col.append('TARGET')
train = train[col]



model = GradientBoostingClassifier(
    n_estimators=70, learning_rate=0.1, min_samples_split=150, min_samples_leaf=200,
    max_depth=4,max_features=28, subsample=0.6, random_state=10
)
model.fit(train.iloc[:,1:-1], train['TARGET'])
predict_result = pd.DataFrame(test['ID'], columns=['ID'])
test.drop('ID', axis=1, inplace=True)
predict_y = model.predict_proba(test)[:,1]
predict_result['TARGET'] = predict_y
predict_result.to_csv('submission.csv', index=False)
