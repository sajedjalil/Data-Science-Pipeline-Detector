import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn.cross_validation import cross_val_score

np.random.seed(8)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
target = train['TARGET']
train = train.drop(['ID','TARGET'],axis=1)
test = test.drop('ID',axis=1)

print('Percentile 25:', np.percentile(train.var15, 25))
print('Percentile 50:', np.percentile(train.var15, 50))
print('Percentile 75:', np.percentile(train.var15, 75))

train['var15_1'] = train['var15'].map(lambda x: x if x < 23 else 23)
test['var15_1'] = test['var15'].map(lambda x: x if x < 23 else 23)

train['var15_2'] = train['var15'].map(lambda x: x if x < 28 else 28).map(lambda x: x if x > 23 else 23)
test['var15_2'] = test['var15'].map(lambda x: x if x < 28 else 28).map(lambda x: x if x > 23 else 23)
train['var15_3'] = train['var15'].map(lambda x: x if x > 40 else 40)
test['var15_3'] = test['var15'].map(lambda x: x if x > 40 else 40)

selected_features = ['saldo_var30', 'var15', 'num_var30', 'ind_var8_0',
'num_med_var22_ult3', 'ind_var30_0', 'imp_op_var39_efect_ult1',
'var38', 'ind_var30', 'num_var5', 'saldo_var5', 'num_var22_ult1',
'num_meses_var5_ult3', 'num_var43_recib_ult1', 'num_var14',
'ind_var31_0', 'saldo_medio_var8_hace2', 'saldo_var26',
'var15_1', 'var15_2', 'var15_3']

len(selected_features)

X_train = np.array(train[selected_features])
y_train = target
score = cross_val_score(linear_model.LinearRegression(),
                        X_train, y_train, cv=10, scoring='roc_auc')
print('Score:',score)
print('AUC',score.mean(),'+',score.std())