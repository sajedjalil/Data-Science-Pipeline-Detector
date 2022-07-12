import numpy as np
import xgboost as xgb
import pandas as pd
import math

from sklearn.cross_validation import train_test_split
from ml_metrics import rmsle

print ('')
print ('Loading Data...')

def evalerror(preds, dtrain):

    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    labels = labels.tolist()
    preds = preds.tolist()
    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]
    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5

types = {'Semana': np.uint8, 'Agencia_ID': np.uint16, 'Canal_ID': np.uint8,
'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32, 'Producto_ID': np.uint16,
'Demanda_uni_equil': np.uint32}
types_test = {'id': np.uint32,'Semana': np.uint8, 'Agencia_ID': np.uint16, 'Canal_ID': np.uint8,
'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32, 'Producto_ID': np.uint16}
train = pd.read_csv('../input/train.csv', nrows = 11000000,usecols=types.keys(), dtype=types)
test = pd.read_csv('../input/test.csv',usecols=types_test.keys(), dtype=types_test)

print ('')
print ('Training_Shape:', train.shape)

ids = test['id']
test = test.drop(['id'],axis = 1)

y = train['Demanda_uni_equil']
X = train[test.columns.values]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)

print ('Division_Set_Shapes:', X.shape, y.shape)
print ('Validation_Set_Shapes:', X_train.shape, X_test.shape)

params = {}
params['objective'] = "reg:linear"
#params['booster']="gblinear"  #default gbtree
params['eta'] = 0.015
params['max_depth'] = 5
params['subsample'] = 0.8
params['colsample_bytree'] = 0.6
params['silent'] = True

print ('')

test_preds = np.zeros(test.shape[0])
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

watchlist = [(xg_train, 'train')]
num_rounds = 100

xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)
preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)

print ('RMSLE Score:', rmsle(y_test, preds))

fxg_test = xgb.DMatrix(test)
fold_preds = np.around(xgclassifier.predict(fxg_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)
test_preds += fold_preds

submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': test_preds})
submission.to_csv('submission.csv', index=False)