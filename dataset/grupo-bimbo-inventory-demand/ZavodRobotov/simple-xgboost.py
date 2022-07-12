import numpy as np
import xgboost as xgb
import pandas as pd
import math

from sklearn.cross_validation import train_test_split
from ml_metrics import rmsle

print ('')
print ('Loading Data...')

def same_amount(dem,ret):
    return max(0, dem - ret)

def rssample(df):
  #  mask = df.val1 == 0
   # if np.all(mask):
    #    return None
    #else:
    idx1 = mask.idxmin()
    idx0 = np.random.choice(mask[mask].index)
    return df.loc[[idx0, idx1]]

def evalerror(preds, dtrain):

    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    labels = labels.tolist()
    preds = preds.tolist()
    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]
    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5

train = pd.read_csv('../input/train.csv', nrows = 7000, sep=',', encoding='utf-8-sig')
train_ss = train.sample(n=100)
#train_ss = rssample(train)
print('a')
train_ss.to_csv('inp.csv', index=False, sep=',', encoding='utf-8-sig')
print('b')
test = pd.read_csv('../input/test.csv')

test_ss = test.sample(n=100)
test_ss.to_csv('t.csv', index=False, sep=',', encoding='utf-8-sig')

print('b1')
preds1 = np.zeros(test.shape[0])
print('b2')

print(train.head(30))
print(test.head(30))
#for idx in test.index:
 #  test.ix[idx]['Venta_uni_hoy'] -test.ix[idx]['Dev_uni_proxima']

test.transpose()
vuh = test['Venta_uni_hoy']
dup = test['Dev_uni_proxima']
preds1 = vuh - dup

#for index, row in test.iterrows():
    #row[1] - row[2]
    #row['Venta_uni_hoy'] - row['Dev_uni_proxima']
    
    #preds1[index] = max(0, row['Venta_uni_hoy'] - row['Dev_uni_proxima'])
    
#preds1 = test.apply(lambda row: same_amount(row['Venta_uni_hoy'], row['Dev_uni_proxima']),axis = 1)
print('b3')
submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': preds1})
print('b4')
submission.to_csv('submission1.csv', index=False)
print('b5')

#print(train.head(30))

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
params['eta'] = 0.025
params['max_depth'] = 5
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8
params['silent'] = True

print ('')

test_preds = np.zeros(test.shape[0])
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

watchlist = [(xg_train, 'train')]
num_rounds = 10

xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)
preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)

print ('RMSLE Score:', rmsle(y_test, preds))

fxg_test = xgb.DMatrix(test)
fold_preds = np.around(xgclassifier.predict(fxg_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)
test_preds += fold_preds

submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': test_preds})
submission.to_csv('submission.csv', index=False)