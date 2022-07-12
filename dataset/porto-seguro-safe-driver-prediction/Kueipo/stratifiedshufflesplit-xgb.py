import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


y = train.target.values
id_test = test['id'].values

train = train.drop(['id','target'], axis=1)
test = test.drop(['id'], axis=1)

################################################
drop = [col for col in train.columns if 'calc' in col]

train.drop(drop,axis=1,inplace=True)
test.drop(drop,axis=1,inplace=True)

################################################


def ginic(actual, pred):
    actual = np.asarray(actual) 
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:,1] 
    return ginic(a, p) / ginic(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score
    
X = train.values

# Set xgb parameters

params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.03
params['silent'] = True
params['max_depth'] = 5
params['subsample'] = 0.9
params['min_child_weight'] = 10
params['colsample_bytree'] = 0.9
params['colsample_bylevel'] = 0.9

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = np.zeros_like(id_test)


kfold = 4
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=9487)
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=200)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test)
    sub['target'] += p_test/kfold

sub.to_csv('FattyKimJungUn.csv', index=False)
