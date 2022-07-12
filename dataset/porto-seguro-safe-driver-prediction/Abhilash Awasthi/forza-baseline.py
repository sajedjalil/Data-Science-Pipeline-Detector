import numpy as np
import pandas as pd
from sklearn import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, SGDClassifier
import xgboost as xgb

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
col = [c for c in train.columns if c not in ['id','target']]

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_xgb(preds, y):
    y = y.get_label()
    return 'gini', gini(y, preds) / gini(y, y)
    
# Feature Selection by Lasso
print('Running Lasso..')
scaler = StandardScaler()
std_data = scaler.fit_transform(train[col].values)
clf = LogisticRegression(penalty='l1', C=0.1, random_state=42, solver='liblinear', n_jobs=1)
clf.fit(std_data, train['target'].values.reshape((-1,)))
imp_feats_ind = np.nonzero(clf.coef_[0])[0]
final_feats = np.array(col)[imp_feats_ind]
print('Lasso Completed!')
print('Total features selected are:', len(final_feats))
print('Features Selected:', final_feats)

# XGBoost
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train[final_feats], train['target'], test_size=0.25, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=10, early_stopping_rounds=100)
test['target'] = model.predict(xgb.DMatrix(test[final_feats]), ntree_limit=model.best_ntree_limit+50)

test[['id','target']].to_csv('submission.csv', index=False, float_format='%.5f')