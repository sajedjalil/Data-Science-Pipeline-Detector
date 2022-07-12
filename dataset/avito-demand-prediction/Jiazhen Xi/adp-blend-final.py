import warnings
warnings.filterwarnings("ignore")
import os
import gc
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
os.environ['OMP_NUM_THREADS'] = '4'

DEV = True

import sys
UTILS_PATH = '../input/myutils/'
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

import adp_utils as au
eval_sets, y, train_num = au.load_eval_sets_lables()

paths = list(filter(
    lambda p: p.endswith('result') or p.endswith('output'), 
    os.listdir('../input/')
))
paths = sorted(paths)
print('\n'.join(paths))
csv_li = []
name_li = []
score_li = []
for p in paths:
    res = list(filter(
        lambda s: s.endswith('.csv'),
        os.listdir('../input/{}'.format(p))
    ))
    names = [s.split('_')[0] for s in res]
    scores = [float(s.split('_')[-1].split('.csv')[0]) for s in res]
    res = ['../input/{}/{}'.format(p, s) for s in res]
    name_li += names
    score_li += scores
    csv_li += res
dfs = []
for p, name in zip(csv_li, name_li):
    res = pd.read_csv(p)
    res.columns = ['_'.join(name.split('-'))]
    dfs.append(res)
dfs = pd.concat(dfs, axis=1)
# bf = set(dfs.columns.tolist())
# del_cond = lambda c: 'strange' in c
# dfs = au.delete_cols(dfs, [], del_cond)
# aft = set(dfs.columns.tolist())
# print(bf-aft, 'deleted')
feature_name = dfs.columns.tolist()
preds_val = dfs[:train_num]
preds_test = dfs[train_num:]
preds_val = preds_val.values
preds_test = preds_test.values
print('train', preds_val.shape)
print('test', preds_test.shape)
del dfs; gc.collect()
y_valid = y.copy()
data = pd.DataFrame(preds_val, columns=feature_name)
data['y_valid'] = y_valid.copy()
df_corr = data.sort_index().corr()
del data; gc.collect();
max_num_features = min(len(feature_name), 300)
f, ax = plt.subplots(figsize=[max_num_features//3, max_num_features//3-3])
sns.heatmap(df_corr)
plt.savefig('feature-corr.png')
del df_corr; gc.collect();

###
# Use: preds_val, preds_test, y_valid
###
X, y, X_test = preds_val.copy(), y_valid.copy(), preds_test.copy()
del preds_val, preds_test, y_valid; gc.collect();

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge, SGDRegressor, BayesianRidge

lgb_pred = np.zeros((X_test.shape[0],))
rg_pred = np.zeros((X_test.shape[0],))
ls_pred = np.zeros((X_test.shape[0],))

lgb_pred_val = np.zeros((X.shape[0],))
rg_pred_val = np.zeros((X.shape[0],))
ls_pred_val = np.zeros((X.shape[0],))

lgb_cv_scores = []
rg_cv_scores = []
ls_cv_scores = []

lgb_feat_imps = []
ridge_coefs = []

valid_fold = 0
for valid_fold in range(10):
    mask_te = eval_sets==valid_fold
    mask_tr = ~mask_te
    print('[level 1] processing fold %d...'%valid_fold)
    ridge = Ridge(alpha=23)
    ridge.fit(X[mask_tr], y[mask_tr])
    rg_pred_val[mask_te] = ridge.predict(X[mask_te])
    scr = np.sqrt(mean_squared_error(y[mask_te], rg_pred_val[mask_te]))
    print('ridge rmse:', scr)
    rg_pred += ridge.predict(X_test)/10
    ridge_coefs.append(ridge.coef_)
    rg_cv_scores.append(scr)

valid_fold = 0
for valid_fold in range(10):
    mask_te = eval_sets==valid_fold
    mask_tr = ~mask_te
    print('[level 1] processing fold %d...'%valid_fold)
    ls = SGDRegressor(loss='squared_loss', penalty='l1', l1_ratio=0.0)
    ls.fit(X[mask_tr], y[mask_tr])
    ls_pred_val[mask_te] = ls.predict(X[mask_te])
    scr = np.sqrt(mean_squared_error(y[mask_te], ls_pred_val[mask_te]))
    print('sgd rmse:', scr)
    ls_pred += ls.predict(X_test)/10
    ls_cv_scores.append(scr)

lgb_params =  {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 
    'metric': 'rmse', 
    'num_threads': 4, 
    #'max_bin': 255//2, 
    'min_data_in_leaf': 100, #20
    'max_depth': 4, #10, 
    'seed': 23333333,
    #'two_round': True,
    #'num_leaves': 2**( 10 - 1 ) - 1, #256+511
    'feature_fraction': 0.75, #0.7
    'bagging_fraction': 0.75, #0.7
    'bagging_freq': 3, #4
    'learning_rate': 0.02, #0.016
    'verbose': -1
}
num_boost_round = 3 if not DEV else 20000
early_stopping_rounds = 200
verbose_eval = 10000
valid_fold = 0
for valid_fold in range(10):
    mask_te = eval_sets==valid_fold
    mask_tr = ~mask_te
    print('[level 1] processing fold %d...'%valid_fold)
    with au.timer('lgb fit & predict'):
        dtrain = lgb.Dataset(
            X[mask_tr], y[mask_tr],
            feature_name=feature_name,
            free_raw_data=False
        )
        dvalid = lgb.Dataset(
            X[mask_te], y[mask_te],
            feature_name=feature_name,
            free_raw_data=False
        )
        evals_result = {}
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=['train','valid'],
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds, #50
            verbose_eval=verbose_eval
        )
        lgb_pred_val[mask_te] = model.predict(X[mask_te])
        lgb_pred += model.predict(X_test)/10
        lgb_feat_imps.append(model.feature_importance()/model.best_iteration)
        scr = evals_result['valid']['rmse'][model.best_iteration-1]
        print('lgb rmse:', scr)
        lgb_cv_scores.append(scr)

xgb_params = {
    'objective': 'reg:linear', 
    'eval_metric': 'rmse', 
    'booster': 'gblinear',
    'reg_lambda': 0,
    'reg_alpha': 0,
    'updater': 'shotgun',
    'eta': 0.2
}
num_boost_round = 3 if not DEV else 1000
early_stopping_rounds = 50
verbose_eval = 10000

xgb_pred_val = np.zeros((X.shape[0],))
xgb_pred = np.zeros((X_test.shape[0],))
xgb_cv_scores = []

valid_fold = 0
l1_preds_val = [rg_pred_val, ls_pred_val, lgb_pred_val]
l1_preds = [rg_pred, ls_pred, lgb_pred]
for valid_fold in range(10):
    mask_te = eval_sets==valid_fold
    mask_tr = ~mask_te
    print('[level 2] processing fold %d...'%valid_fold)
    with au.timer('gbl fit & predict'):
        dtrain = xgb.DMatrix(np.vstack([p[mask_tr] for p in l1_preds_val]).T, y[mask_tr])
        dvalid = xgb.DMatrix(np.vstack([p[mask_te] for p in l1_preds_val]).T, y[mask_te])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        evals_result = {}
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            maximize=False,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        xgb_pred_val[mask_te] = model.predict(dvalid)
        scr = np.sqrt(mean_squared_error(y[mask_te], xgb_pred_val[mask_te]))
        print('gbl rmse:', scr)
        dtest = xgb.DMatrix(np.vstack([p for p in l1_preds]).T)
        xgb_pred += model.predict(dtest)/10
        xgb_cv_scores.append(scr)
        
def get_mean_score(name, scores):
    cv_scores = np.array(scores)
    print(f'{name} scores', cv_scores)
    print(f'{name} worst', cv_scores.max())
    score = np.mean(cv_scores)
    print(f'{name} mean', score)
    return score

lgb_score = get_mean_score('lgb', lgb_cv_scores)
rg_score = get_mean_score('ridge', rg_cv_scores)
ls_score = get_mean_score('sgd', ls_cv_scores)
xgb_score = get_mean_score('xgb', xgb_cv_scores)

ridge_coefs = pd.DataFrame(
    np.vstack(ridge_coefs).T, 
    columns=['fold_{}'.format(i) for i in range(10)],
    index=feature_name,
)
ridge_coefs.to_csv('ridge_imps.csv', index=False)
lgb_imps = pd.DataFrame(
    np.vstack(lgb_feat_imps).T, 
    columns=['fold_{}'.format(i) for i in range(10)],
    index=feature_name,
)
lgb_imps.to_csv('lgb_imps.csv', index=False)

max_num_features = len(feature_name)
f, ax = plt.subplots(figsize=[8, max_num_features//5])
data = ridge_coefs.copy()
data_mean = data.mean(1).sort_values()
data = data.loc[data_mean.index]
data_index = data.index.copy()
data = [data[c].values for c in data.columns]
data = np.hstack(data)
data = pd.DataFrame(data, index=data_index.tolist()*10, columns=['rg_coef'])
data = data.reset_index()
data.columns = ['feature_name', 'rg_coef']
sns.barplot(x='rg_coef', y='feature_name', data=data, orient='h', ax=ax)
plt.grid()
plt.savefig('rg-bld-imp.png')

max_num_features = len(feature_name)
f, ax = plt.subplots(figsize=[8, max_num_features//5])
data = lgb_imps.copy()
data_mean = data.mean(1).sort_values(ascending=False)
data = data.loc[data_mean.index]
data_index = data.index.copy()
data = [data[c].values for c in data.columns]
data = np.hstack(data)
data = pd.DataFrame(data, index=data_index.tolist()*10, columns=['igb_imp'])
data = data.reset_index()
data.columns = ['feature_name', 'igb_imp']
sns.barplot(x='igb_imp', y='feature_name', data=data, orient='h', ax=ax)
plt.grid()
plt.savefig('lgb-bld-imp.png')

print('lgb mean', np.mean(lgb_pred))
print('ridge mean', np.mean(rg_pred))
print('sgd mean', np.mean(ls_pred))
print('xgb mean', np.mean(xgb_pred))
print('labels mean', np.mean(y))

def save_pred(pred, save_name):
    sub = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
    sub['deal_probability'] = pred
    sub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
    sub.to_csv("{}.csv".format(save_name), index=False)
    print("{}.csv".format(save_name), 'saved!')

pred_li = [
    lgb_pred, 
    rg_pred,
    ls_pred,
    xgb_pred,
]   

name_li = [
    'bld_lgb_{}'.format(lgb_score),
    'bld_rg_{}'.format(rg_score),
    'bld_ls_{}'.format(ls_score),
    'bld_xgb_{}'.format(xgb_score),
]

for pred, name in zip(pred_li, name_li):
    save_pred(pred, name)










