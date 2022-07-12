import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
import lightgbm as lgb
from multiprocessing import *

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
col = [c for c in train.columns if c not in ['id','target']]
print(len(col))
col = [c for c in col if not c.startswith('ps_calc_')]
print(len(col))

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)

params = {'eta': 0.02, 'max_depth': 4, 'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 0.77, 'scale_pos_weight': 1.6, 'gamma': 10, 'reg_alpha': 8, 'reg_lambda': 1.3, 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

x1 = multi_transform(x1)
x2 = multi_transform(x2)
test = multi_transform(test)

col = [c for c in x1.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
print(x1.values.shape, x2.values.shape)

#remove duplicates just in case
tdups = multi_transform(train)
dups = tdups[tdups.duplicated(subset=col, keep=False)]

x1 = x1[~(x1['id'].isin(dups['id'].values))]
x2 = x2[~(x2['id'].isin(dups['id'].values))]
print(x1.values.shape, x2.values.shape)

y1 = x1['target']
y2 = x2['target']
x1 = x1[col]
x2 = x2[col]

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=200)
test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
test[['id','target']].to_csv('xgb_submission.csv', index=False, float_format='%.5f')

#LightGBM
def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

params = {'learning_rate': 0.02, 'max_depth': 4, 'boosting': 'gbdt', 'objective': 'binary', 'max_bin': 10, 'subsample': 0.8, 'subsample_freq': 10, 'colsample_bytree': 0.8, 'min_child_samples': 500, 'metric': 'auc', 'is_training_metric': False, 'seed': 99}
model2 = lgb.train(params, lgb.Dataset(x1, label=y1), 1000, lgb.Dataset(x2, label=y2), verbose_eval=50, feval=gini_lgb, early_stopping_rounds=200)
test['target'] = model2.predict(test[col], num_iteration=model2.best_iteration)
test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
test[['id','target']].to_csv('lgb_submission.csv', index=False, float_format='%.5f')

df1 = pd.read_csv('xgb_submission.csv')
df2 = pd.read_csv('lgb_submission.csv')
df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
blend = pd.merge(df1, df2, how='left', on='id')
for c in df1.columns:
    if c != 'id':
        blend[c] = (blend[c] * 0.5)  + (blend[c+'_'] * 0.5)
blend = blend[df1.columns]
blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0,1)
blend.to_csv('blend1.csv', index=False, float_format='%.5f')