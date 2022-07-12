# # Credit to Jazivxt

import numpy as np 
import pandas as pd
from sklearn import *
import lightgbm as lgb

train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

def features(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 50_000
    df['batch_index'] = df.index  - (df.batch * 50_000)
    df['batch_slices'] = df['batch_index']  // 5_000
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for c in ['batch','batch_slices2']:
        d = {}
        d['mean'+c] = df.groupby([c])['signal'].mean()
        d['median'+c] = df.groupby([c])['signal'].median()
        d['max'+c] = df.groupby([c])['signal'].max()
        d['min'+c] = df.groupby([c])['signal'].min()
        d['std'+c] = df.groupby([c])['signal'].std()
        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        for v in d:
            df[v] = df[c].map(d[v].to_dict())
        df['range'+c] = df['max'+c] - df['min'+c]
        df['maxtomin'+c] = df['max'+c] / df['min'+c]
        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2
    
    #add shifts
    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+1'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-1'][i] = np.nan

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:
        df[c+'_msignal'] = df[c] - df['signal']
        
    return df

train = features(train)
test = features(test)

col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]
x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['open_channels'], test_size=0.3, random_state=7)
del train

def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = metrics.cohen_kappa_score(labels, preds, weights = 'quadratic')
    return ('KaggleMetric', score, True)
 
params = {'learning_rate': 0.8, 'max_depth': 50, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1} 
model = lgb.train(params, lgb.Dataset(x1, y1), 2000,  lgb.Dataset(x2, y2), verbose_eval=50, early_stopping_rounds=100, feval=lgb_Metric)
preds = model.predict(test[col], num_iteration=model.best_iteration)
test['open_channels'] = np.round(np.clip(preds, 0, 10)).astype(int)

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')

# Credit to Alex

from catboost import Pool,CatBoostRegressor

model = CatBoostRegressor(task_type = "CPU",
                          iterations=1000,
                          learning_rate=0.1,
                          random_seed = 42,
                          depth=2,
                         )
# Fit model
model.fit(x1, y1)
# Get predictions
preds_catb = model.predict(test[col])
train_preds = model.predict(train[col], num_iteration=model.best_iteration)
test[['time','open_channels']].to_csv('submission_cat3.csv', index=False, float_format='%.4f')
# Again, all credit to alex
preds_comb = 0.80 * preds + 0.20 * preds_catb
test['open_channels'] = np.round(np.clip(preds_comb, 0, 10)).astype(int)
test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')