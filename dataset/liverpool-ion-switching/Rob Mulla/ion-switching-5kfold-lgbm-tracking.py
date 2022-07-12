"""
Baseline Model Pipeline:
    - Baseline LGBM with some batch features
    - 5KFold Shuffled
    - Regression (code also allows for multiclass)
    - OOF / Predictions and Feature importances saved as CSV
    - Feature importance plot.
    
Updated in this version:
    - +/-2 shifts
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import lightgbm as lgb
#from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score
from datetime import datetime
import os

###########
# SETTINGS
###########

MODEL = os.path.basename(__file__).split('.')[0]

TARGET = 'open_channels'

TOTAL_FOLDS = 5
RANDOM_SEED = 529
MODEL_TYPE = 'LGBM'
LEARNING_RATE = 0.01
SHUFFLE = True
NUM_BOOST_ROUND = 500_000
EARLY_STOPPING_ROUNDS = 50
N_THREADS = -1
OBJECTIVE = 'regression'
METRIC = 'rmse'
NUM_LEAVES = 2**8+1
MAX_DEPTH = 4
FEATURE_FRACTION = 1
BAGGING_FRACTION = 1
BAGGING_FREQ = 0

####################
# READING IN FILES
####################

train = pd.read_csv('../input/liverpool-ion-switching/train.csv')
test = pd.read_csv('../input/liverpool-ion-switching/test.csv')
ss = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
train['train'] = True
test['train'] = False
tt = pd.concat([train, test], sort=False).reset_index(drop=True)
tt['train'] = tt['train'].astype('bool')

###########
# TRACKING
###########

run_id = "{:%m%d_%H%M}".format(datetime.now())


def update_tracking(
        run_id, field, value, csv_file="./tracking.csv",
        integer=False, digits=None, nround=6,
        drop_broken_runs=False):
    """
    Tracking function for keep track of model parameters and
    CV scores. `integer` forces the value to be an int.
    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except:
        df = pd.DataFrame()
    if drop_broken_runs:
        df = df.dropna(subset=['1_f1'])
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[run_id, field] = value  # Model number is index
    df = df.round(nround)
    df.to_csv(csv_file)


# Update Tracking
update_tracking(run_id, 'model_number', MODEL)
update_tracking(run_id, 'model_type', MODEL_TYPE)
update_tracking(run_id, 'seed', RANDOM_SEED, integer=True)
update_tracking(run_id, 'nfolds', TOTAL_FOLDS, integer=True)
update_tracking(run_id, 'lr', LEARNING_RATE)
update_tracking(run_id, 'shuffle', SHUFFLE)
update_tracking(run_id, 'boost_rounds', NUM_BOOST_ROUND)
update_tracking(run_id, 'es_rounds', EARLY_STOPPING_ROUNDS)
update_tracking(run_id, 'threads', N_THREADS)
update_tracking(run_id, 'objective', OBJECTIVE)
update_tracking(run_id, 'metric', METRIC)
update_tracking(run_id, 'num_leaves', NUM_LEAVES)
update_tracking(run_id, 'max_depth', MAX_DEPTH)
update_tracking(run_id, 'feature_fraction', FEATURE_FRACTION)
update_tracking(run_id, 'bagging_fraction', BAGGING_FRACTION)
update_tracking(run_id, 'bagging_freq', BAGGING_FREQ)

###########
# FEATURES
###########

# # Include batch
tt = tt.sort_values(by=['time']).reset_index(drop=True)
tt.index = ((tt.time * 10_000) - 1).values
tt['batch'] = tt.index // 50_000
tt['batch_index'] = tt.index - (tt.batch * 50_000)
tt['batch_slices'] = tt['batch_index'] // 5_000
tt['batch_slices2'] = tt.apply(lambda r: '_'.join(
    [str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

# 50_000 Batch Features
tt['signal_batch_min'] = tt.groupby('batch')['signal'].transform('min')
tt['signal_batch_max'] = tt.groupby('batch')['signal'].transform('max')
tt['signal_batch_std'] = tt.groupby('batch')['signal'].transform('std')
tt['signal_batch_mean'] = tt.groupby('batch')['signal'].transform('mean')
tt['mean_abs_chg_batch'] = tt.groupby(['batch'])['signal'].transform(
    lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch'] = tt.groupby(
    ['batch'])['signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch'] = tt.groupby(
    ['batch'])['signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch'] = tt['signal_batch_max'] - tt['signal_batch_min']
tt['maxtomin_batch'] = tt['signal_batch_max'] / tt['signal_batch_min']
tt['abs_avg_batch'] = (tt['abs_min_batch'] + tt['abs_max_batch']) / 2

# 5_000 Batch Features
tt['signal_batch_5k_min'] = tt.groupby(
    'batch_slices2')['signal'].transform('min')
tt['signal_batch_5k_max'] = tt.groupby(
    'batch_slices2')['signal'].transform('max')
tt['signal_batch_5k_std'] = tt.groupby(
    'batch_slices2')['signal'].transform('std')
tt['signal_batch_5k_mean'] = tt.groupby(
    'batch_slices2')['signal'].transform('mean')
tt['mean_abs_chg_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch_5k'] = tt['signal_batch_5k_max'] - tt['signal_batch_5k_min']
tt['maxtomin_batch_5k'] = tt['signal_batch_5k_max'] / tt['signal_batch_5k_min']
tt['abs_avg_batch_5k'] = (tt['abs_min_batch_5k'] + tt['abs_max_batch_5k']) / 2


# add shifts
tt['signal_shift+1'] = tt.groupby(['batch']).shift(1)['signal']
tt['signal_shift-1'] = tt.groupby(['batch']).shift(-1)['signal']
tt['signal_shift+2'] = tt.groupby(['batch']).shift(2)['signal']
tt['signal_shift-2'] = tt.groupby(['batch']).shift(-2)['signal']

for c in ['signal_batch_min', 'signal_batch_max',
          'signal_batch_std', 'signal_batch_mean',
          'mean_abs_chg_batch', 'abs_max_batch',
          'abs_min_batch',
          'range_batch', 'maxtomin_batch', 'abs_avg_batch',
          'signal_shift+1', 'signal_shift-1',
          'signal_batch_5k_min', 'signal_batch_5k_max',
          'signal_batch_5k_std',
          'signal_batch_5k_mean', 'mean_abs_chg_batch_5k',
          'abs_max_batch_5k', 'abs_min_batch_5k',
          'range_batch_5k', 'maxtomin_batch_5k',
          'abs_avg_batch_5k','signal_shift+2','signal_shift-2']:
    tt[f'{c}_msignal'] = tt[c] - tt['signal']


# FEATURES = [f for f in tt.columns if f not in ['open_channels','index','time','train','batch',
#                                                'batch_index','batch_slices','batch_slices2']]


FEATURES = ['signal',
            'signal_batch_min',
            'signal_batch_max',
            'signal_batch_std',
            'signal_batch_mean',
            'mean_abs_chg_batch',
            #'abs_max_batch',
            #'abs_min_batch',
            #'abs_avg_batch',
            'range_batch',
            'maxtomin_batch',
            'signal_batch_5k_min',
            'signal_batch_5k_max',
            'signal_batch_5k_std',
            'signal_batch_5k_mean',
            'mean_abs_chg_batch_5k',
            'abs_max_batch_5k',
            'abs_min_batch_5k',
            'range_batch_5k',
            'maxtomin_batch_5k',
            'abs_avg_batch_5k',
            'signal_shift+1',
            'signal_shift-1',
            # 'signal_batch_min_msignal',
            'signal_batch_max_msignal',
            'signal_batch_std_msignal',
            # 'signal_batch_mean_msignal',
            'mean_abs_chg_batch_msignal',
            'abs_max_batch_msignal',
            'abs_min_batch_msignal',
            'range_batch_msignal',
            'maxtomin_batch_msignal',
            'abs_avg_batch_msignal',
            'signal_shift+1_msignal',
            'signal_shift-1_msignal',
            'signal_batch_5k_min_msignal',
            'signal_batch_5k_max_msignal',
            'signal_batch_5k_std_msignal',
            'signal_batch_5k_mean_msignal',
            'mean_abs_chg_batch_5k_msignal',
            'abs_max_batch_5k_msignal',
            'abs_min_batch_5k_msignal',
            #'range_batch_5k_msignal',
            'maxtomin_batch_5k_msignal',
            'abs_avg_batch_5k_msignal',
            'signal_shift+2',
            'signal_shift-2']

print('....: FEATURE LIST :....')
print([f for f in FEATURES])

update_tracking(run_id, 'n_features', len(FEATURES), integer=True)
update_tracking(run_id, 'target', TARGET)

###########
# Metric
###########


def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    print(preds.shape)
    print(preds)
    preds = np.argmax(preds, axis=0)
#     score = metrics.cohen_kappa_score(labels, preds, weights = 'quadratic')
    score = f1_score(labels, preds, average='macro')
    return ('KaggleMetric', score, True)


###########
# MODEL
###########
tt['train'] = tt['train'].astype('bool')
train = tt.query('train').copy()
test = tt.query('not train').copy()
train['open_channels'] = train['open_channels'].astype(int)
X = train[FEATURES]
X_test = test[FEATURES]
y = train[TARGET].values
sub = test[['time']].copy()
groups = train['batch']

if OBJECTIVE == 'multiclass':
    NUM_CLASS = 11
else:
    NUM_CLASS = 1

# define hyperparammeter (some random hyperparammeters)
params = {'learning_rate': LEARNING_RATE,
          'max_depth': MAX_DEPTH,
          'num_leaves': NUM_LEAVES,
          'feature_fraction': FEATURE_FRACTION,
          'bagging_fraction': BAGGING_FRACTION,
          'bagging_freq': BAGGING_FREQ,
          'n_jobs': N_THREADS,
          'seed': RANDOM_SEED,
          'metric': METRIC,
          'objective': OBJECTIVE,
          'num_class': NUM_CLASS
          }

kfold = KFold(n_splits=TOTAL_FOLDS, shuffle=SHUFFLE, random_state=RANDOM_SEED)

oof_df = train[['signal', 'open_channels']].copy()
fi_df = pd.DataFrame(index=FEATURES)

fold = 1  # init fold counter
for tr_idx, val_idx in kfold.split(X, y, groups=groups):
    print(f'====== Fold {fold:0.0f} of {TOTAL_FOLDS} ======')
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val)

    model = lgb.train(params,
                      train_set,
                      num_boost_round=NUM_BOOST_ROUND,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      valid_sets=[train_set, val_set],
                      verbose_eval=50)
    # feval=lgb_Metric)

    if OBJECTIVE == 'multi_class':
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.argmax(preds, axis=1)
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.argmax(test_preds, axis=1)
    elif OBJECTIVE == 'regression':
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.round(np.clip(preds, 0, 10)).astype(int)
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)

    oof_df.loc[oof_df.iloc[val_idx].index, 'oof'] = preds
    sub[f'open_channels_fold{fold}'] = test_preds

    f1 = f1_score(oof_df.loc[oof_df.iloc[val_idx].index]['open_channels'],
                  oof_df.loc[oof_df.iloc[val_idx].index]['oof'],
                  average='macro')
    rmse = np.sqrt(mean_squared_error(oof_df.loc[oof_df.index.isin(val_idx)]['open_channels'],
                                      oof_df.loc[oof_df.index.isin(val_idx)]['oof']))

    update_tracking(run_id, f'{fold}_best_iter', model.best_iteration, integer=True)
    update_tracking(run_id, f'{fold}_rmse', rmse)
    update_tracking(run_id, f'{fold}_f1', f1)
    fi_df[f'importance_{fold}'] = model.feature_importance()
    print(f'Fold {fold} - validation f1: {f1:0.5f}')
    print(f'Fold {fold} - validation rmse: {rmse:0.5f}')

    fold += 1

oof_f1 = f1_score(oof_df['open_channels'],
                  oof_df['oof'],
                  average='macro')
oof_rmse = np.sqrt(mean_squared_error(oof_df['open_channels'],
                                      oof_df['oof']))

update_tracking(run_id, f'oof_f1', oof_f1)
update_tracking(run_id, f'oof_rmse', oof_rmse)

###############
# SAVE RESULTS
###############

s_cols = [s for s in sub.columns if 'open_channels' in s]

sub['open_channels'] = sub[s_cols].median(axis=1).astype(int)
sub.to_csv(f'./pred_{MODEL}_{oof_f1:0.6}.csv', index=False)
sub[['time', 'open_channels']].to_csv(f'./sub_{MODEL}_{oof_f1:0.10f}.csv',
                                      index=False,
                                      float_format='%0.4f')

oof_df.to_csv(f'./oof_{MODEL}_{oof_f1:0.6}.csv', index=False)

fi_df['importance'] = fi_df.sum(axis=1)
fi_df.to_csv(f'./fi_{MODEL}_{oof_f1:0.6}.csv', index=True)


fig, ax = plt.subplots(figsize=(15, 30))
fi_df.sort_values('importance')['importance'] \
    .plot(kind='barh',
          figsize=(15, 30),
          title=f'{MODEL} - Feature Importance',
          ax=ax)
plt.savefig(f'./{MODEL}__{oof_f1:0.6}.png')
