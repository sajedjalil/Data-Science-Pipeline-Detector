"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt

debug = False
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

uint8_max = np.iinfo(np.uint8).max
uint16_max = np.iinfo(np.uint16).max
uint32_max = np.iinfo(np.uint32).max


def choose_int_type(n):
    if n <= uint8_max:
        return 'uint8'
    elif n <= uint16_max:
        return 'uint16'
    elif n <= uint32_max:
        return 'uint32'
    else:
        return 'uint64'


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.2,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return bst1, bst1.best_iteration


def get_nunique(train_df, selcols):
    gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].nunique()
    gp = gp.astype(choose_int_type(gp.max()))
    gp = gp.reset_index().rename(columns={selcols[-1]: 'nunique_{}'.format('_'.join(selcols))})
    train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
    del gp
    gc.collect()
    return train_df


def get_count(train_df, selcols):
    gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].count()
    gp = gp.astype(choose_int_type(gp.max()))
    gp = gp.reset_index().rename(columns={selcols[-1]: 'count_{}'.format('_'.join(selcols))})
    train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
    del gp
    gc.collect()
    return train_df


def get_var(train_df, selcols):
    gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].var().reset_index().rename(columns={selcols[-1]: 'var_{}'.format('_'.join(selcols))})
    train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
    del gp
    gc.collect()
    return train_df


def get_mean(train_df, selcols):
    gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].mean().reset_index().rename(columns={selcols[-1]: 'mean_{}'.format('_'.join(selcols))})
    train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
    del gp
    gc.collect()
    return train_df


def get_cumcount(train_df, selcols):
    gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].cumcount()
    gp = gp.astype(choose_int_type(gp.max()))
    train_df.loc[:, 'cumcount_{}'.format('_'.join(selcols))] = gp.values
    del gp
    gc.collect()
    return train_df


def get_order(series):
    return list(range(len(series)))


def process(frm, to):
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    print('loading train data...', frm, to)
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1, frm), nrows=to-frm, dtype=dtypes, usecols=train_cols)
    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=test_cols)
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=test_cols)

    len_train = len(train_df)
    train_df = train_df.append(test_df)
    del test_df
    gc.collect()
    print('doing nextClick')
    next_click_cols = ['ip', 'app', 'device', 'os']
    print('Group by ...')
    gp = train_df.groupby(next_click_cols)
    print('Add epochtime ...')
    train_df.loc[:, 'epochtime'] = train_df.click_time.astype(np.int64)
    print('Add order_left ...')
    train_df.loc[:, 'order_left'] = gp['channel'].transform(get_order)
    print('Add order_right ...')
    train_df.loc[:, 'order_right'] = train_df.order_left - 1
    left_selcols = train_df.columns.tolist()
    left_selcols.remove('order_right')
    right_selcols = next_click_cols + ['order_right', 'epochtime']
    print('Start merging ...')
    start_time = time.time()
    train_df = pd.merge(train_df.loc[:, left_selcols], train_df.loc[:, right_selcols],
                        left_on=next_click_cols+['order_left'], right_on=next_click_cols+['order_right'],
                        how='left', suffixes=('_left', '_right'))
    print('Finished merging, lasts ', time.time() - start_time)
    train_df = train_df.fillna(-1)
    train_df.loc[:, 'nextclick'] = train_df.epochtime_right - train_df.epochtime_left
    train_df.drop(['order_left', 'order_right', 'epochtime_left', 'epochtime_right'], axis=1, inplace=True)
    gc.collect()

    print('Extracting features...')
    train_df.loc[:, 'click_time'] = pd.to_datetime(train_df.click_time, format='%Y-%m-%d %H:%M:%S')
    train_df.loc[:, 'hour'] = train_df.click_time.dt.hour.astype('uint8')
    train_df.loc[:, 'day'] = train_df.click_time.dt.day.astype('uint8')
    gc.collect()
    print('Get unique values ...')
    get_nunique_list = [
        ['ip', 'channel'],
        ['ip', 'device', 'os', 'app'],
        ['ip', 'day', 'hour'],
        ['ip', 'app'],
        ['ip', 'app', 'os'],
        ['ip', 'device'],
        ['app', 'channel']
    ]
    for selcols in get_nunique_list:
        train_df = get_nunique(train_df, selcols)
    gc.collect()
    print('Get cumcounts ...')
    get_cumcount_list = [
        ['ip', 'device', 'os', 'app'],
        ['ip', 'os'],
    ]
    for selcols in get_cumcount_list:
        train_df = get_cumcount(train_df, selcols)
    gc.collect()
    print('Get counts ...')
    get_count_list = [
        ['ip', 'day', 'hour', 'channel'],
        ['ip', 'app', 'channel'],
        ['ip', 'app', 'os', 'channel'],
    ]
    for selcols in get_count_list:
        train_df = get_count(train_df, selcols)
    gc.collect()
    print('Get vars ...')
    get_var_list = [
        ['ip', 'day', 'channel', 'hour'],
        ['ip', 'app', 'os', 'hour'],
        ['ip', 'app', 'channel', 'day']
    ]
    for selcols in get_var_list:
        train_df = get_var(train_df, selcols)
    gc.collect()
    print('Get means ...')
    get_mean_list = [
        ['ip', 'app', 'channel', 'hour']
    ]
    for selcols in get_mean_list:
        train_df = get_mean(train_df, selcols)
    gc.collect()

    target = 'is_attributed'
    predictors = train_df.columns.tolist()
    to_removes = ['ip', 'click_time', 'click_id']
    for to_remove in to_removes:
        predictors.remove(to_remove)
    predictors.remove(target)
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    print('predictors: ', predictors)

    test_df = train_df.iloc[len_train:]
    val_df = train_df.iloc[(len_train-val_size):len_train]
    train_df = train_df.iloc[:(len_train-val_size)]
    print('Train set classes: ')
    print(train_df.is_attributed.value_counts(normalize=True))
    print('Val set classes: ')
    print(val_df.is_attributed.value_counts(normalize=True))

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub.loc[:, 'click_id'] = test_df.click_id.astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }
    bst, best_iteration = lgb_modelfit_nocv(
        params,
        train_df,
        val_df,
        predictors,
        target,
        objective='binary',
        metrics='auc',
        early_stopping_rounds=30,
        verbose_eval=True,
        num_boost_round=1000,
        categorical_features=categorical
    )

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()
    print('Plot feature importances...')
    lgb.plot_importance(bst, max_num_features=100)
    plt.savefig('feats_importance.png')

    print("Predicting...")
    sub.loc[:, 'is_attributed'] = bst.predict(test_df[predictors], num_iteration=best_iteration)
    if not debug:
        print("writing...")
        sub.to_csv('sub_it.csv.gz', index=False, compression='gzip')
    print("done...")
    return sub


nrows = 184903891 - 1
nchunk = 40000000
val_size = 2500000

frm = nrows - 75000000
if debug:
    frm = 0
    nchunk = 100000
    val_size = 10000

to = frm + nchunk

sub = process(frm, to, 0)
