import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
import gc
import lightgbm as lgb


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10,
                      categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
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
        'min_split_gain': 0.5,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 5,  # L1 regularization term on weights
        'reg_lambda': 10,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric': metrics
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
    print(metrics + ":", evals_results['valid'][metrics][bst1.best_iteration - 1])

    return (bst1, bst1.best_iteration)



@timeit
def prepare_features(train_test_df):

    print("preparing 1")
    gp = train_test_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])['channel']\
        .count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_test_df = train_test_df.merge(gp, on=['ip', 'app'], how='left')
    del gp
    gc.collect()

    print("preparing 2")
    gp = train_test_df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])['hour']\
        .count().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_count'})
    train_test_df = train_test_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print("preparing 3")
    gp = train_test_df[['ip', 'hour', 'channel', 'weekday']].groupby(by=['ip', 'hour', 'channel'])['weekday']\
        .count().reset_index().rename(index=str, columns={'weekday': 'ip_hour_channel_count'})
    train_test_df = train_test_df.merge(gp, on=['ip', 'hour', 'channel'], how='left')
    del gp
    gc.collect()

    print("preparing 4")
    gp = train_test_df[['channel', 'hour']].loc[train_test_df['is_attributed']==1].groupby(by=['channel'])['hour']\
        .mean().reset_index().rename(index=str, columns={'hour': 'diff_from_hit_mean_hour_of_channel'})
    train_test_df = train_test_df.merge(gp, on=['channel'], how='left')
    train_test_df['diff_from_hit_mean_hour_of_channel'] = \
        (train_test_df['diff_from_hit_mean_hour_of_channel']-train_test_df['hour']).abs()
    del gp
    gc.collect()

    print("preparing 5")
    gp = train_test_df[['app', 'hour']].loc[train_test_df['is_attributed'] == 1].groupby(by=['app'])['hour'] \
        .mean().reset_index().rename(index=str, columns={'hour': 'diff_from_hit_mean_hour_of_app'})
    train_test_df = train_test_df.merge(gp, on=['app'], how='left')
    train_test_df['diff_from_hit_mean_hour_of_app'] = \
        (train_test_df['diff_from_hit_mean_hour_of_app'] - train_test_df['hour']).abs()
    del gp
    gc.collect()

    train_test_df = train_test_df.round(1)

    return train_test_df

@timeit
#add prefix for the new columns
def get_date_features(df, timestamp_column, prefix_in_new_cols=''):
    df[prefix_in_new_cols + 'hour'] = pd.to_datetime(df[timestamp_column]).dt.hour.astype('uint8')
    # Get the day of the week where monday = 0, tuesday = 1 and so on
    df[prefix_in_new_cols + 'weekday'] = pd.to_datetime(df[timestamp_column]).dt.dayofweek.astype('uint8')

    return df



if __name__ == "__main__":

    debug = True

    # loading data

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    if debug:
        raw_test_file = '../input/test.csv'
        raw_train_file = '../input/train_sample.csv'
        number_of_rows_to_read = 100000
    else:
        raw_test_file = '../input/test.csv'
        raw_train_file = '../input/train.csv'
        number_of_rows_to_read = None

    print("loading train data...")
    train_df = pd.read_csv(raw_train_file, dtype=dtypes, nrows=number_of_rows_to_read)
    train_df_len = len(train_df)
    print("train data loaded. size = " + str(train_df_len))

    print("loading test data...")
    test_df = pd.read_csv(raw_test_file, dtype=dtypes, nrows=number_of_rows_to_read)
    test_df_len = len(test_df)
    print("test data loaded. size = " + str(test_df_len))

    print("joining test and train data")
    train_test_df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    print('getting time features from the data')
    train_test_df = get_date_features(train_test_df, 'click_time')
    train_test_df = train_test_df.drop(columns=['attributed_time', 'click_time'])

    print("getting additional features")
    train_test_df = prepare_features(train_test_df)

    print("breaking again the test and train data")
    test_df = train_test_df[train_df_len:]
    train_df = train_test_df[:train_df_len]
    del train_test_df
    gc.collect()

    print('removing useless columns from train')
    train_df.drop(columns=['click_id'])

    print('removing useless columns from test')
    test_df.drop(columns=['is_attributed'])

    print('splitting train df to train and val')
    train_df, validation_df = train_test_split(train_df, test_size=0.6)

    categorical_features = ['app', 'channel', 'device', 'ip', 'os', 'hour', 'weekday']
    label_col_name = 'is_attributed'

    # making predictors list
    predictors = list(train_df)
    predictors.remove(label_col_name)


    print("Training...")

    params = {
        'learning_rate': 0.150,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 63,  # 2^max_depth - 1
        'max_depth': 6,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0.5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }
    (bst, best_iteration) = lgb_modelfit_nocv(params,
                                              train_df,
                                              validation_df,
                                              predictors,
                                              label_col_name,
                                              objective='binary',
                                              metrics='auc',
                                              early_stopping_rounds=30,
                                              verbose_eval=True,
                                              num_boost_round=1000,
                                              categorical_features=categorical_features)

    del train_df
    del validation_df
    gc.collect()

    print("Predicting...")

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    sub['is_attributed'] = bst.predict(test_df[predictors], num_iteration=best_iteration)
    sub = sub.sort_values('click_id')
    if not debug:
        print("writing...")
        sub.to_csv('sub_it%d.csv' % (2), index=False)
    else:
        print(sub)
    print("done...")
