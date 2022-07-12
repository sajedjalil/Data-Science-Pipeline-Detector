import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit


def gini(actual, pred):
    actual = np.asarray(actual)  # In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    gini_sum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return gini_sum / n


def gini_normalized(a, p):
    if p.ndim == 2:  # Required for sklearn wrapper
        p = p[:, 1]  # If proba array contains proba for both 0 and 1 classes, just pick class 1
    return gini(a, p) / gini(a, a)


def get_groups(y, group_bounds):
    if group_bounds is not None:
        groups = y.copy()
        groups.loc[y < group_bounds[0][0]] = 0
        for i, (l, r) in enumerate(group_bounds):
            groups.loc[(y >= l) & (y < r)] = i + 1
        groups.loc[y >= group_bounds[-1][1]] = len(group_bounds) + 1
    else:
        groups = None
    return groups


def insample_outsample_split(x, y, train_size=.5, holdout_num=5, holdout_frac=.7, random_state=0, full_holdout=False,
                             group_bounds=None):
    if isinstance(train_size, float):
        int(train_size * len(y))
    groups = get_groups(y, group_bounds)
    if groups is None:
        train_index, h_index = ShuffleSplit(n_splits=1, train_size=train_size, test_size=None,
                                            random_state=random_state).split(y).__next__()
    else:
        train_index, h_index = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=None,
                                                      random_state=random_state).split(y, groups).__next__()
    train_x = x.take(train_index)
    train_y = y.take(train_index)
    h_x = x.take(h_index)
    h_y = y.take(h_index)

    groups = get_groups(h_y, group_bounds)
    h_set = []
    for i in range(holdout_num):
        if groups is None:
            off_index, v_index = ShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                              random_state=random_state + i).split(h_y).__next__()
        else:
            off_index, v_index = StratifiedShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                                        random_state=random_state + i).split(h_y, groups).__next__()
        valid_x = h_x.take(v_index)
        valid_y = h_y.take(v_index)
        h_set.append((valid_x, valid_y))

    if full_holdout:
        return train_x, train_y, h_set, h_x, h_y
    return train_x, train_y, h_set


def get_data():
    # drop features without bias
    def drop_alpha(df):
        cols = ['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04']
        df = df.drop(cols, axis=1)
        return df

    def recon(reg):
        integer = int(np.round((40 * reg) ** 2))
        for a in range(32):
            if (integer - a) % 31 == 0:
                A = a
        M = (integer - A) // 31
        return A, M

    def decode_features(df):
        df['ps_reg_03_A'] = df['ps_reg_03'].apply(lambda ele: recon(ele)[0])
        df['ps_reg_03_M'] = df['ps_reg_03'].apply(lambda ele: recon(ele)[1])
        df['ps_reg_03_A'].replace(19, -1, inplace=True)
        df['ps_reg_03_M'].replace(51, -1, inplace=True)

    def encode_features(df):
        df['negative_one_vals'] = (df == -1).sum(axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

        df.loc[0 == df.ps_calc_06, 'ps_calc_06'] = 1
        df.loc[1 == df.ps_calc_08, 'ps_calc_08'] = 2
        df.loc[df.ps_calc_11 > 19, 'ps_calc_11'] = 19
        df.loc[df.ps_calc_12 > 10, 'ps_calc_12'] = 10
        df.loc[df.ps_calc_13 > 13, 'ps_calc_13'] = 13
        df.loc[df.ps_calc_14 > 23, 'ps_calc_14'] = 23

        for col in ['ps_reg_03', 'ps_reg_03_M', 'ps_car_12', 'ps_car_13', 'ps_car_14']:
            df[col + '_median_range'] = (df[col] > d_median[col]).astype(np.uint8)
            df[col + '_mean_range'] = (df[col] > d_mean[col]).astype(np.uint8)

    def fill_na(df):
        df.loc[df.ps_car_02_cat == -1, 'ps_car_02_cat'] = 1
        df.loc[df.ps_car_11 == -1, 'ps_car_11'] = 3
        df.loc[df.ps_car_12 == -1, 'ps_car_12'] = 0

        df.loc[df.ps_ind_02_cat == -1, 'ps_ind_02_cat'] = 3
        df.loc[df.ps_ind_04_cat == -1, 'ps_ind_04_cat'] = 1
        df.loc[df.ps_car_01_cat == -1, 'ps_car_01_cat'] = 5
        df.loc[df.ps_car_09_cat == -1, 'ps_car_09_cat'] = 3

    def generalize_feature(df):
        df.loc[df.ps_calc_05.isin(range(5)), 'ps_calc_05'] = 4
        df['ps_calc_05'] -= 4

        df.loc[df.ps_calc_06 > 5, 'ps_calc_06'] = 5
        df['ps_calc_06'] -= 1

        df.loc[df.ps_calc_07 < 4, 'ps_calc_07'] = 4
        df['ps_calc_07'] -= 4

        df.loc[df.ps_calc_08 > 7, 'ps_calc_08'] = 7
        df['ps_calc_08'] -= 2

        df.loc[df.ps_calc_09 < 4, 'ps_calc_09'] = 4
        df['ps_calc_09'] -= 4

        df.loc[df.ps_calc_10.isin(range(2, 13)), 'ps_calc_10'] = 2
        df.loc[df.ps_calc_10 > 22, 'ps_calc_10'] = 22
        df.loc[df.ps_calc_10 >= 13, 'ps_calc_10'] -= 10

        df.loc[df.ps_calc_11 > 18, 'ps_calc_11'] = 18

        df.loc[df.ps_calc_12 < 5, 'ps_calc_12'] = 5
        df.loc[df.ps_calc_12 > 9, 'ps_calc_12'] = 9
        df['ps_calc_12'] -= 5

        df.loc[df.ps_calc_13 < 6, 'ps_calc_13'] = 6
        df.loc[df.ps_calc_13 > 12, 'ps_calc_13'] = 12
        df['ps_calc_13'] -= 6

        df.loc[df.ps_calc_14.isin(range(4, 13)), 'ps_calc_14'] = 4
        df.loc[df.ps_calc_14 > 21, 'ps_calc_14'] = 21
        df.loc[df.ps_calc_14 >= 13, 'ps_calc_14'] -= 8

    def combine_features(df):
        # combine invariable features
        df['invar_combo_bin'] = df[invar_cols[0]]
        for i in range(1, len(invar_cols)):
            df['invar_combo_bin'] += 2 ** i * df[invar_cols[i]]

        # combine calc_bin infos
        df['combo_calc_bin'] = df[calc_bin_cols[0]]
        for i in range(1, len(calc_bin_cols)):
            df['combo_calc_bin'] += 2 ** i * df[calc_bin_cols[i]]

        df['combo_calc_579'] = df.ps_calc_05 + 3 * df.ps_calc_07 + 18 * df.ps_calc_09
        df['combo_calc_68'] = df.ps_calc_06 + 5 * df.ps_calc_08
        df['combo_calc_1213'] = df.ps_calc_12 + 5 * df.ps_calc_13

        return df.drop(
            invar_cols + calc_bin_cols + ['ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
                                          'ps_calc_12', 'ps_calc_13'], axis=1)

    def prune_features(df):
        cols = ['ps_car_11_cat', 'ps_ind_18_bin', 'ps_car_13_x_ps_reg_03']
        return df.drop(cols, axis=1)

    def combo_features(df):
        df['ps_ind_03*ps_ind_07_bin'] = df['ps_ind_03'] * df['ps_ind_07_bin']
        df['ps_car_13*ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    # read data
    data_dir = '../input'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    submission = pd.DataFrame(test_df.id.copy())

    X = train_df.drop(['target', 'id'], axis=1)
    y = train_df['target'].copy()
    y_cnts = y.value_counts()
    y_pct = y_cnts / y.shape[0]
    print(y_cnts[0], y_cnts[1], y_pct[0], y_pct[1])
    test_x = test_df.drop('id', axis=1)
    del train_df, test_df

    X = drop_alpha(X)
    test_x = drop_alpha(test_x)

    decode_features(X)
    decode_features(test_x)

    d_median = X.median(axis=0)
    d_mean = X.mean(axis=0)
    encode_features(X)
    encode_features(test_x)

    fill_na(X)
    fill_na(test_x)

    # combine infos
    bin_cols = [col for col in X.columns if '_bin' in col]
    invariance_pct = y_pct[1] * 0.05
    invar_cols = []
    for col in bin_cols:
        tr_pct = X[col].value_counts() / X.shape[0]
        if tr_pct[0] < invariance_pct or tr_pct[1] < invariance_pct:
            invar_cols.append(col)
    print(invar_cols)

    calc_bin_cols = ['ps_calc_' + str(i) + '_bin' for i in range(15, 21)]
    print(calc_bin_cols)

    generalize_feature(X)
    generalize_feature(test_x)

    X = combine_features(X)
    test_x = combine_features(test_x)

    X = prune_features(X)
    test_x = prune_features(test_x)

    combo_features(X)
    combo_features(test_x)

    return X, y, test_x, submission


X, y, test_x, submission = get_data()

print('train begin ...')
num_boost_round = 1181
params = {'objective': ['binary'], 'boosting_type': 'gbdt', 'metric': ['auc'], 'verbose': -1,
          'learning_rate': 0.0085, 'num_leaves': 16, 'max_depth': 8, 'max_bin': 255, 'min_data': 1187,
          'bagging_fraction': 0.9, 'feature_fraction': 0.7, 'bagging_freq': 3, 'lambda_l1': 48, 'lambda_l2': 0.4,
          'is_unbalance': True}

k = 2
models = []
train_aucs = []
train_ginis = []
valid_aucs = []
valid_ginis = []
for train_index, valid_index in KFold(n_splits=k, shuffle=True, random_state=853).split(y):
    train_x = X.take(train_index)
    train_y = y.take(train_index)
    valid_x = X.take(valid_index)
    valid_y = y.take(valid_index)

    model = lgb.train(params, lgb.Dataset(train_x.values, label=train_y.values), num_boost_round=num_boost_round)
    models.append(model)

    train_p = model.predict(train_x)
    train_auc = metrics.roc_auc_score(train_y, train_p)
    train_gini = gini_normalized(train_y, train_p)
    train_aucs.append(train_auc)
    train_ginis.append(train_gini)

    valid_p = model.predict(valid_x)
    valid_auc = metrics.roc_auc_score(valid_y, valid_p)
    valid_gini = gini_normalized(valid_y, valid_p)
    valid_aucs.append(valid_auc)
    valid_ginis.append(valid_gini)

    print('train auc:%s, valid auc:%s' % (str(train_auc), str(valid_auc)))
    print('train gini:%s, valid gini:%s' % (str(train_gini), str(valid_gini)))
    print('----------------------------------------------------------------------------')
print('train_auc_mean=', np.mean(train_aucs), ', train_auc_std=', np.std(train_aucs))
print('train_gini_mean=', np.mean(train_ginis), ', train_gini_std=', np.std(train_ginis))
print('----------------------------------------------------------------------------')
print('valid_auc_mean=', np.mean(valid_aucs), ', valid_auc_std=', np.std(valid_aucs))
print('valid_gini_mean=', np.mean(valid_ginis), ', valid_gini_std=', np.std(valid_ginis))
print('----------------------------------------------------------------------------')

submission['target'] = 0.0
for model in models:
    submission['target'] += (pd.Series(model.predict(test_x.values)).rank() / test_x.shape[0]).values
submission['target'] /= k

submission.to_csv('lgb_prune_%s.csv' % datetime.now().strftime("%Y%m%d%H%M%S"), index=False)