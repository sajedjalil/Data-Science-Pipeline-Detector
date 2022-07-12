# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.precision = 15

import matplotlib.pyplot as plt
# %matplotlib inline
from tqdm import tqdm

# SKLearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Keras
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Concatenate, Multiply, BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras.callbacks import Callback, ModelCheckpoint

# SciPy
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats

# Gradient Boosted Models
import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostClassifier
import seaborn as sns
import altair as alt
from altair.vega import v3

# Utils
import time
import datetime
import json
import ast
import shap
import gc
import itertools
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will
# list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# From Andrew Lukyanenko, kaggle kernel: Where do the robots drive?
# FEATURE GENERATION
# We have 128 measurements in each series, so it makes sense to create
# aggregate features. I create several groups of them:

# Usual aggregations: mean, std, min and max, absolute min and max. Max to min rate;
# Mean change rate in absolute and relative values - it shows how fast values change;
# Quantiles - showing extreme values;
# Trend features - to show whether values decrease or increase;
# Rolling features - to show mean/std values with windows;
# Various statistical features from LANL competition;
# Descriptions will be done later. I use ideas from my kernel for another
# competition:
# https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples


def feature_generation(train_df, train):

    def calc_change_rate(x):
        change = (np.diff(x) / x[:-1]).values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        return np.mean(change)

    def add_trend_feature(arr, abs_values=False):
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]

    def classic_sta_lta(x, length_sta, length_lta):

        sta = np.cumsum(x ** 2)

        # Convert to float
        sta = np.require(sta, dtype=np.float)

        # Copy for LTA
        lta = sta.copy()

        # Compute the STA and the LTA
        sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
        sta /= length_sta
        lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
        lta /= length_lta

        # Pad zeros
        sta[:length_lta - 1] = 0

        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny

        return sta / lta

    train_df['total_xyz'] = (train['orientation_X']**2 + train['orientation_Y']**2 + train['orientation_Z'])**0.5
    train_df['total_angular_velocity'] = (train['angular_velocity_X'] ** 2 + train['angular_velocity_Y'] ** 2 + train['angular_velocity_Z'] ** 2) ** 0.5
    train_df['total_linear_acceleration'] = (train['linear_acceleration_X'] ** 2 + train['linear_acceleration_Y'] ** 2 + train['linear_acceleration_Z'] ** 2) ** 0.5
    train_df['acc_vs_vel'] = train_df['total_linear_acceleration'] / train_df['total_angular_velocity']
    
    # Are there any reasons to not automatically normalize a quaternion? And if there are, what quaternion operations do result in non-normalized quaternions?

    # Any operation that produces a quaternion will need to be normalized because floating-point precession errors will cause it to not be unit length. 
    # I would advise against standard routines performing normalization automatically for performance reasons. Any competent programmer should be aware 
    # of the precision issues and be able to normalize the quantities when necessary - and it is not always necessary to have a unit length quaternion. 
    # The same is true for vector operations.
    # source: https://stackoverflow.com/questions/11667783/quaternion-and-normalization
    
    train_df['norm_quat'] = (train['orientation_X']**2 + train['orientation_Y']**2 + train['orientation_Z']**2 + train['orientation_W']**2)
    train_df['mod_quat'] = (train_df['norm_quat'])**0.5
    train_df['norm_X'] = train['orientation_X'] / train_df['mod_quat']
    train_df['norm_Y'] = train['orientation_Y'] / train_df['mod_quat']
    train_df['norm_Z'] = train['orientation_Z'] / train_df['mod_quat']
    train_df['norm_W'] = train['orientation_W'] / train_df['mod_quat']        


    for col in tqdm(train.columns):
        if col in ['row_id','series_id','measurement_number']:
            continue
        train_df[col + '_mean'] = train.groupby(['series_id'])[col].mean()
        train_df[col + '_std'] = train.groupby(['series_id'])[col].std()
        train_df[col + '_max'] = train.groupby(['series_id'])[col].max()
        train_df[col + '_min'] = train.groupby(['series_id'])[col].min()
        train_df[col + '_max_to_min'] = train_df[col + '_max'] / train_df[col + '_min']
        train_df[col + '_mean_abs_chg'] = train.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        train_df[col + '_abs_max'] = train.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        train_df[col + '_abs_min'] = train.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        train_df[col + '_abs_avg'] = (train_df[col + '_abs_min'] + train_df[col + '_abs_max'])/2

        # for i in train_df['series_id']:
        #     train_df.loc[i, col + '_mean_change_abs'] = np.mean(
        #         np.diff(train.loc[train['series_id'] == i, col]))
        #     train_df.loc[i, col + '_mean_change_rate'] = calc_change_rate(
        #         train.loc[train['series_id'] == i, col])

        #     # min/max
        #     train_df.loc[
        #         i, col + '_abs_min'] = np.abs(train.loc[train['series_id'] == i, col]).min()
        #     train_df.loc[
        #         i, col + '_abs_max'] = np.abs(train.loc[train['series_id'] == i, col]).max()

        #     # trend
        #     train_df.loc[i, col + '_trend'] = add_trend_feature(
        #         train.loc[train['series_id'] == i, col].values)
        #     train_df.loc[i, col + '_abs_trend'] = add_trend_feature(
        #         train.loc[train['series_id'] == i, col].values, abs_values=True)
        #     train_df.loc[
        #         i, col + '_abs_mean'] = np.abs(train.loc[train['series_id'] == i, col]).mean()
        #     train_df.loc[
        #         i, col + '_abs_std'] = np.abs(train.loc[train['series_id'] == i, col]).std()
                
            # 95, 99 and 05 quantile
            # train_df.loc[
            #     i, col + '_q95'] = np.quantile(train.loc[train['series_id'] == i, col], 0.95)
            # train_df.loc[
            #     i, col + '_q99'] = np.quantile(train.loc[train['series_id'] == i, col], 0.99)
            # train_df.loc[
            #     i, col + '_q05'] = np.quantile(train.loc[train['series_id'] == i, col], 0.05)

            # kurtosis - shape of probability distribution
            # skewness - measure of asymmetry
            # MAD - robust measure of variability of a univariate sample
            # train_df.loc[
            #     i, col + '_mad'] = train.loc[train['series_id'] == i, col].mad()
            # train_df.loc[
            #     i, col + '_kurt'] = train.loc[train['series_id'] == i, col].kurtosis()
            # train_df.loc[
            #     i, col + '_skew'] = train.loc[train['series_id'] == i, col].skew()
            # train_df.loc[
            #     i, col + '_med'] = train.loc[train['series_id'] == i, col].median()

            # Hilbert Mean ???
            # train_df.loc[i, col + '_Hilbert_mean'] = np.abs(
            #     hilbert(train.loc[train['series_id'] == i, col])).mean()

            # The Hann function is typically used as a window function in
            # digital signal processing to select a subset of a series of
            # samples in order to perform a Fourier transform or other
            # calculations.
            # train_df.loc[i, col + '_Hann_window_mean'] = (convolve(
            #     train.loc[train['series_id'] == i, col], hann(15), mode='same') / sum(hann(15))).mean()
            # train_df.loc[i, col + '_classic_sta_lta1_mean'] = classic_sta_lta(
            #     train.loc[train['series_id'] == i, col], 10, 50).mean()

            # How does this help ??
            # train_df.loc[i, col + '_Moving_average_10_mean'] = train.loc[
            #     train['series_id'] == i, col].rolling(window=10).mean().mean(skipna=True)
            # train_df.loc[i, col + '_Moving_average_16_mean'] = train.loc[
            #     train['series_id'] == i, col].rolling(window=16).mean().mean(skipna=True)
            # train_df.loc[i, col + '_Moving_average_10_std'] = train.loc[
            #     train['series_id'] == i, col].rolling(window=10).std().mean(skipna=True)
            # train_df.loc[i, col + '_Moving_average_16_std'] = train.loc[
            #     train['series_id'] == i, col].rolling(window=16).std().mean(skipna=True)

            # IQR - interquartile range
            # train_df.loc[i, col + 'iqr'] = np.subtract(*np.percentile(
            #     train.loc[train['series_id'] == i, col], [75, 25]))
            # train_df.loc[i, col + 'ave10'] = stats.trim_mean(
            #     train.loc[train['series_id'] == i, col], 0.1)  # WHAT IS THIS FOR ??

    return train_df, train


def get_df(train):
    train_df = train[['series_id']].drop_duplicates().reset_index(drop=True)
    return train_df

train = pd.read_csv('../input/career-con-2019/X_train.csv')
y = pd.read_csv('../input/career-con-2019/y_train.csv')
test = pd.read_csv('../input/career-con-2019/X_test.csv')
sub = pd.read_csv('../input/career-con-2019/sample_submission.csv')

# print(y[y['surface']=='hard_tiles'].count())

TEST = test
TRAIN = train

for col in train.columns:
    if 'orient' in col:
        scaler = StandardScaler()
        train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
        test[col] = scaler.transform(test[col].values.reshape(-1, 1))

train_df = get_df(train)
test_df = get_df(test)

print("TRAIN")
print(train_df.shape)
print(train.shape)
print("TEST")
print(test_df.shape)
print(test.shape)

train_df, _ = feature_generation(train_df, train)
# train_df = pd.read_csv('../input/careercondf/train_df.csv', header=0, dtype='float32')
test_df, _ = feature_generation(test_df, test)
# test_df = pd.read_csv('../input/careercondf/test_df.csv', header=0, dtype='float32')

print("TRAIN")
print(train_df.shape)
print("TEST")
print(test_df.shape)

TEST.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
TRAIN.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
scaler = StandardScaler()
TRAIN = scaler.fit_transform(TRAIN)
TRAIN = TRAIN.reshape(3810, 128, 10, 1)
TRAIN = TRAIN.transpose(0, 2, 1, 3)
TEST = scaler.fit_transform(TEST)
TEST = TEST.reshape(3816, 128, 10, 1)
TEST = TEST.transpose(0, 2, 1, 3)


# improving classifier accuracy
n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019)

# Encoding output classes
le = LabelEncoder()
le.fit(y['surface'])
y['surface'] = le.transform(y['surface'])

train_df = train_df.drop(['series_id'], axis=1)
test_df = test_df.drop(['series_id'], axis=1)


# BUILDING MODEL


# Eval: Accuracy

def eval_acc(preds, dtrain):
    labels = dtrain.get_label()
    return 'acc', accuracy_score(labels, preds.argmax(1)), True


# CONV NET
# From Nicolas Taylor, kaggle kernel: CareerCon ConvNeuralNet Starter
def conv_model():

    # These features occur frequently throughout, so for easu of use, it's
    # easier to change them up here.
    FIRST = 30  # 20
    SECOND = 20  # 10
    HEIGHT1 = 4  # 4
    HEIGHT2 = 3  # 4
    DROPOUT = 0.5
    STRIDES = None
    PS = 5

    input0 = Input(shape=(290, 1))
    layer0 = Conv1D(filters=128, kernel_size=10, strides=2, input_shape=(
        290, 1), kernel_initializer='uniform', activation='relu')(input0)
    layer0 = BatchNormalization()(layer0)
    layer0 = Conv1D(filters=81, kernel_size=10, strides=2, input_shape=(
        290, 1), kernel_initializer='uniform', activation='relu')(layer0)
    layer0 = BatchNormalization()(layer0)
    layer0 = MaxPooling1D(strides=STRIDES, pool_size=9)(layer0)
    layer0 = Flatten()(layer0)

    input1 = Input(shape=(128, 1))
    a = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input1)
    a = BatchNormalization()(a)
    a = MaxPooling1D(strides=STRIDES, pool_size=PS)(a)
    a = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(a)
    a = BatchNormalization()(a)
    a = MaxPooling1D(strides=STRIDES, pool_size=PS)(a)
    a = Flatten()(a)
    a = Dropout(DROPOUT)(a)

    input2 = Input(shape=(128, 1))
    b = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input2)
    b = BatchNormalization()(b)
    b = MaxPooling1D(strides=STRIDES, pool_size=PS)(b)
    b = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(b)
    b = BatchNormalization()(b)
    b = MaxPooling1D(strides=STRIDES, pool_size=PS)(b)
    b = Flatten()(b)
    b = Dropout(DROPOUT)(b)

    input3 = Input(shape=(128, 1))
    c = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input3)
    c = BatchNormalization()(c)
    c = MaxPooling1D(strides=STRIDES, pool_size=PS)(c)
    c = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(c)
    c = BatchNormalization()(c)
    c = MaxPooling1D(strides=STRIDES, pool_size=PS)(c)
    c = Flatten()(c)
    c = Dropout(DROPOUT)(c)

    input4 = Input(shape=(128, 1))
    d = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input4)
    d = BatchNormalization()(d)
    d = MaxPooling1D(strides=STRIDES, pool_size=PS)(d)
    d = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(d)
    d = BatchNormalization()(d)
    d = MaxPooling1D(strides=STRIDES, pool_size=PS)(d)
    d = Flatten()(d)
    d = Dropout(DROPOUT)(d)

    input5 = Input(shape=(128, 1))
    e = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input5)
    e = BatchNormalization()(e)
    e = MaxPooling1D(strides=STRIDES, pool_size=PS)(e)
    e = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(e)
    e = BatchNormalization()(e)
    e = MaxPooling1D(strides=STRIDES, pool_size=PS)(e)
    e = Flatten()(e)
    e = Dropout(DROPOUT)(e)

    input6 = Input(shape=(128, 1))
    f = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input6)
    f = BatchNormalization()(f)
    f = MaxPooling1D(strides=STRIDES, pool_size=PS)(f)
    f = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(strides=STRIDES, pool_size=PS)(f)
    f = Flatten()(f)
    f = Dropout(DROPOUT)(f)

    input7 = Input(shape=(128, 1))
    g = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input7)
    g = BatchNormalization()(g)
    g = MaxPooling1D(strides=STRIDES, pool_size=PS)(g)
    g = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(g)
    g = BatchNormalization()(g)
    g = MaxPooling1D(strides=STRIDES, pool_size=PS)(g)
    g = Flatten()(g)
    g = Dropout(DROPOUT)(g)

    input8 = Input(shape=(128, 1))
    h = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input8)
    h = BatchNormalization()(h)
    h = MaxPooling1D(strides=STRIDES, pool_size=PS)(h)
    h = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(h)
    h = BatchNormalization()(h)
    h = MaxPooling1D(strides=STRIDES, pool_size=PS)(h)
    h = Flatten()(h)
    h = Dropout(DROPOUT)(h)

    input9 = Input(shape=(128, 1))
    i = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input9)
    i = BatchNormalization()(i)
    i = MaxPooling1D(strides=STRIDES, pool_size=PS)(i)
    i = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(i)
    i = BatchNormalization()(i)
    i = MaxPooling1D(strides=STRIDES, pool_size=PS)(i)
    i = Flatten()(i)
    i = Dropout(DROPOUT)(i)

    input10 = Input(shape=(128, 1))
    j = Conv1D(FIRST, HEIGHT1, activation="relu",
               kernel_initializer="uniform")(input10)
    j = BatchNormalization()(j)
    j = MaxPooling1D(strides=STRIDES, pool_size=PS)(j)
    j = Conv1D(SECOND, HEIGHT2, activation="relu",
               kernel_initializer="uniform")(j)
    j = BatchNormalization()(j)
    j = MaxPooling1D(strides=STRIDES, pool_size=PS)(j)
    j = Flatten()(j)
    j = Dropout(DROPOUT)(j)

    input11 = Input(shape=(100,))
    k = Dense(256, activation="relu", kernel_initializer="uniform")(input11)
    k = Dense(128, activation="relu", kernel_initializer="uniform")(k)
    k = Dense(30, activation="relu", kernel_initializer="uniform")(k)
    k = Dropout(0.25)(k)

    merged = Concatenate()([a, b])
    merged = Concatenate()([merged, c])
    merged = Concatenate()([merged, d])
    merged = Concatenate()([merged, e])
    merged = Concatenate()([merged, f])
    merged = Concatenate()([merged, g])
    merged = Concatenate()([merged, h])
    merged = Concatenate()([merged, i])
    merged = Concatenate()([merged, j])
    merged = Concatenate()([merged, k])
    merged = Dense(30, activation="relu", kernel_initializer="uniform")(merged)
    merged = Dropout(0.25)(merged)

    output = Dense(9, activation="softmax",
                   kernel_initializer="uniform")(merged)
    # testout = Dense(9, activation="softmax", kernel_initializer="uniform")(layer0)
    # model = Model([input1, input2, input3, input4, input5, input6, input7, input8, input9, input10], output)
    model = Model([input1, input2, input3, input4, input5, input6,
                   input7, input8, input9, input10, input11], output)
    # model = Model(input0, testout)
    return model


def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None, groups=y['group_id'], sets=None):

    oof = np.zeros((len(X), 9))
    prediction = np.zeros((len(X_test), 9))
    scores = []
    feature_importance = pd.DataFrame()
    
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, groups)):
        print('Fold', fold_n, 'started at', time.ctime())
        # print('Train Index: ', train_index, 'Valid Index: ', valid_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=10000, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric='multi_logloss',
                      verbose=5000, early_stopping_rounds=50)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(
                X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist,
                              early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict_proba(X_valid)
            score = accuracy_score(y_valid, y_pred_valid.argmax(1))
            print(f'Fold {fold_n}. Accuracy: {score:.4f}.')
            print('')

            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(
                iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type == 'conv':
            train, test = sets
            # train = train.values.reshape(3810, 128, 10, 1)
            train, valid = train[train_index], train[valid_index]

            X_train_conv = [train[:,i] for i in range(10)]
            X_train_conv.append(X_train.values)
            X_valid_conv = [valid[:, i] for i in range(10)]
            X_valid_conv.append(X_valid.values)
            X_test_conv = [test[:, i] for i in range(10)]
            X_test_conv.append(X_test.values)

            model.compile(loss="categorical_crossentropy",
                          optimizer="adam", metrics=["accuracy"])

            model.fit(X_train_conv, np.eye(9)[y_train],
                      validation_data=(X_valid_conv, np.eye(9)[y_valid]),
                      epochs=20, shuffle=True, class_weight="balanced")

            y_pred_valid = model.predict(X_valid_conv)
            y_pred = model.predict(X_test_conv)
            # test_predictions+= model.predict(te)/20

        oof[valid_index] = y_pred_valid
        scores.append(accuracy_score(y_valid, y_pred_valid.argmax(1)))

        prediction += y_pred

        if model_type == 'lgb' or model_type == 'sklearn':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[
                feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            return oof, prediction, feature_importance
        return oof, prediction

    else:
        return oof, prediction

def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):
    cm = confusion_matrix(truth, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', size=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
train_df.replace(-np.inf, 0, inplace=True)
train_df.replace(np.inf, 0, inplace=True)
test_df.replace(-np.inf, 0, inplace=True)
test_df.replace(np.inf, 0, inplace=True)

# Conv
# model = conv_model()
# # Prediction and training
# oof_conv, prediction_conv = train_model(X=train_df, X_test=test_df, y=y['surface'], model=model, model_type='conv', sets=(TRAIN, TEST))
# print("CONV")
# print(confusion_matrix(y['surface'], oof_conv.argmax(1)))

# LightGBM params
params = {'num_leaves': 123,
          'min_data_in_leaf': 12,
          'objective': 'multiclass',
          'max_depth': 21,
          'learning_rate': 0.04680350949723872,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8933018355190274,
          "bagging_seed": 11,
          "verbosity": -1,
          'reg_alpha': 0.9498109326932401,
          'reg_lambda': 0.8058490960546196,
          "num_class": 9,
          'nthread': -1,
          'min_split_gain': 0.009913227240564853,
          'subsample': 0.9027358830703129
         }
         
params_new={'boosting_type': 'gbdt',
 'colsample_bytree': 0.85,
 'learning_rate': 0.1,
 'max_bin': 512,
 'max_depth': -1,
 'metric': 'multi_error',
 'min_child_samples': 8,
 'min_child_weight': 1,
 'min_split_gain': 0.5,
 'nthread': 3,
 'num_class': 9,
 'num_leaves': 31,
 'objective': 'multiclass',
 'reg_alpha': 0.8,
 'reg_lambda': 1.2,
 'scale_pos_weight': 1,
 'subsample': 0.7,
 'subsample_for_bin': 200,
 'subsample_freq': 1}         
oof_lgb, prediction_lgb, feature_importance = train_model(X=train_df, X_test=test_df, y=y['surface'], params=params_new, model_type='lgb', plot_feature_importance=True)
# Confusion matrix
# plot_confusion_matrix(y['surface'], oof_lgb.argmax(1), le.classes_)
print("LGB")
print(confusion_matrix(y['surface'], oof_lgb.argmax(1)))

# SVC
# model = SVC(probability=True)
# oof_svc, prediction_svc = train_model(X=train_df, X_test=test_df, y=y['surface'], params=None, model_type='sklearn', model=model)
# print("SVM")
# print(confusion_matrix(y['surface'], oof_svc.argmax(1)))

# Random Forest
# model = RandomForestClassifier(n_estimators=800, n_jobs=-1)
# oof_rf, prediction_rf = train_model(X=train_df, X_test=test_df, y=y['surface'], params=None, model_type='sklearn', model=model)
# print("RF")
# print(confusion_matrix(y['surface'], oof_rf.argmax(1)))

# X_train, X_valid, y_train, y_valid = train_test_split(train_df, y['surface'], test_size=0.2, stratify=y['surface'])
# eli5.show_weights(model, targets=[0, 1], feature_names=list(X_train.columns), top=40, feature_filter=lambda x: x != '<BIAS>')

# Submission
sub['surface'] = le.inverse_transform(prediction_lgb.argmax(1))
sub.to_csv('lgb_sub.csv', index=False)
# sub['surface'] = le.inverse_transform(prediction_svc.argmax(1))
# sub.to_csv('svc_sub.csv', index=False)
# sub['surface'] = le.inverse_transform(prediction_rf.argmax(1))
# sub.to_csv('rf_sub.csv', index=False)
# sub['surface'] = le.inverse_transform(prediction_conv.argmax(1))
# sub.to_csv('conv_sub.csv', index=False)
# sub['surface'] = le.inverse_transform((prediction_lgb + prediction_rf).argmax(1))
# sub.to_csv('blend.csv', index=False)


