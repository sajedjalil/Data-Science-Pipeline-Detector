# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# coding:utf-8
# !/usr/bin/env python
# coding: utf-8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
# ===========================================================
# Library
# ===========================================================
import os
import gc
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager
import time
import scipy as sp
import matplotlib.gridspec as gridspec
import random
from functools import partial
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
# import torch
import warnings
from _datetime import datetime
import os
from tqdm import tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import json

pd.set_option('display.max_columns', 1000)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set_color_codes(palette='deep')
import os
import pickle
from itertools import chain
from sklearn.feature_selection import VarianceThreshold
import psutil
import os
# numpy and pandas for data manipulation
import pandas as pd
# model used for feature importances
# utility for early stopping with a validation set
from sklearn.model_selection import train_test_split
# visualizations
import seaborn as sns
# memory management
import gc
from itertools import chain


def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))


time_now = datetime.now().strftime("%Y%m%d_%H%M%S")


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def safe_mk(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


logger = get_logger(time_now)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0} s')


# ===========================================================
# Config
# ===========================================================
# PARENT_DICT = '../input/data-science-bowl-2019/'
PARENT_DICT = '/kaggle/input/data-science-bowl-2019/'
# PARENT_DICT = '../../data/'
df_path_dict = {'train': PARENT_DICT + 'train.csv',
                'test': PARENT_DICT + 'test.csv',
                'train_labels': PARENT_DICT + 'train_labels.csv',
                'specs': PARENT_DICT + 'specs.csv',
                'sample_submission': PARENT_DICT + 'sample_submission.csv'}
OUTPUT_DICT = ''
ID = 'installation_id'
TARGET = 'accuracy_group'
SEED = 42
seed_everything(seed=SEED)
N_FOLD = 5
Fold = GroupKFold(n_splits=N_FOLD)
multip = False
online = False
gen_training_features = True
has_features = False
output_dir = 'output_dir/'
split_dir = 'split_dir/'
safe_mk(output_dir)
safe_mk(split_dir)


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def quadratic_weighted_kappa(self, y_hat, y):
        return cohen_kappa_score(y_hat, y, weights='quadratic')

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        ll = self.quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        return self.coef_['x']


class RegClsCvModel(object):
    '''利用回归来做分类模型'''

    def __init__(self):
        self.OUTPUT_DICT = output_dir
        self.safe_mk(self.OUTPUT_DICT)
        self.ID = ID
        self.TARGET = TARGET
        self.time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pass

    def run_single_lightgbm(self, param, train_df, test_df, folds, features, target, fold_num=0, categorical=[]):
        trn_idx = folds[folds.fold != fold_num].index
        val_idx = folds[folds.fold == fold_num].index
        logger.info(f'len(trn_idx) : {len(trn_idx)}')
        logger.info(f'len(val_idx) : {len(val_idx)}')

        if categorical == []:
            trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                                   label=target.iloc[trn_idx])
            val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                                   label=target.iloc[val_idx])
        else:
            trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                                   label=target.iloc[trn_idx],
                                   categorical_feature=categorical)
            val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                                   label=target.iloc[val_idx],
                                   categorical_feature=categorical)

        oof = np.zeros(len(train_df))
        predictions = np.zeros(len(test_df))
        num_round = 10000
        clf = lgb.train(param,
                        trn_data,
                        num_round,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=1000,
                        early_stopping_rounds=100)

        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold_num
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)
        # RMSE
        logger.info(
            "fold{} RMSE score: {:<8.5f}".format(fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx]))))
        # QWK
        optR = OptimizedRounder()
        optR.fit(oof[val_idx], target[val_idx])
        coefficients = optR.coefficients()
        # coefficients = [0.5, 1.5, 2.5]
        logger.info(f"coefficients: {coefficients}")
        qwk_oof = optR.predict(oof[val_idx], coefficients)
        fold_score = self.quadratic_weighted_kappa(qwk_oof, target[val_idx])
        logger.info("fold{} QWK score: {:<8.5f}".format(fold_num, fold_score))

        return oof, predictions, fold_importance_df, fold_score

    def run_kfold_lightgbm(self, param, train, test, folds, features, target, n_fold=5, categorical=[]):
        logger.info(f"================================= {n_fold}fold lightgbm =================================")
        oof = np.zeros(len(train))
        predictions = np.zeros(len(test))
        feature_importance_df = pd.DataFrame()
        fold_score_list = []
        for fold_ in range(n_fold):
            print("Fold {}".format(fold_))
            _oof, _predictions, fold_importance_df, _fold_score = self.run_single_lightgbm(param,
                                                                                           train,
                                                                                           test,
                                                                                           folds,
                                                                                           features,
                                                                                           target,
                                                                                           fold_num=fold_,
                                                                                           categorical=categorical)
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            oof += _oof
            predictions += _predictions / n_fold
            fold_score_list.append(_fold_score)

        # RMSE
        logger.info("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof))))
        # QWK
        optR = OptimizedRounder()
        optR.fit(oof, target)
        coefficients = optR.coefficients()
        # coefficients = [0.5, 1.5, 2.5]
        logger.info(f"coefficients: {coefficients}")
        qwk_oof = optR.predict(oof, coefficients)
        CV_QWK_score = self.quadratic_weighted_kappa(qwk_oof, target)
        logger.info("CV QWK score: {:<8.5f}".format(CV_QWK_score))
        qwk_predictions = optR.predict(predictions, coefficients)
        submission = pd.DataFrame({f"{self.ID}": test[self.ID].values, f"{self.TARGET}": qwk_predictions})
        submission[self.TARGET] = submission[self.TARGET].astype(int)
        submission.to_csv('submission.csv', index=False)
        feature_importance_df.to_csv(self.OUTPUT_DICT + self.time_str + '_feature_importance_df_lightgbm.csv',
                                     index=False)
        # logger.info(f"CV_QWK_score:{CV_QWK_score}, features{'@@@@'.join([ str(_) for _ in features])}")
        logger.info(f"CV_QWK_score:{CV_QWK_score}, len features{len(features)}")
        logger.info(f"=========================================================================================")

        result_dict = {
            'feature_importance_df': feature_importance_df,
            'qwk_oof': qwk_oof,
            'oof': oof,
            'qwk_predictions': qwk_predictions,
            'predictions': predictions,
            'coefficients': coefficients,
            'target': target,
            'CV_QWK_score': CV_QWK_score,
            'fold_score_list': fold_score_list,
            'fold_score_mean': np.mean(fold_score_list),
            'fold_score_std': np.std(fold_score_list),
            'features': features,
            'categorical': categorical,
        }
        self.print_result(result_dict)
        self.save_result_dict(result_dict)
        return result_dict

    def quadratic_weighted_kappa(self, y_hat, y):
        return cohen_kappa_score(y_hat, y, weights='quadratic')

    def plot_result(self, plot_values, title_name):
        '''绘制直方图'''
        f, ax = plt.subplots(figsize=(8, 7))
        # Check the new distribution
        sns.distplot(plot_values, color="b")
        ax.xaxis.grid(False)
        ax.set(ylabel="Frequency")
        ax.set(xlabel=f"{title_name}")
        ax.set(title=f"{title_name} distribution")
        sns.despine(trim=True, left=True)
        # plt.savefig(f'{self.OUTPUT_DICT}{time_now}_{title_name}.png')
        plt.show()
        plt.close()

    def safe_mk(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def print_result(self, result_dict):
        '''打印结果
        '''
        logger.info(f"CV_QWK_score:{result_dict['CV_QWK_score']} ,"
                    f"fold_score_mean:{result_dict['fold_score_mean']} ,"
                    f"fold_score_std:{result_dict['fold_score_std']}")
        self.plot(result_dict)

    def plot(self, result_dict):
        try:
            self.plot_result(result_dict['target'], 'target')
            self.plot_result(result_dict['qwk_oof'], 'qwk_oof')
            self.plot_result(result_dict['oof'], 'oof')
            self.plot_result(result_dict['qwk_predictions'], 'qwk_predictions')
            self.plot_result(result_dict['predictions'], 'predictions')
        except:
            logger.info('bug in plot!')

    def save_result_dict(self, result_dict):
        cv = result_dict['CV_QWK_score']
        m = result_dict['fold_score_mean']
        m_std = result_dict['fold_score_std']
        f_num = len(result_dict['features'])
        save_path = self.OUTPUT_DICT + self.time_str + f'features_{f_num}_cv_{cv}_mean_{m}_std_{m_std}_result_dict.pkl'
        with open(save_path, 'wb') as fw:
            pickle.dump(result_dict, fw, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'result_dict save to {save_path}')

    def load_result_dict(self, save_path):
        with open(save_path, 'rb') as fr:
            result_dict = pickle.load(fr)
            return result_dict


def make_folds(_df, _id, target, fold, group=None, save_path='folds.csv'):
    df = _df.copy()
    if group == None:
        for n, (train_index, val_index) in enumerate(fold.split(df, df[target])):
            df.loc[val_index, 'fold'] = int(n)
    else:
        le = preprocessing.LabelEncoder()
        groups = le.fit_transform(df[group].astype(str).values)
        for n, (train_index, val_index) in enumerate(fold.split(df, df[target], groups)):
            df.loc[val_index, 'fold'] = int(n)

    df['fold'] = df['fold'].astype(int)
    df[[_id, target, 'fold']].to_csv(output_dir + time_now + '_' + save_path, index=None)
    return df[[_id, target, 'fold']]


# ===========================================================
# Feature Engineering
# credits:
# https://www.kaggle.com/ragnar123/simple-exploratory-data-analysis-and-model
# https://www.kaggle.com/gpreda/data-science-bowl-fast-compact-solution
# ===========================================================
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
    return df


def load_df(path, df_name, debug=False):
    if path.split('.')[-1] == 'csv':
        if debug:
            df = pd.read_csv(path, nrows=10000)
        else:
            df = pd.read_csv(path)
    elif path.split('.')[-1] == 'pkl':
        df = pd.read_pickle(path)
    if logger == None:
        print(f"{df_name} shape / {df.shape} ")
    else:
        logger.info(f"{df_name} shape / {df.shape} ")
    return df


def get_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df


def get_column_cnt_dict(train, test, column, cnt_name, need_list=False, need_recompose_name=True):
    """获取某一column的cnt字典"""
    column_list = list(set(train[column].unique()).union(set(test[column].unique())))
    if need_recompose_name:
        column_list = [column + f'_{_}_{cnt_name}' for _ in column_list]
    if not need_list:
        user_column_count = dict(zip(column_list, [0] * len(column_list)))
    else:
        user_column_count = dict()
        for c in column_list:
            user_column_count[c] = []
    return user_column_count


def concat_column_cnt_dict(train, test, column, column_x_list, column_y_list, cnt_name, need_list=False):
    column_list = []
    for c_x in column_x_list:
        for c_y in column_y_list:
            column_list.append(str(c_x) + '_' + str(c_y))
    if not need_list:
        user_column_count = dict(zip([column + f'_{_}_{cnt_name}' for _ in column_list], [0] * len(column_list)))
    else:
        user_column_count = dict()
        for c in [column + f'_{_}_{cnt_name}' for _ in column_list]:
            user_column_count[c] = []
    return user_column_count


def concate_feature(train, test, cate_x, cate_y):
    col_name = f'{cate_x}_{cate_y}'
    train[col_name] = list(map(lambda x, y: col_name + '_' + x + '_' + y + '_cnt', train[cate_x].astype(str),
                               train[cate_y].astype(str)))
    test[col_name] = list(map(lambda x, y: col_name + '_' + x + '_' + y + '_cnt', test[cate_x].astype(str),
                              test[cate_y].astype(str)))
    return train, test

    # train[col_name] = list(map(lambda x, y: x + '_' + y , train[cate_x].astype(str),
    #                            train[cate_y].astype(str)))
    # test[col_name] = list(map(lambda x, y: x + '_' + y , test[cate_x].astype(str),
    #                           test[cate_y].astype(str)))


def get_data_myself(user_sample, assess_titles, win_code, list_of_title, title_count_base, world_count_base, type_count_base,
             event_code_count_base, event_id_count_base, title_event_code_count_base,
             type_event_code_count_base, world_event_code_count_base, title_event_id_count_base,
             type_event_id_count_base, world_event_id_count_base, user_type_durations_base,
             user_world_durations_base, user_title_durations_base, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_type = 0
    user_session_cnt = 0
    user_type_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    # new features: time spent in each type
    last_session_time_sec = 0
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    # time_first_type = float(user_sample['timestamp'].values[0])
    durations = []
    true_attempts_list = []
    false_attempts_list = []
    accuracy_list = []
    session_durations_list = []
    month_list = []
    hour_list = []
    dayofweek_list = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    # with timer('base cnt dict copy'):
    title_count: Dict[str, int] = title_count_base.copy()
    world_count: Dict[str, int] = world_count_base.copy()
    type_count: Dict[str, int] = type_count_base.copy()
    event_code_count: Dict[str, int] = event_code_count_base.copy()
    event_id_count: Dict[str, int] = event_id_count_base.copy()
    title_event_code_count: Dict[str, int] = title_event_code_count_base.copy()
    type_event_code_count: Dict[str, int] = type_event_code_count_base.copy()
    world_event_code_count: Dict[str, int] = world_event_code_count_base.copy()
    title_event_id_count: Dict[str, int] = title_event_id_count_base.copy()
    type_event_id_count: Dict[str, int] = type_event_id_count_base.copy()
    world_event_id_count: Dict[str, int] = world_event_id_count_base.copy()

    # user_type_durations = user_type_durations_base.copy()
    # user_world_durations = user_world_durations_base.copy()
    # user_title_durations = user_title_durations_base.copy()

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_world = session['world'].iloc[0]
        installation_id = session['installation_id'].iloc[0]

        session_durations = int(session['game_time'].iloc[-1] / 1000)
        session_month = session['month'].iloc[0]
        session_hour = session['hour'].iloc[0]
        session_dayofweek = session['dayofweek'].iloc[0]
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'type': 0, 'Assessment': 0, 'Game':0}
            features = user_type_count.copy()
            features['installation_id'] = installation_id
            features['game_session'] = i
            features['session_type'] = session_type
            features['session_world'] = session_world
            features['session_title'] = session_title
            features['user_session_cnt'] = user_session_cnt
            features.update(last_accuracy_title.copy())
            features.update(title_count.copy())
            features.update(world_count.copy())
            features.update(type_count.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_event_code_count.copy())
            features.update(type_event_code_count.copy())
            features.update(world_event_code_count.copy())
            features.update(title_event_id_count.copy())
            features.update(type_event_id_count.copy())
            features.update(world_event_id_count.copy())

            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far
            def sts_feature(name, sts_list):
                if sts_list == []:
                    features[f'{name}_mean'] = 0
                    features[f'{name}_std'] = 0
                    features[f'{name}_max'] = 0
                    features[f'{name}_min'] = 0
                    features[f'{name}_sum'] = 0
                    features[f'last_{name}'] = 0
                else:
                    features[f'{name}_mean'] = np.mean(sts_list)
                    features[f'{name}_std'] = np.std(sts_list)
                    features[f'{name}_max'] = np.max(sts_list)
                    features[f'{name}_min'] = np.min(sts_list)
                    features[f'{name}_sum'] = np.sum(sts_list)
                    features[f'last_{name}'] = sts_list[-1]

            # with timer("duration cnt"):
            sts_feature("durations", durations)
            sts_feature("true_attempts_list", true_attempts_list)
            sts_feature("false_attempts_list", false_attempts_list)
            sts_feature("accuracy_list", accuracy_list)
            sts_feature("session_durations_list", session_durations_list)
            sts_feature("month_list", month_list)
            sts_feature("hour_list", hour_list)
            sts_feature("dayofweek_list", dayofweek_list)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            accuracy = true_attempts / (true_attempts + false_attempts) if (
                                                                                   true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            true_attempts_list.append(true_attempts)
            false_attempts_list.append(false_attempts)
            accuracy_list.append(accuracy)
            features['true_attempts'] = true_attempts
            features['false_attempts'] = false_attempts
            features['accuracy'] = accuracy
            last_accuracy_title['acc_' + session_title] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            def group_sts_feature(sts_list_dict):
                for k, v_list in sts_list_dict.items():
                    features[f'{k}_sum'] = np.sum(v_list) if v_list != [] else 0
                    features[f'{k}_mean'] = np.mean(v_list) if v_list != [] else 0
                    features[f'{k}_max'] = np.max(v_list) if v_list != [] else 0
                    features[f'{k}_min'] = np.min(v_list) if v_list != [] else 0
                    features[f'{k}_std'] = np.std(v_list) if v_list != [] else 0
                    features[f'last_{k}'] = v_list[-1] if v_list != [] else 0

            # group durations features
            # with timer("group durations:"):
            # group_sts_feature(user_type_durations)
            # group_sts_feature(user_world_durations)
            # group_sts_feature(user_title_durations)

            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts + false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                x = str(col) + '_' + str(k) + '_cnt'
                counter[x] += num_of_session_count[k]
            return counter

        def update_counters_two(counter: dict, col: str, col_1: str, col_2: str):
            # with timer('update_counters_two add time:'):
            #     session[col] = str(col) + '_' + session[col_1].astype(str)+\
            #                    '_'+session[col_2].astype(str) + '_cnt'
            # with timer('update_counters_two add time new:'):
            #     session[col] = session[[col_1,col_2]].apply(lambda x:f'{col}_{x[0]}_{x[1]}_cnt',axis=1)
            # with timer('update_counters_two add time new:'):
            #     session[col] = list(map(lambda x, y: col+'_'+x + '_' + y+'_cnt', session[col_1].astype(str), session[col_2].astype(str)))
            # with timer('update_counters_two count time:'):
            num_of_session_count = Counter(session[col])
            # with timer('update_counters_two fill time1:'):
            #     for k in num_of_session_count.keys():
            #         x = str(col) + '_' + str(k) + '_cnt'
            #         counter[x] += num_of_session_count[k]
            # with timer('update_counters_two fill time2:'):
            for k, v in num_of_session_count.items():
                # print(session[col])
                # print(counter)
                # print(k)
                # print(v)
                counter[k] += v

            # for k in num_of_session_count.keys():
            #     x = str(col) + '_' + str(k) + '_cnt'
            #     counter[x] += num_of_session_count[k]

            return counter

        # with timer("base type cnt:"):
        title_count = update_counters(title_count, 'title')
        world_count = update_counters(world_count, 'world')
        type_count = update_counters(type_count, 'type')
        event_code_count = update_counters(event_code_count, 'event_code')
        event_id_count = update_counters(event_id_count, "event_id")

        # with timer("multi type cnt:"):
        title_event_code_count = update_counters_two(title_event_code_count, 'title_event_code', 'title',
                                                     'event_code')
        type_event_code_count = update_counters_two(type_event_code_count, 'type_event_code', 'type',
                                                    'event_code')
        world_event_code_count = update_counters_two(world_event_code_count, 'world_event_code', 'world',
                                                     'event_code')
        title_event_id_count = update_counters_two(title_event_id_count, 'title_event_id', 'title', 'event_id')
        type_event_id_count = update_counters_two(type_event_id_count, 'type_event_id', 'type', 'event_id')
        world_event_id_count = update_counters_two(world_event_id_count, 'world_event_id', 'world', 'event_id')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_type != session_type:
            user_type_count[session_type] += 1
            last_type = session_type

        user_session_cnt += 1
        # user_type_durations[f'type_{session_type}_durations'].append(session_durations)
        # user_world_durations[f'world_{session_world}_durations'].append(session_durations)
        # user_title_durations[f'title_{session_title}_durations'].append(session_durations)
        session_durations_list.append(session_durations)
        month_list.append(session_month)
        hour_list.append(session_hour)
        dayofweek_list.append(session_dayofweek)
        # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    #     if test_set:
    #         return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments


class BaseFeatures(object):

    def __init__(self):

        pass

    def get_features(self, train, test):
        train, test = self._concate_feature(train, test)
        assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
            set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
        # make a list with all the unique 'titles' from the train and test set
        list_of_title = list(set(train['title']).union(set(test['title'])))
        list_of_world = list(set(train['world'].unique()).union(set(test['world'].unique())))
        list_of_type = list(set(train['type'].unique()).union(set(test['type'].unique())))
        list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
        list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
        title_count_base = get_column_cnt_dict(train, test, 'title', 'cnt')
        world_count_base = get_column_cnt_dict(train, test, 'world', 'cnt')
        type_count_base = get_column_cnt_dict(train, test, 'type', 'cnt')
        event_code_count_base = get_column_cnt_dict(train, test, 'event_code', 'cnt')
        event_id_count_base = get_column_cnt_dict(train, test, 'event_id', 'cnt')

        # title_event_code_count_base = concat_column_cnt_dict(train, test, 'title_event_code', list_of_title,list_of_event_code,'cnt')
        # type_event_code_count_base = concat_column_cnt_dict(train, test, 'type_event_code', list_of_type,list_of_event_code,'cnt')
        # world_event_code_count_base = concat_column_cnt_dict(train, test, 'world_event_code', list_of_world,list_of_event_code,'cnt')
        # title_event_id_count_base = concat_column_cnt_dict(train, test, 'title_event_id', list_of_title,list_of_event_id,'cnt')
        # type_event_id_count_base = concat_column_cnt_dict(train, test, 'type_event_id', list_of_type,list_of_event_id,'cnt')
        # world_event_id_count_base = concat_column_cnt_dict(train, test, 'world_event_id', list_of_world,list_of_event_id,'cnt')

        title_event_code_count_base = get_column_cnt_dict(train, test, 'title_event_code', 'cnt',
                                                          need_recompose_name=False)
        type_event_code_count_base = get_column_cnt_dict(train, test, 'type_event_code', 'cnt',
                                                         need_recompose_name=False)
        world_event_code_count_base = get_column_cnt_dict(train, test, 'world_event_code', 'cnt',
                                                          need_recompose_name=False)
        title_event_id_count_base = get_column_cnt_dict(train, test, 'title_event_id', 'cnt', need_recompose_name=False)
        type_event_id_count_base = get_column_cnt_dict(train, test, 'type_event_id', 'cnt', need_recompose_name=False)
        world_event_id_count_base = get_column_cnt_dict(train, test, 'world_event_id', 'cnt', need_recompose_name=False)

        user_type_durations_base = get_column_cnt_dict(train, test, 'type', 'durations', True)
        user_world_durations_base = get_column_cnt_dict(train, test, 'world', 'durations', True)
        user_title_durations_base = get_column_cnt_dict(train, test, 'title', 'durations', True)

        # I didnt undestud why, but this one makes a dict where the value of each element is 4100
        win_code = dict(zip(list_of_title, (4100 * np.ones(len(list_of_title))).astype('int')))
        # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
        win_code['Bird Measurer (Assessment)'] = 4110

        # reduce_train, reduce_test = split_get_train_and_test(train, test)
        reduce_train, reduce_test = self.get_train_and_test(train, test, assess_titles, win_code, list_of_title,
                                                            title_count_base, world_count_base,
                                                            type_count_base,
                                                            event_code_count_base, event_id_count_base,
                                                            title_event_code_count_base,
                                                            type_event_code_count_base, world_event_code_count_base,
                                                            title_event_id_count_base,
                                                            type_event_id_count_base, world_event_id_count_base,
                                                            user_type_durations_base,
                                                            user_world_durations_base, user_title_durations_base)

        return reduce_train, reduce_test

    def get_train_and_test(self, train, test, assess_titles, win_code, list_of_title, title_count_base,
                           world_count_base,
                           type_count_base,
                           event_code_count_base, event_id_count_base,
                           title_event_code_count_base,
                           type_event_code_count_base, world_event_code_count_base,
                           title_event_id_count_base,
                           type_event_id_count_base, world_event_id_count_base,
                           user_type_durations_base,
                           user_world_durations_base, user_title_durations_base):

        if not multip:
            compiled_train = []
            compiled_test = []

            if gen_training_features:
                for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)),
                                                     total=17000):
                    # with timer(f"{ins_id} training data:"):
                    compiled_train += get_data_myself(user_sample, assess_titles, win_code, list_of_title, title_count_base,
                                               world_count_base,
                                               type_count_base,
                                               event_code_count_base, event_id_count_base,
                                               title_event_code_count_base,
                                               type_event_code_count_base, world_event_code_count_base,
                                               title_event_id_count_base,
                                               type_event_id_count_base, world_event_id_count_base,
                                               user_type_durations_base,
                                               user_world_durations_base, user_title_durations_base)

            for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
                # with timer(f"{ins_id} test data:"):
                test_data = get_data_myself(user_sample, assess_titles, win_code, list_of_title, title_count_base,
                                     world_count_base,
                                     type_count_base,
                                     event_code_count_base, event_id_count_base, title_event_code_count_base,
                                     type_event_code_count_base, world_event_code_count_base,
                                     title_event_id_count_base,
                                     type_event_id_count_base, world_event_id_count_base, user_type_durations_base,
                                     user_world_durations_base, user_title_durations_base, test_set=True)

                if gen_training_features:
                    if len(test_data) > 1:
                        compiled_train += test_data[:-1]
                compiled_test.append(test_data[-1])
            reduce_train = pd.DataFrame(compiled_train)
            reduce_test = pd.DataFrame(compiled_test)

        else:
            compiled_train = []
            compiled_test = []
            import multiprocessing
            if gen_training_features:
                pool = multiprocessing.Pool(processes=64)
                train_results = []
                for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)),
                                                     total=17000):
                    train_results.append(
                        pool.apply_async(get_data_myself, (user_sample, assess_titles, win_code, list_of_title,
                                                    title_count_base, world_count_base,
                                                    type_count_base, event_code_count_base,
                                                    event_id_count_base,
                                                    title_event_code_count_base,
                                                    type_event_code_count_base,
                                                    world_event_code_count_base,
                                                    title_event_id_count_base,
                                                    type_event_id_count_base,
                                                    world_event_id_count_base,
                                                    user_type_durations_base,
                                                    user_world_durations_base,
                                                    user_title_durations_base,)))
                pool.close()
                pool.join()
                for res in train_results:
                    compiled_train += res.get()

            pool = multiprocessing.Pool(processes=64)
            test_results = []
            for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
                test_results.append(pool.apply_async(get_data_myself, (user_sample, assess_titles, win_code, list_of_title,
                                                                title_count_base, world_count_base,
                                                                type_count_base, event_code_count_base,
                                                                event_id_count_base,
                                                                title_event_code_count_base,
                                                                type_event_code_count_base,
                                                                world_event_code_count_base,
                                                                title_event_id_count_base,
                                                                type_event_id_count_base,
                                                                world_event_id_count_base,
                                                                user_type_durations_base,
                                                                user_world_durations_base,
                                                                user_title_durations_base, True,)))
            pool.close()
            pool.join()

            for res in test_results:
                test_data = res.get()
                if gen_training_features:
                    if len(test_data) > 1:
                        # pass
                        compiled_train += test_data[:-1]
                compiled_test.append(test_data[-1])
            reduce_train = pd.DataFrame(compiled_train)
            reduce_test = pd.DataFrame(compiled_test)
        return reduce_train, reduce_test

    def _concate_feature(self, train, test, DEBUG=False):
        # concate_feature('title', 'world')
        # concate_feature('type', 'world')
        train, test = concate_feature(train, test, 'title', 'event_code')
        train, test = concate_feature(train, test, 'type', 'event_code')
        train, test = concate_feature(train, test, 'world', 'event_code')
        train, test = concate_feature(train, test, 'title', 'event_id')
        train, test = concate_feature(train, test, 'type', 'event_id')
        train, test = concate_feature(train, test, 'world', 'event_id')
        # concate_feature('event_id', 'event_code')

        train = get_time(train)
        test = get_time(test)
        return train, test


def encode_title(train, test):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
        set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    # train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

# this is the function that convert the raw data into processed features
def get_data_original(user_sample,assess_titles,list_of_event_code,list_of_event_id,
             activities_labels,all_title_event_code,win_code, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0

    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()}
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())

            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts + false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                x = k
                if col == 'title':
                    x = activities_labels[k]
                counter[x] += num_of_session_count[k]
            return counter

        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type

            # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    #     if test_set:
    #         return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments


def get_train_and_test(train, test):
    train, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, \
    list_of_event_id, all_title_event_code = encode_title(train,test)
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data_original(user_sample,assess_titles,list_of_event_code,list_of_event_id,
             activities_labels,all_title_event_code,win_code)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data_original(user_sample,assess_titles,list_of_event_code,list_of_event_id,
             activities_labels,all_title_event_code,win_code, test_set = True)
        if len(test_data)>1:
            compiled_train += test_data[:-1]
        compiled_test.append(test_data[-1])
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_train, reduce_test

# ===========================================================

class FeatureSelectorMyself(object):

    def __init__(self, categoricals, lgb_param, reduce_train, reduce_test, folds, y):

        self.actual_imp_df_file = 'actual_imp_df_file.csv'
        self.null_imp_df_file = 'null_imp_df_file.csv'
        self.target = 'accuracy_group'
        # self.categorical_feats = ['session_title']
        self.categorical_feats = categoricals
        self.lgb_param = lgb_param
        self.reduce_train = reduce_train
        self.reduce_test = reduce_test
        self.folds = folds
        self.y = y
        pass

    def run(self, data, train_features):
        if not os.path.exists(self.actual_imp_df_file) or not os.path.exists(self.null_imp_df_file):
            self.get_actual_and_null_imp_df(data, train_features)
        else:
            self.actual_imp_df = pd.read_csv(self.actual_imp_df_file)
            self.null_imp_df = pd.read_csv(self.null_imp_df_file)
        logger.info('do get_correlation_scores...')
        self.get_correlation_scores()
        logger.info('do get_feature_scores...')
        self.get_feature_scores()
        logger.info('do run_score_performance...')
        self.run_score_performance()
        logger.info('do do_features_threshold_performance...')
        self.do_features_threshold_performance()
        # logger.info('do do_score_features_select...')
        # self.do_score_features_select()
        logger.info('all done!')

    def do_score_features_select(self):
        scores_df_sorted = self.scores_df.sort_values('gain_score', ascending=False)
        gain_score_usefull_features = list(scores_df_sorted[scores_df_sorted['gain_score'] > 0]['feature'])
        logger.info(f'gain_score_usefull_features size:{len(gain_score_usefull_features)}')
        usefull_features = gain_score_usefull_features[:20]
        new_features = gain_score_usefull_features[20:500]
        self.do_feature_select(usefull_features, new_features)

    def do_feature_select(self, usefull_features, new_features, best_cv_or_mean=True):
        cate_features = [_ for _ in usefull_features if _ in self.categorical_feats]
        logger.info('run usefull_features start, len usefull_features:', len(usefull_features))
        result_usefull_features_dict = self.score_feature_selection(usefull_features, cate_features)
        logger.info('run usefull_features done, len usefull_features:', len(usefull_features))
        if best_cv_or_mean:
            score = result_usefull_features_dict[0]
        else:
            score = result_usefull_features_dict[1]
        logger.info(f'usefull_features score:{score}')
        usefull_new_features = []
        useless_features = []
        for i in new_features:
            evaluating_features = usefull_features + usefull_new_features + [i]
            evaluating_cate = [_ for _ in evaluating_features if _ in self.categorical_feats]
            evaluating_result_dict = self.score_feature_selection(evaluating_features, evaluating_cate)
            if best_cv_or_mean:
                loss_score = evaluating_result_dict[0]
            else:
                loss_score = evaluating_result_dict[1]
            logger.info(f'best score:{score}, now feature score:{loss_score}')
            if loss_score > score:
                logger.info('Feature {} is usefull, adding feature to usefull_new_features_list'.format(i))
                usefull_new_features.append(i)
                score = loss_score
            else:
                print('Feature {} is useless'.format(i))
                useless_features.append(i)

        logger.info('The best features are: ' + 'usefull_new_features@@@'.join(usefull_new_features))
        logger.info(f'Our best cohen kappa score is : {score}')
        logger.info(f'best score features:' + 'bestscorefeature@@@'.join(usefull_features + usefull_new_features))
        return usefull_features + usefull_new_features, useless_features

    def score_feature_selection(self, train_features, cat_feats):
        model = RegClsCvModel()
        result_dict = model.run_kfold_lightgbm(self.lgb_param, self.reduce_train, self.reduce_test, self.folds,
                                               train_features, self.y, n_fold=N_FOLD,
                                               categorical=cat_feats)

        return result_dict['CV_QWK_score'], result_dict['fold_score_mean'], result_dict['fold_score_std']

    def run_score_performance(self):
        logger.info('run_score_performance start!')
        split_score_scores_df = self.scores_df.sort_values('split_score', ascending=False)
        gain_score_scores_df = self.scores_df.sort_values('gain_score', ascending=False)
        for threshold in [50, 100, 200, 300, 400, 500, 600, 700]:
            split_feats = list(split_score_scores_df['feature'])[:threshold]
            split_cat_feats = [_ for _ in split_feats if _ in self.categorical_feats]
            gain_feats = list(gain_score_scores_df['feature'])[:threshold]
            gain_cat_feats = [_ for _ in gain_feats if _ in self.categorical_feats]
            logger.info('Results for threshold %3d' % threshold)
            logger.info(f'len split_feats:{len(split_feats)}, len split_cat_feats:{len(split_cat_feats)},'
                        f'len gain_feats:{len(gain_feats)}, len gain_cat_feats:{len(gain_cat_feats)}')
            split_results = self.score_feature_selection(train_features=split_feats, cat_feats=split_cat_feats)
            logger.info(
                '\t SPLIT : oof cv: %.6f mean: %.6f +/- %.6f' % (split_results[0], split_results[1], split_results[2]))
            gain_results = self.score_feature_selection(train_features=gain_feats, cat_feats=gain_cat_feats)
            logger.info(
                '\t GAIN  : oof cv: %.6f mean: %.6f +/- %.6f' % (gain_results[0], gain_results[1], gain_results[2]))
        logger.info('run_score_performance done!')

    def do_features_threshold_performance(self):
        logger.info('do_features_threshold_performance start!')
        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
            split_feats = [_f for _f, _score, _ in self.correlation_scores if _score >= threshold]
            split_cat_feats = [_f for _f, _score, _ in self.correlation_scores if
                               (_score >= threshold) & (_f in self.categorical_feats)]
            gain_feats = [_f for _f, _, _score in self.correlation_scores if _score >= threshold]
            gain_cat_feats = [_f for _f, _, _score in self.correlation_scores if
                              (_score >= threshold) & (_f in self.categorical_feats)]

            logger.info('Results for threshold %3d' % threshold)
            logger.info(f'len split_feats:{len(split_feats)}, len split_cat_feats:{len(split_cat_feats)},'
                        f'len gain_feats:{len(gain_feats)}, len gain_cat_feats:{len(gain_cat_feats)}')
            split_results = self.score_feature_selection(train_features=split_feats, cat_feats=split_cat_feats)
            logger.info(
                '\t SPLIT : oof cv: %.6f mean: %.6f +/- %.6f' % (split_results[0], split_results[1], split_results[2]))
            gain_results = self.score_feature_selection(train_features=gain_feats, cat_feats=gain_cat_feats)
            logger.info(
                '\t GAIN  : oof cv: %.6f mean: %.6f +/- %.6f' % (gain_results[0], gain_results[1], gain_results[2]))
        logger.info('do_features_threshold_performance done!')

    def plot_correlation_scores(self):
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature',
                    data=self.corr_scores_df.sort_values('split_score', ascending=False).iloc[0:50], ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature',
                    data=self.corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:50], ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
        fig.subplots_adjust(top=0.93)

    def get_correlation_scores(self):
        correlation_scores = []
        for _f in self.actual_imp_df['feature'].unique():
            f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        self.correlation_scores = correlation_scores
        self.corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
        try:
            self.plot_correlation_scores()
        except:
            logger.info('bug in plot!')

    def plot_feature_scores(self):
        plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature',
                    data=self.scores_df.sort_values('split_score', ascending=False).iloc[0:20],
                    ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature',
                    data=self.scores_df.sort_values('gain_score', ascending=False).iloc[0:20],
                    ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()

    def get_feature_scores(self):

        feature_scores = []
        for _f in self.actual_imp_df['feature'].unique():
            f_null_imps_gain = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(
                1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(
                1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))
        self.scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
        try:
            self.plot_feature_scores()
        except:
            logger.info('bug in plot!')

    def get_actual_and_null_imp_df(self, data, train_features):
        # Get the actual importance, i.e. without shuffling
        self.actual_imp_df = self.get_feature_importances(data=data, shuffle=False, train_features=train_features)
        self.actual_imp_df_sorted = self.actual_imp_df.sort_values(by='importance_gain', ascending=False)
        self.null_imp_df = self.get_null_importances(data, train_features)
        self.actual_imp_df.to_csv(self.actual_imp_df_file, index=False)
        self.null_imp_df.to_csv(self.null_imp_df_file, index=False)
        try:
            self.display_distributions(self.actual_imp_df, self.null_imp_df, 'Clip')
            self.display_distributions(self.actual_imp_df, self.null_imp_df, 'Mushroom Sorter (Assessment)')
        except:
            logger.info('bug in plot!')

    def get_feature_importances(self, data, shuffle, train_features, seed=None):
        # Gather real features
        # Go over fold and keep track of CV score (train and valid) and feature importances
        # Shuffle target if required
        # Seed the unexpected randomness of this world
        np.random.seed(123)
        y = data[self.target].copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            y = data[self.target].copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'data_random_seed': 2019,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_data_in_leaf': 100,
        }
        # Fit the model
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200,
                        categorical_feature=self.categorical_feats)
        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        return imp_df

    def get_null_importances(self, data, train_features):
        null_imp_df = pd.DataFrame()
        nb_runs = 80
        import time
        start = time.time()
        dsp = ''
        for i in range(nb_runs):
            # Get current run importances
            imp_df = self.get_feature_importances(data=data, shuffle=True, train_features=train_features)
            imp_df['run'] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
            print(dsp, end='', flush=True)
        return null_imp_df

    def display_distributions(self, actual_imp_df_, null_imp_df_, feature_):
        plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values,
                    label='Null importances')
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
                  ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
        plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values,
                    label='Null importances')
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
                  ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
        plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
        # plt.savefig(f"{feature_.replace(' ','')}_feature_importance.png")


class FeatureSelector():
    """
    Class for performing feature selection for machine learning or data preprocessing.

    Implements five different methods to identify features for removal

        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        5. Find low importance features that do not contribute to a specified cumulative feature importance from the gbm

    Parameters
    --------
        data : dataframe
            A dataset with observations in the rows and features in the columns
        labels : array or series, default = None
            Array of labels for training the machine learning model to find feature importances. These can be either binary labels
            (if task is 'classification') or continuous targets (if task is 'regression').
            If no labels are provided, then the feature importance based methods are not available.

    Attributes
    --------

    ops : dict
        Dictionary of operations run and features identified for removal

    missing_stats : dataframe
        The fraction of missing values for all features

    record_missing : dataframe
        The fraction of missing values for features with missing fraction above threshold

    unique_stats : dataframe
        Number of unique values for all features

    record_single_unique : dataframe
        Records the features that have a single unique value

    corr_matrix : dataframe
        All correlations between all features in the data

    record_collinear : dataframe
        Records the pairs of collinear variables with a correlation coefficient above the threshold

    feature_importances : dataframe
        All feature importances from the gradient boosting machine

    record_zero_importance : dataframe
        Records the zero importance features in the data according to the gbm

    record_low_importance : dataframe
        Records the lowest importance features not needed to reach the threshold of cumulative importance according to the gbm


    Notes
    --------

        - All 5 operations can be run with the `identify_all` method.
        - If using feature importances, one-hot encoding is used for categorical variables which creates new columns

    """

    def __init__(self, data, labels=None):

        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')

        self.base_features = list(data.columns)
        self.one_hot_features = None

        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None

        # Dictionary to hold removal operations
        self.ops = {}

        self.one_hot_correlated = False

    def identify_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above `missing_threshold`"""

        self.missing_threshold = missing_threshold

        # Calculate the fraction of missing in each column
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns=
                                                                                                               {
                                                                                                                   'index': 'feature',
                                                                                                                   0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.ops['missing'] = to_drop

        print('%d features with greater than %0.2f missing values.\n' % (
        len(self.ops['missing']), self.missing_threshold))

    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature',
                     0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.
        Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

        Parameters
        --------
        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features
        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients
        """

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:

            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
        len(self.ops['collinear']), self.correlation_threshold))

    def identify_zero_importance(self, task, eval_metric=None,
                                 n_iterations=10, early_stopping=True):
        """

        Identify the features with zero importance according to a gradient boosting machine.
        The gbm can be trained with early stopping using a validation set to prevent overfitting.
        The feature importances are averaged over `n_iterations` to reduce variance.

        Uses the LightGBM implementation (http://lightgbm.readthedocs.io/en/latest/index.html)
        Parameters
        --------
        eval_metric : string
            Evaluation metric to use for the gradient boosting machine for early stopping. Must be
            provided if `early_stopping` is True
        task : string
            The machine learning task, either 'classification' or 'regression'
        n_iterations : int, default = 10
            Number of iterations to train the gradient boosting machine

        early_stopping : boolean, default = True
            Whether or not to use early stopping with a validation set when training


        Notes
        --------

        - Features are one-hot encoded to handle the categorical variables before training.
        - The gbm is not optimized for any particular task and might need some hyperparameter tuning
        - Feature importances, including zero importance features, can change across runs
        """

        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")

        if self.labels is None:
            raise ValueError("No training labels provided.")

        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1,))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        print('Training Gradient Boosting Model\n')

        # Iterate through each fold
        for _ in range(n_iterations):

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)

            else:
                raise ValueError('Task must be either "classification" or "regression"')

            # If training using early stopping need a validation set
            if early_stopping:

                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels,
                                                                                              test_size=0.15,
                                                                                              stratify=labels)

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric=eval_metric,
                          eval_set=[(valid_features, valid_labels)],
                          early_stopping_rounds=100, verbose=-1)

                # Clean up memory
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()

            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances[
            'importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop

        print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))

    def identify_low_importance(self, cumulative_importance):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine. As an example, if cumulative
        importance is set to 0.95, this will retain only the most important features needed to
        reach 95% of the total feature importance. The identified features are those not needed.
        Parameters
        --------
        cumulative_importance : float between 0 and 1
            The fraction of cumulative importance to account for
        """

        self.cumulative_importance = cumulative_importance

        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined. 
                                         Call the `identify_zero_importance` method first.""")

        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[
            self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop

        print('%d features required for cumulative importance of %0.2f after one hot encoding.' % (
        len(self.feature_importances) -
        len(self.record_low_importance), self.cumulative_importance))
        print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
                                                                                      self.cumulative_importance))

    def identify_all(self, selection_params):
        """
        Use all five of the methods to identify features to remove.

        Parameters
        --------

        selection_params : dict
           Parameters to use in the five feature selection methhods.
           Params must contain the keys ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']

        """

        # Check for all required parameters
        for param in ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']:
            if param not in selection_params.keys():
                raise ValueError('%s is a required parameter for this method.' % param)

        # Implement each of the five methods
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_collinear(selection_params['correlation_threshold'])
        self.identify_zero_importance(task=selection_params['task'], eval_metric=selection_params['eval_metric'])
        self.identify_low_importance(selection_params['cumulative_importance'])

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)

        print('%d total features out of %d identified for removal after one-hot encoding.\n' % (self.n_identified,
                                                                                                self.data_all.shape[1]))

    def check_removal(self, keep_one_hot=True):

        """Check the identified features before removal. Returns a list of the unique features identified."""

        self.all_identified = set(list(chain(*list(self.ops.values()))))
        print('Total of %d features identified for removal' % len(self.all_identified))

        if not keep_one_hot:
            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:
                one_hot_to_remove = [x for x in self.one_hot_features if x not in self.all_identified]
                print('%d additional one-hot features can be removed' % len(one_hot_to_remove))

        return list(self.all_identified)

    def remove(self, methods, keep_one_hot=True):
        """
        Remove the features from the data according to the specified methods.

        Parameters
        --------
            methods : 'all' or list of methods
                If methods == 'all', any methods that have identified features will be used
                Otherwise, only the specified methods will be used.
                Can be one of ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
            keep_one_hot : boolean, default = True
                Whether or not to keep one-hot encoded features

        Return
        --------
            data : dataframe
                Dataframe with identified features removed


        Notes
        --------
            - If feature importances are used, the one-hot encoded columns will be added to the data (and then may be removed)
            - Check the features that will be removed before transforming data!

        """

        features_to_drop = []

        if methods == 'all':

            # Need to use one-hot encoded data as well
            data = self.data_all

            print('{} methods have been run\n'.format(list(self.ops.keys())))

            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))

        else:
            # Need to use one-hot encoded data as well
            if 'zero_importance' in methods or 'low_importance' in methods or self.one_hot_correlated:
                data = self.data_all

            else:
                data = self.data

            # Iterate through the specified methods
            for method in methods:

                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)

                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])

            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        features_to_drop = list(features_to_drop)

        if not keep_one_hot:

            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:

                features_to_drop = list(set(features_to_drop) | set(self.one_hot_features))

        # Remove the features and return the data
        data = data.drop(columns=features_to_drop)
        self.removed_features = features_to_drop

        if not keep_one_hot:
            print('Removed %d features including one-hot features.' % len(features_to_drop))
        else:
            print('Removed %d features.' % len(features_to_drop))

        return data

    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")

        self.reset_plot()

        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins=np.linspace(0, 1, 11), edgecolor='k', color='red',
                 linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11));
        plt.xlabel('Missing Fraction', size=14);
        plt.ylabel('Count of Features', size=14);
        plt.title("Fraction of Missing Values Histogram", size=16);

    def plot_unique(self):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')

        self.reset_plot()

        # Histogram of number of unique values
        self.unique_stats.plot.hist(edgecolor='k', figsize=(7, 5))
        plt.ylabel('Frequency', size=14);
        plt.xlabel('Unique Values', size=14);
        plt.title('Number of Unique Values Histogram', size=16);

    def plot_collinear(self, plot_all=False):
        """
        Heatmap of the correlation values. If plot_all = True plots all the correlations otherwise
        plots only those features that have a correlation above the threshold

        Notes
        --------
            - Not all of the plotted correlations are above the threshold because this plots
            all the variables that have been idenfitied as having even one correlation above the threshold
            - The features on the x-axis are those that will be removed. The features on the y-axis
            are the correlated features with those on the x-axis

        Code adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        """

        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')

        if plot_all:
            corr_matrix_plot = self.corr_matrix
            title = 'All Correlations'

        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])),
                                                    list(set(self.record_collinear['drop_feature']))]

            title = "Correlations Above Threshold"

        f, ax = plt.subplots(figsize=(10, 8))

        # Diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})

        # Set the ylabels
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]));

        # Set the xlabels
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]));
        plt.title(title, size=14)

    def plot_feature_importances(self, plot_n=15, threshold=None):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.
        Parameters
        --------

        plot_n : int, default = 15
            Number of most important features to plot. Defaults to 15 or the maximum number of features whichever is smaller

        threshold : float, between 0 and 1 default = None
            Threshold for printing information about cumulative importances
        """

        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')

        # Need to adjust number of features if greater than the features in the data
        if plot_n > self.feature_importances.shape[0]:
            plot_n = self.feature_importances.shape[0] - 1

        self.reset_plot()

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.feature_importances.index[:plot_n]))),
                self.feature_importances['normalized_importance'][:plot_n],
                align='center', edgecolor='k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(self.feature_importances.index[:plot_n]))))
        ax.set_yticklabels(self.feature_importances['feature'][:plot_n], size=12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size=16);
        plt.title('Feature Importances', size=18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(1, len(self.feature_importances) + 1)), self.feature_importances['cumulative_importance'],
                 'r-')
        plt.xlabel('Number of Features', size=14);
        plt.ylabel('Cumulative Importance', size=14);
        plt.title('Cumulative Feature Importance', size=16);

        if threshold:
            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
            plt.vlines(x=importance_index + 1, ymin=0, ymax=1, linestyles='--', colors='blue')
            plt.show();

            print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    def reset_plot(self):
        plt.rcParams = plt.rcParamsDefault

# ===========================================================
# main
# ===========================================================
def main():
    DEBUG = False

    with timer('Data Loading'):
        train = load_df(path=df_path_dict['train'], df_name='train', debug=DEBUG)
        if not DEBUG:
            train = reduce_mem_usage(train)
        train_labels = load_df(path=df_path_dict['train_labels'], df_name='train_labels', debug=DEBUG)
        test = load_df(path=df_path_dict['test'], df_name='test', debug=False)
        if not DEBUG:
            test = reduce_mem_usage(test)
        specs = load_df(path=df_path_dict['specs'], df_name='specs')
        sample_submission = load_df(path=df_path_dict['sample_submission'], df_name='sample_submission')
        logger.info(cpu_stats())


    with timer("gen features..."):
        # train.drop(columns=['event_data'],inplace=True)
        # test.drop(columns=['event_data'],inplace=True)
        reduce_train, reduce_test = get_train_and_test(train, test)
#         baseFeatures = BaseFeatures()
#         reduce_train, reduce_test = baseFeatures.get_features(train, test)
        reduce_train.to_csv('reduce_train.csv', index=False)
        reduce_test.to_csv('reduce_test.csv', index=False)
        del train,test
        gc.collect()
        logger.info(cpu_stats())

    with timer('Run lightgbm'):
        # reduce_train = pd.read_csv('/kaggle/input/split-data/reduce_train.csv')
        # reduce_test = pd.read_csv('/kaggle/input/split-data/reduce_test.csv')
        columns = list(set(reduce_train.columns).union(set(reduce_test.columns)))
        replace_dict = {_: str(_).replace(' ', '&&').replace(',', '--') for _ in columns}
        reduce_train.rename(columns=replace_dict, inplace=True)
        reduce_test.rename(columns=replace_dict, inplace=True)
        logger.info(f'train_df shape : {reduce_train.shape}')
        logger.info(f'test_df shape : {reduce_test.shape}')

        # 1.remove loss_rate higher columns
        reduce_train.fillna(0, inplace=True)
        reduce_test.fillna(0, inplace=True)

        lgb_param = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'data_random_seed': SEED,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_data_in_leaf': 100,
        }
        columns = list(set(reduce_train.columns).union(set(reduce_test.columns)))
        label_features = ['accuracy_group', 'installation_id', 'installation_id', 'game_session',
                          'true_attempts', 'false_attempts', 'accuracy']
        categoricals_base = ['session_title', 'session_type', 'session_world',
                             'session_month', 'session_hour', 'session_dayofweek',
                             'session_title_world', 'session_type_world']
        all_features = [_ for _ in columns if _ not in label_features]
        categoricals = [_ for _ in categoricals_base if _ in all_features]
        for column in categoricals:
            if reduce_train[column].dtype == 'object':
                column_set = set(reduce_train[column].unique()).union(set(reduce_test[column].unique()))
                columns_map = dict(zip(list(column_set), np.arange(len(column_set))))
                reduce_train[column] = reduce_train[column].map(columns_map)
                reduce_test[column] = reduce_test[column].map(columns_map)
        X, y = reduce_train[all_features], reduce_train['accuracy_group']
        fs = FeatureSelector(data=reduce_train[all_features], labels=y)
        fs.identify_missing(missing_threshold=0.5)
        fs.identify_single_unique()
        fs.identify_collinear(correlation_threshold=0.975)
        # ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
        train_removed = fs.remove(methods=['missing','single_unique','collinear'])

        all_features = list(train_removed.columns)
        categoricals = [_ for _ in categoricals_base if _ in all_features]

        logger.info(f"all_features:{len(all_features)}")
        folds = make_folds(reduce_train, ID, TARGET, Fold, group='installation_id')
        model = RegClsCvModel()
        result_dict = model.run_kfold_lightgbm(lgb_param, reduce_train, reduce_test, folds, all_features, y, n_fold=N_FOLD,
                                                   categorical=categoricals)

        # f = FeatureSelectorMyself(categoricals, lgb_param, reduce_train, reduce_test, folds, y)
        # f.run(data=reduce_train, train_features=all_features)


if __name__ == '__main__':
    main()