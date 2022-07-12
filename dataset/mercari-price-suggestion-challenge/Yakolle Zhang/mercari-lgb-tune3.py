import pyximport

pyximport.install()

import os
import random

import numpy as np
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
random.seed(10002)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)

from keras import backend

tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import copy
import gc
import os
import string
import warnings
from multiprocessing import current_process, Process, Queue

from datetime import *
from time import time

import pandas as pd
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack, vstack
from scipy.stats import skew, kurtosis
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import inspect
import re
import threading

from pandas import Series

warnings.filterwarnings('ignore')


# -------------------------------------------util---------------------------------------
def get_time_stamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")
    
    
def backup(obj):
    return copy.deepcopy(obj)


def round_float_str(info):
    def promote(matched):
        return str(float(matched.group()) + 9e-16)

    def trim1(matched):
        return matched.group(1) + matched.group(2)

    def trim2(matched):
        return matched.group(1)

    info = re.sub(r'[\d.]+?9{4,}[\de-]+', promote, info)
    info = re.sub(r'([\d.]*?)\.?0{4,}\d+(e-\d+)', trim1, info)
    info = re.sub(r'([\d.]+?)0{4,}\d+', trim2, info)

    return info


def get_valid_function_parameters(func, param_dic):
    all_param_names = list(inspect.signature(func).parameters.keys())

    valid_param_dic = {}
    for param_key, param_val in param_dic.items():
        if param_key in all_param_names:
            valid_param_dic[param_key] = param_val

    return valid_param_dic


def arange(start, end, step):
    arr = []
    ele = start
    while ele < end:
        arr.append(round(ele, 10))
        ele += step
    return arr


def calc_best_score_index(means, stds, mean_std_coeff=(1.0, 1.0), max_optimization=True):
    if max_optimization:
        scores = mean_std_coeff[0] * Series(means) - mean_std_coeff[1] * Series(stds)
        return scores.idxmax()
    else:
        scores = mean_std_coeff[0] * Series(means) + mean_std_coeff[1] * Series(stds)
        return scores.idxmin()


# -------------------------------------------data_util---------------------------------------
def balance(x, y, mode=None, ratio=1.0):
    if mode is not None:
        pos = y[1 == y]
        neg = y[0 == y]
        pos_len = len(pos)
        neg_len = len(neg)
        expect_pos_len = int(neg_len * ratio)
        if pos_len < expect_pos_len:
            if "under" == mode:
                expect_neg_len = int(pos_len / ratio)
                y = pos.append(neg.sample(n=expect_neg_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
            else:
                y = y.append(pos.sample(expect_pos_len - pos_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
        elif pos_len > expect_pos_len:
            if "under" == mode:
                y = neg.append(pos.sample(n=expect_pos_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
            else:
                expect_neg_len = int(pos_len / ratio)
                y = y.append(neg.sample(expect_neg_len - neg_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    return x, y


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


def insample_outsample_split(x, y, train_size=.5, holdout_num=5, holdout_frac=.7, random_state=0, full_holdout=False):
    if isinstance(train_size, float):
        int(train_size * len(y))

    in_ind, out_ind = ShuffleSplit(n_splits=1, train_size=train_size, test_size=None, random_state=random_state).split(
        y).__next__()
    in_x = x[in_ind]
    in_y = y[in_ind]
    out_x = x[out_ind]
    out_y = y[out_ind]

    out_set = []
    for i in range(holdout_num):
        _, h_ind = ShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                random_state=random_state + i).split(out_y).__next__()
        h_x = out_x[h_ind]
        h_y = out_y[h_ind]
        out_set.append((h_x, h_y))

    if full_holdout:
        return in_x, in_y, out_set, out_x, out_y
    return in_x, in_y, out_set


# -------------------------------------------cv util---------------------------------------
def _cv_trainer(learning_model, x, y, cv_set_iter, measure_func, cv_scores, inlier_indices, balance_mode, lock=None,
                fit_params=None):
    local_cv_scores = []
    for train_index, test_index in cv_set_iter:
        train_x = x[train_index]
        train_y = y[train_index]

        if inlier_indices is not None:
            train_y = train_y.loc[np.intersect1d(train_y.index, inlier_indices)]
            train_x = train_x.loc[train_y.index]

        if hasattr(learning_model, 'warm_start') and learning_model.warm_start:
            model = learning_model
        else:
            model = backup(learning_model)
        if balance_mode is not None:
            fit_x, fit_y = balance(train_x, train_y, mode=balance_mode)
        else:
            fit_x, fit_y = train_x, train_y

        if fit_params is None:
            model.fit(fit_x, fit_y)
        else:
            model.fit(fit_x, fit_y, **fit_params)

        test_x = x[test_index]
        test_y = y[test_index]
        test_p = model.predict(test_x)

        cur_score = measure_func(test_y, test_p)
        local_cv_scores.append(cur_score)
        print(cur_score)

    if lock is None:
        cv_scores += local_cv_scores
    else:
        lock.acquire()
        cv_scores += local_cv_scores
        lock.release()


def bootstrap_k_fold_cv_train(learning_model, x, y, statistical_size=30, repeat_times=1, refit=False, random_state=0,
                              measure_func=metrics.accuracy_score, balance_mode=None, kc=None, inlier_indices=None,
                              holdout_data=None, nthread=1, fit_params=None, group_bounds=None):
    if kc is not None:
        k = kc[0]
        c = kc[1]
    else:
        k = int(x.shape[0] / (x.shape[1] * statistical_size))
        if k < 3:
            k = int(x.shape[0] / (statistical_size * 2))
        c = int(np.ceil(statistical_size * repeat_times / k))

    if group_bounds is not None:
        groups = get_groups(y, group_bounds)
    else:
        groups = None

    cv_scores = []
    if nthread <= 1:
        for i in range(c):
            if random_state is not None:
                random_state += i
            if groups is None:
                _cv_trainer(learning_model, x, y, KFold(n_splits=k, shuffle=True, random_state=random_state).split(y),
                            measure_func, cv_scores, inlier_indices, balance_mode, fit_params=fit_params)
            else:
                _cv_trainer(learning_model, x, y,
                            StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(y, groups),
                            measure_func, cv_scores, inlier_indices, balance_mode, fit_params=fit_params)
    else:
        learning_model = backup(learning_model)
        if hasattr(learning_model, 'warm_start'):
            learning_model.warm_start = False
        lock = threading.RLock()

        for i in range(c):
            if random_state is not None:
                random_state += i
            if groups is None:
                cv_set = [(train_index, test_index) for train_index, test_index in
                          KFold(n_splits=k, shuffle=True, random_state=random_state).split(y)]
            else:
                cv_set = [(train_index, test_index) for train_index, test_index in
                          StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(y, groups)]
            batch_size = int(len(cv_set) / nthread) + 1

            tasks = []
            for j in range(nthread):
                cv_set_part = cv_set[j * batch_size: (j + 1) * batch_size]
                t = threading.Thread(target=_cv_trainer, args=(learning_model, x, y, cv_set_part, measure_func,
                                                               cv_scores, inlier_indices, balance_mode, lock,
                                                               fit_params))
                tasks.append(t)
            for t in tasks:
                t.start()
            for t in tasks:
                t.join()

    if holdout_data is not None:
        model = backup(learning_model)
        if inlier_indices is not None:
            y = y.loc[np.intersect1d(y.index, inlier_indices)]
            x = x.loc[y.index]

        if fit_params is None:
            model.fit(x, y)
        else:
            model.fit(x, y, **fit_params)

        holdout_scores = []
        for holdout_x, holdout_y in holdout_data:
            holdout_scores.append(measure_func(holdout_y, model.predict(holdout_x)))

        if refit:
            return cv_scores, holdout_scores, model
        else:
            return cv_scores, holdout_scores
    else:
        if refit:
            model = backup(learning_model)
            if inlier_indices is not None:
                y = y.loc[np.intersect1d(y.index, inlier_indices)]
                x = x.loc[y.index]

            if fit_params is None:
                model.fit(x, y)
            else:
                model.fit(x, y, **fit_params)
            return cv_scores, model
        else:
            return cv_scores


def bootstrap_k_fold_cv_factor(learning_model, x, y, factor_key, factor_values, get_next_elements, factor_table,
                               cv_repeat_times=1, random_state=0, measure_func=metrics.accuracy_score,
                               balance_mode=None, data_dir=None, kc=None, mean_std_coeff=(1.0, 1.0), detail=False,
                               max_optimization=True, inlier_indices=None, holdout_data=None, nthread=1,
                               fit_params=None, group_bounds=None, factor_cache=None):
    if data_dir is not None or factor_cache is not None:
        score_cache = read_cache(learning_model, factor_key, factor_table, data_dir=data_dir, factor_cache=factor_cache)
    else:
        score_cache = {}

    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num

    cv_score_means = []
    cv_score_stds = []
    last_time = int(datetime.now().timestamp())
    for factor_val in factor_values:
        if factor_val not in score_cache:
            try:
                learning_model, x, y, inlier_indices, holdout_data = get_next_elements(learning_model, x, y, factor_key,
                                                                                       factor_val, inlier_indices,
                                                                                       holdout_data)
                cv_scores = bootstrap_k_fold_cv_train(learning_model, x, y, repeat_times=cv_repeat_times,
                                                      random_state=random_state, measure_func=measure_func,
                                                      balance_mode=balance_mode, kc=kc, holdout_data=holdout_data,
                                                      inlier_indices=inlier_indices, nthread=nthread,
                                                      fit_params=fit_params, group_bounds=group_bounds)

                if holdout_data is not None:
                    cv_scores, holdout_scores = cv_scores
                    cv_score_mean = np.mean(cv_scores)
                    cv_score_std = np.std(cv_scores)
                    score_cache[factor_val] = cv_score_mean, cv_score_std, holdout_scores
                else:
                    cv_score_mean = np.mean(cv_scores)
                    cv_score_std = np.std(cv_scores)
                    score_cache[factor_val] = cv_score_mean, cv_score_std
            except Exception as e:
                cv_score_mean = bad_score
                cv_score_std = large_num / 10

                print(e)
        else:
            cache_val = score_cache[factor_val]
            if 3 == len(cache_val):
                cv_score_mean, cv_score_std, holdout_scores = cache_val
            else:
                cv_score_mean, cv_score_std = cache_val
        cv_score_means.append(cv_score_mean)
        cv_score_stds.append(cv_score_std)

        if detail:
            if 'holdout_scores' in dir():
                print('----------------', factor_key, '=', factor_val, ', cv_mean=', cv_score_mean, ', cv_std=',
                      cv_score_std, ', holdout_mean=', np.mean(holdout_scores), ', holdout_std=',
                      np.std(holdout_scores), holdout_scores, '---------------')
            else:
                print('----------------', factor_key, '=', factor_val, ', mean=', cv_score_mean, ', std=', cv_score_std,
                      '---------------')

        if data_dir is not None or factor_cache is not None:
            cur_time = int(datetime.now().timestamp())
            if cur_time - last_time >= 300:
                last_time = cur_time
                write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir,
                            factor_cache=factor_cache)
                if factor_cache is not None:
                    print(factor_cache)
                else:
                    print(get_time_stamp())

    if data_dir is not None or factor_cache is not None:
        write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir, factor_cache=factor_cache)

    best_factor_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                              max_optimization=max_optimization)
    best_factor = factor_values[best_factor_index]
    print('--best factor: ', factor_key, '=', best_factor, ', mean=', cv_score_means[best_factor_index], ', std=',
          cv_score_stds[best_factor_index])

    return best_factor, cv_score_means[best_factor_index], cv_score_stds[best_factor_index]


def read_cache(model, factor_key, factor_table, data_dir=None, factor_cache=None):
    if factor_cache is not None:
        return read_cache_from_memory(model, factor_key, factor_table, factor_cache)
    return read_cache_from_file(model, factor_key, data_dir, factor_table)


def read_cache_from_memory(model, factor_key, factor_table, factor_cache):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    factor_cache_key = type(model).__name__ + '-' + factor_key
    if factor_cache_key in factor_cache:
        cache = factor_cache[factor_cache_key]
        cache_key = round_float_str(str(factor_table).replace(':', '-'))
        if cache_key in cache:
            return cache[cache_key]

    return {}


def read_cache_from_file(model, factor_key, data_dir, factor_table):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    file_path = data_dir + 'cache\\' + type(model).__name__ + '-' + factor_key
    if os.path.exists(file_path):
        with open(file_path, 'r') as cache_file:
            cache_str = cache_file.readline()
            if cache_str:
                cache = eval(cache_str)
                cache_key = round_float_str(str(factor_table).replace(':', '-'))
                if cache_key in cache:
                    return cache[cache_key]
    return {}


def write_cache(model, factor_key, score_map, factor_table, data_dir=None, factor_cache=None):
    if data_dir is not None:
        write_cache_to_file(model, factor_key, score_map, data_dir, factor_table)
    if factor_cache is not None:
        write_cache_to_memory(model, factor_key, score_map, factor_table, factor_cache)


def write_cache_to_memory(model, factor_key, score_map, factor_table, factor_cache):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    if score_map:
        cache = {}
        factor_cache_key = type(model).__name__ + '-' + factor_key
        if factor_cache_key in factor_cache:
            cache = factor_cache[factor_cache_key]

        cache_key = round_float_str(str(factor_table).replace(':', '-'))
        if cache_key in cache:
            cache[cache_key].update(score_map)
        else:
            cache[cache_key] = score_map
        factor_cache[factor_cache_key] = cache


def write_cache_to_file(model, factor_key, score_map, data_dir, factor_table):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    if score_map:
        cache = {}
        file_path = data_dir + 'cache\\' + type(model).__name__ + '-' + factor_key
        if os.path.exists(file_path):
            with open(file_path, 'r') as cache_file:
                cache_str = cache_file.readline()
                if cache_str:
                    cache = eval(cache_str)

        with open(file_path, 'w') as cache_file:
            cache_key = round_float_str(str(factor_table).replace(':', '-'))
            if cache_key in cache:
                cache[cache_key].update(score_map)
            else:
                cache[cache_key] = score_map
            cache_file.write(round_float_str(str(cache)))


def probe_best_factor(learning_model, x, y, factor_key, factor_values, get_next_elements, factor_table, detail=False,
                      cv_repeat_times=1, kc=None, score_min_gain=1e-4, measure_func=metrics.accuracy_score,
                      balance_mode=None, random_state=0, mean_std_coeff=(1.0, 1.0), max_optimization=True, nthread=1,
                      data_dir=None, inlier_indices=None, holdout_data=None, fit_params=None, group_bounds=None,
                      factor_cache=None):
    int_flag = all([isinstance(ele, int) for ele in factor_values])
    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num
    last_best_score = bad_score

    if data_dir is not None or factor_cache is not None:
        score_cache = read_cache(learning_model, factor_key, factor_table, data_dir=data_dir, factor_cache=factor_cache)
    else:
        score_cache = {}

    last_time = int(datetime.now().timestamp())
    while True:
        cv_score_means = []
        cv_score_stds = []
        for factor_val in factor_values:
            if factor_val not in score_cache:
                try:
                    learning_model, x, y, inlier_indices, holdout_data = get_next_elements(learning_model, x, y,
                                                                                           factor_key, factor_val,
                                                                                           inlier_indices, holdout_data)
                    cv_scores = bootstrap_k_fold_cv_train(learning_model, x, y, repeat_times=cv_repeat_times, kc=kc,
                                                          random_state=random_state, measure_func=measure_func,
                                                          balance_mode=balance_mode, inlier_indices=inlier_indices,
                                                          holdout_data=holdout_data, nthread=nthread,
                                                          fit_params=fit_params, group_bounds=group_bounds)

                    if holdout_data is not None:
                        cv_scores, holdout_scores = cv_scores
                        cv_score_mean = np.mean(cv_scores)
                        cv_score_std = np.std(cv_scores)
                        score_cache[factor_val] = cv_score_mean, cv_score_std, holdout_scores
                    else:
                        cv_score_mean = np.mean(cv_scores)
                        cv_score_std = np.std(cv_scores)
                        score_cache[factor_val] = cv_score_mean, cv_score_std
                except Exception as e:
                    cv_score_mean = bad_score
                    cv_score_std = large_num / 10

                    print(e)
            else:
                cache_val = score_cache[factor_val]
                if 3 == len(cache_val):
                    cv_score_mean, cv_score_std, holdout_scores = cache_val
                else:
                    cv_score_mean, cv_score_std = cache_val
            cv_score_means.append(cv_score_mean)
            cv_score_stds.append(cv_score_std)

            if detail:
                if 'holdout_scores' in dir():
                    print('----------------', factor_key, '=', factor_val, ', cv_mean=', cv_score_mean, ', cv_std=',
                          cv_score_std, ', holdout_mean=', np.mean(holdout_scores), ', holdout_std=',
                          np.std(holdout_scores), holdout_scores, '---------------')
                else:
                    print('----------------', factor_key, '=', factor_val, ', mean=', cv_score_mean, ', std=',
                          cv_score_std, '---------------')

            if data_dir is not None or factor_cache is not None:
                cur_time = int(datetime.now().timestamp())
                if cur_time - last_time >= 300:
                    last_time = cur_time
                    write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir,
                                factor_cache=factor_cache)
                    if factor_cache is not None:
                        print(factor_cache)
                    else:
                        print(get_time_stamp())

        if data_dir is not None or factor_cache is not None:
            write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir,
                        factor_cache=factor_cache)

        best_factor_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                                  max_optimization=max_optimization)
        if abs(cv_score_means[best_factor_index] - last_best_score) < score_min_gain or max(cv_score_means) - min(
                cv_score_means) < score_min_gain:
            best_factor = factor_values[best_factor_index]
            print('--best factor: ', factor_key, '=', best_factor, ', mean=', cv_score_means[best_factor_index],
                  ', std=', cv_score_stds[best_factor_index])

            return best_factor, cv_score_means[best_factor_index], cv_score_stds[best_factor_index]
        last_best_score = cv_score_means[best_factor_index]

        factor_size = len(factor_values)
        if max_optimization:
            cur_best_index1 = max(range(factor_size), key=lambda i: cv_score_means[i] + cv_score_stds[i])
            cur_best_index2 = max(range(factor_size), key=lambda i: cv_score_means[i] - cv_score_stds[i])
        else:
            cur_best_index1 = min(range(factor_size), key=lambda i: cv_score_means[i] + cv_score_stds[i])
            cur_best_index2 = min(range(factor_size), key=lambda i: cv_score_means[i] - cv_score_stds[i])

        l = min(cur_best_index1, cur_best_index2) - 1
        r = max(cur_best_index1, cur_best_index2) + 1
        if r >= factor_size:
            r = factor_size
            right_value = factor_values[-1]
            if right_value > 0:
                right_value = round(right_value * 1.5, 10)
            else:
                right_value = round(right_value / 2, 10)
            right_value = int(right_value) if int_flag else right_value
            factor_values.append(right_value)
        if l < 0:
            l = 0
            left_value = factor_values[0]
            if 0 != left_value:
                if left_value > 0:
                    left_value = round(left_value / 2, 10)
                else:
                    left_value = round(left_value * 1.5, 10)
                left_value = int(left_value) if int_flag else left_value
                factor_values.insert(0, left_value)
                r += 1

        step = (factor_values[l + 1] - factor_values[l]) / 2
        step = step if not int_flag else int(np.ceil(step))
        if step <= 1e-10:
            continue

        factor_size = (factor_values[r] - factor_values[l]) / step
        if factor_size < 5:
            step /= 2
        elif factor_size > 16:
            step = (factor_values[r] - factor_values[l]) / 16
        step = step if not int_flag else 1 if step <= 1 else int(step)
        next_factor_values = arange(factor_values[l], factor_values[r] + step, step)
        if factor_values[l + 1] not in next_factor_values:
            next_factor_values.append(factor_values[l + 1])
        if factor_values[r - 1] not in next_factor_values:
            next_factor_values.append(factor_values[r - 1])

        factor_values = sorted(next_factor_values)
        print(factor_values)


# -------------------------------------------tune util---------------------------------------
def tune(model, x, y, init_param, param_dic, measure_func=metrics.accuracy_score, cv_repeat_times=1, data_dir=None,
         balance_mode=None, max_optimization=True, mean_std_coeff=(1.0, 1.0), score_min_gain=1e-4, fit_params=None,
         random_state=0, detail=True, kc=None, inlier_indices=None, holdout_data=None, nthread=1, group_bounds=None,
         factor_cache=None, warm_probe=False):
    def get_next_param(learning_model, train_x, train_y, param_key, param_val, inlier_ids, h_data):
        param = {param_key: param_val}
        learning_model.set_params(**param)
        return learning_model, train_x, train_y, inlier_ids, h_data

    def update_params(params):
        for param_key, param_val in params:
            params = {param_key: param_val}
            model.set_params(**params)

    tune_factor(model, x, y, init_param, param_dic, get_next_param, update_params, cv_repeat_times=cv_repeat_times,
                kc=kc, measure_func=measure_func, balance_mode=balance_mode, max_optimization=max_optimization,
                score_min_gain=score_min_gain, mean_std_coeff=mean_std_coeff, data_dir=data_dir, nthread=nthread,
                random_state=random_state, detail=detail, inlier_indices=inlier_indices, holdout_data=holdout_data,
                fit_params=fit_params, group_bounds=group_bounds, factor_cache=factor_cache, warm_probe=warm_probe)


def tune_factor(model, x, y, init_factor, factor_dic, get_next_elements, update_factors, cv_repeat_times=1, kc=None,
                measure_func=metrics.accuracy_score, balance_mode=None, max_optimization=True, score_min_gain=1e-4,
                mean_std_coeff=(1.0, 1.0), data_dir=None, random_state=0, detail=True, inlier_indices=None,
                holdout_data=None, nthread=1, fit_params=None, group_bounds=None, factor_cache=None, warm_probe=False):
    def rebuild_factor_dic():
        for fk, fv in best_factors:
            fvs = factor_dic[fk]
            fvs_size = len(fvs)
            new_fvs = []
            idx = fvs.index(fv)
            if all([type(ele) in [int, float] for ele in fvs]):
                if idx - 2 >= 0:
                    new_fvs.append(fvs[idx - 2])
                if idx - 1 >= 0:
                    new_fvs.append(fvs[idx - 1])
            else:
                new_fvs.append('grid')
                if idx - 2 >= 0 and fvs[idx - 2] != 'grid':
                    new_fvs.append(fvs[idx - 2])
                if idx - 1 >= 0 and fvs[idx - 1] != 'grid':
                    new_fvs.append(fvs[idx - 1])
            new_fvs.append(fv)
            if idx + 1 < fvs_size:
                new_fvs.append(fvs[idx + 1])
            if idx + 2 < fvs_size:
                new_fvs.append(fvs[idx + 2])
            factor_dic[fk] = new_fvs

    optional_factor_dic = {'measure_func': measure_func, 'cv_repeat_times': cv_repeat_times, 'detail': detail,
                           'max_optimization': max_optimization, 'kc': kc, 'inlier_indices': inlier_indices,
                           'mean_std_coeff': mean_std_coeff, 'score_min_gain': score_min_gain, 'data_dir': data_dir,
                           'holdout_data': holdout_data, 'balance_mode': balance_mode, 'nthread': nthread,
                           'fit_params': fit_params, 'group_bounds': group_bounds, 'factor_cache': factor_cache}

    best_factors = init_factor
    seed_dict = {}
    for i, (factor_key, factor_val) in enumerate(best_factors):
        seed_dict[factor_key] = random_state + i
        factor_values = factor_dic[factor_key]
        if factor_val not in factor_values:
            factor_values.append(factor_val)
            factor_dic[factor_key] = sorted(factor_values)
    last_best_factors = backup(best_factors)

    rebuild_dic_flag = warm_probe
    tmp_hold_factors = []
    last_best_score = 1e10
    cur_best_score = last_best_score
    while True:
        update_factors(best_factors)

        if rebuild_dic_flag:
            rebuild_factor_dic()
        else:
            rebuild_dic_flag = True

        for i, (factor_key, factor_val) in enumerate(best_factors):
            if factor_key not in tmp_hold_factors:
                factor_values = factor_dic[factor_key]
                if all([type(ele) in [int, float] for ele in factor_values]):
                    extra_factor_dic = get_valid_function_parameters(probe_best_factor, optional_factor_dic)
                    best_factor_val, mean, std = probe_best_factor(model, x, y, factor_key, factor_values,
                                                                   get_next_elements, dict(best_factors),
                                                                   random_state=seed_dict[factor_key],
                                                                   **extra_factor_dic)
                    if best_factor_val not in factor_values:
                        factor_values.append(best_factor_val)
                        factor_dic[factor_key] = sorted(factor_values)
                else:
                    extra_factor_dic = get_valid_function_parameters(bootstrap_k_fold_cv_factor, optional_factor_dic)
                    best_factor_val, mean, std = bootstrap_k_fold_cv_factor(model, x, y, factor_key, factor_values,
                                                                            get_next_elements, dict(best_factors),
                                                                            random_state=seed_dict[factor_key],
                                                                            **extra_factor_dic)

                if best_factor_val == factor_val:
                    tmp_hold_factors.append(factor_key)
                else:
                    best_factors[i] = factor_key, best_factor_val
                update_factors([best_factors[i]])
                cur_best_score = mean

                if 1e10 == last_best_score:
                    last_best_score = cur_best_score
        print(best_factors)
        if abs(last_best_score - cur_best_score) < score_min_gain or last_best_factors == best_factors:
            break
        else:
            last_best_score = cur_best_score
        if len(tmp_hold_factors) == len(last_best_factors):
            tmp_hold_factors = []
            last_best_factors = best_factors


# -------------------------------------------get data---------------------------------------
time_records = {}
mode = 1

max_len_dic = {'item_description': 70, 'name': 10}
max_data_size = 800000

name_terms = ['bundle', 'for', 'lularoe', 'pink', 'set', '&', '2', 'lot', 'and', 'new', 'case', 'nike', 'lululemon', 'leggings', 'jordan', '/', "'", 'bag', '-', 'dress', 'shirt', 'size', 'dunn', 'funko', '.', 'black', 'disney', 'gold', '4', '3', 's', 'secret', 'coach', 'of', 'box', '!', 'reserved', 'bundle for', 'jacket', 'vs', 'hold', ',', 'watch', 'top', 'kors', 'boots', 'adidas', 'with', 'apple', 'pokemon', 'shoes', 'purse', 'nintendo', 'vs pink', 'bracelet', 'shorts', 'mini', 'free', 'chanel', 'victoria', '6', 'louis', 'gucci', 'palette', 'vuitton', 'vintage', 'nwt', '5', 'american', '1', 'iphone', 'girl', 'wallet', 'burberry', 'baby', 'the', 'ring', '(', 'leather', 'missing', 'authentic', 'ugg', 'necklace', 'supreme', 'silver', 'kylie', 'jeans', 'missing lularoe', "' s", '7', 'pandora', 'pop', 'body', 'prom', 'diamond', 'white', 'hoodie', '+', 'mac', 'tc', 'games', 'missing bundle', 'gift', 'air', 'cards', 'tote', 'os', 'foundation', 'american girl', 'outfit', 'charger', 'wig', 'brand', 'card', 'blue', 'slime', 'bra', '14k', 'lip', '*', 'sunglasses', 'band', 'pants', 'perfume', 'samsung', 'sticker', 'small', 'men', '10', 'sandals', 'skirt', '21', 'backpack', 'plus', 'charm', 'large', '"', 'lifeproof', 'yeezy', 'patagonia', 'only', 'funko pop', 'stickers', 'mask', 'hunter', 'socks', 'chaco', 'kit', 'beats', 'mk', 'vans', 'ps4', 'women', '8', 'kendra', 'tiffany', 'younique', 'apple watch', 'michael', ')', 'makeup', 'bape', 'scentsy', 'sale', '10k', 'squishy', 'nike nike', 'clothes', 'controller', 'sterling', 'wireless', 'floral', 'prada', 'dust', 'red', 'choker', 'birkenstock', 'brand new', 'earrings', 'hair', 'burch', 'game', '#', 'louboutin', 'sephora', 'dog', 'spade', 'rare', 'bottom', 'xl', 'mug', 'cover', 'lush', 'eyeshadow', 'kate spade', 'mcm', 'matte', 'edition', 'zip', 'cream', 'halloween', 'chacos', 'toms', 'lipstick', 'series', 'sweater', 'tee', 'mario', 'planner', 'tank', 'rae', 'ds', 'pro', 'sz', 'dunn rae', 'bras', 'keychain', 'lilly', 'lipsense', 'versace', 'timberland', 'boys', 'missing bundle for', '[', 'costume', 'puma', 'anastasia', 'yeti', 'face', 'uggs', 'cartier', 'fossil', 'bags', 'collection', 'sarah', 'lotion', 'life', 'love', 'pairs', 'spell', 's secret', 'in', 'silpada', '9', 'handbag', 'paisley', 'blaze', '3ds', 'acacia', 'carly', 'chain', 'macbook', 'shipping', 'hermes', 'one', 'missing for', 'jersey', '11', 'bikini', 'skin', 'tieks', 'coat', 'ipod', 'lv', 'christmas', 'sugar', 'under', 'on', 'nars', 'u', 'by', 'bear', 'listing', 'lace', 'harley', 'dior', 'polo', 'bottoms', 'stone', 'oakley', 'doll', 'lanyard', 'onesie', 'thrasher', 'nmd', 'lularoe lularoe', 'brighton', 'pink bundle', 'medium', 'dustbag', 'book', 'iphone 6', 'crossbody', 'inspired', 'samples', 'brandy', 'james avery', 'bands', 'brush', 'lashes', 'warmer', 'converse', 'prom dress', 'gymshark', 'air jordan', 'boxes', 'contour', 'shakeology', 'toddler', 'becca', 'headphones', 'led', 'suit', 'blush', 'a', 'foamposite', 'light', 'slides', 'custom', 'vest', 'missing new', 'off', 'shirts', 'wii', 'x', 'fidget', 'hat', 'hatchimals', 'alta', 'super', 'iphone 7', 'bebe', 'pack', 'bose', 'missing reserved', 'pink pink', 'brahmin', 'fitbit', 'tyme', 'brown', 'retro', 'lego', 'oil', 'apple iphone', 'ship', 'tag', 'beanie', 'jordans', 'scott', 'k', 'keurig', 'rae dunn', '12', 'high', 'crop', 'charizard', 'style', 'hold for', 'north face', 'clear', 'zelda', 'pokémon', 'boost', 'kay', 'boy', 'monat', 'lot of', 'classic', 'hopper', 'reserved for', 'hatchimal', 'bralette', 'fenty', 'fleece', 'cardigan', 'eyelashes', 'lularoe leggings', 'patch', 'fur', 'pouch', 'sleeve', 'yoga', 'morphe', 'miller', 'curvy', 'pulitzer', 'sherri', 'panties', 'star', 'rose', 'kickee', 'free people', 'mophie', 'tarte', 'beauty', 'rm', 'decal', 'mascara', 'paper', 'cricut', 'julia', 'jewelry', 'zara', 't', 'drawstring', 'sperry', 'to', 'sweatshirt', 'missing hold', 'selena', 'gear', '20', 'sneakers', 'unif', "men '", 'quay', 'touch', 'bath', 'irma', 'amelia', 'fendi', 'unicorn', 'missing iphone', 'tory burch', 'martens', 'switch', 'thong', 'celine', 'movado', 'up', 'girl doll', 'vuitton louis', 'chase', 'camera', 'rainbow', 'travel', 'tsum', 'farsali', 'concealer', 'james', 'rolex', 'nerium', 'sample', 'tommy', 'louis vuitton', 'xbox one', 'bombshell', 'flyknit', 'foamposites', 'lauren', 'missing 3', 'm', 'armour', 'kids', 'living', 'southern', 'tula', 'holder', 'panty', 'pendant', 'sets', 'ultra', 'dot', 'xbox', 'canister', 'proof', 'amazon', 'gloss', 'missing 2', 'gel', 'true religion', 'piece', 'key', 'maker', 'obagi', 'otter', 'pocketbac', 'rollerball', 'foams', 'people', 'teaspoon', 'secret pink', 'books', 'ferragamo', 'stussy', 'gloves', 'green', 'headband', 'moschino', 'coin', 'kendra scott', 'triangl', 'bowl', 'xs', 'blanket', 'belt', 'otterbox', 'decay', 'hipster', 'huarache', 'items', '6s', 'sherpa', 'like', 'nixon', 'machine', 'reborn', 'wars', 'charms', 'gown', 'kobe', 'midori', 'unlocked', 'wedding', 'on hold', 'brushes', 'pullover', 'candy', 'huaraches', 'jolyn', 'add', 'cookies', 'malone', '. 5', 'ultraboost', 'spray', 'battery', 'workout', 'forever', 'robe', 'fleo', 'plated', 'swarovski', 'bracelets', 'la', 'lf', 'metal', 'baseball', 'ford', 'salt', 'girls', 'nikon', 'joy', 'tags', 'miss me', 'glass', 'hobo', 'origami', 'tom', 'ue', 'hourglass', 'vines', 'morgan', 'alo', 'me', 'corral', 'elite', 'knife', 'nes', 'note', 'plexus', 'purple', 'house', 'moana', 'pcs', 'bar', 'russe', 'dorbz', 'mist', 'faced', 'freebird', 'go', 'patches', 'pour', 'tech', '! !', 'obey', 'hot', 'jelly', 'owl', 'reformation', 'lid', 'mega', 'girl american', 'valentino', 'lululemon lululemon', 'duffle', 'eagle', 'roshe', 'ivory ella', '17', 'eye', 'fabletics', 'lumee', 'mimi', 'wen', 'rock revival', 'armani', 'navy', 'stroller', 'toner', 'huge', 'adidas adidas', 'lemons', 'pin', 'freeship', 'frye', 'goyard', 'hill', 'strips', 'brezza', '100', 'chloe', 'ipad', 'randy', 'glasses', 'madewell', 'mouse', 'teapot', 'iphone 6s', 'couture', 'extensions', 'pins', 'neverfull', 'barbie', 'cases', 'eyeliner', 'ralph lauren', '30', 'aveda', 'bluetooth', 'earring', 'mumu', 'nmds', 'carhartt', 'levi', 'mannequin', 'tatcha', 'elephant', 'faux', 'smart', 'rogers', 'sorel', 'controllers', 'infinite', 'legging', 'liner', 'rayban', 'rue21', 'tools', 'boden', 'minifigure', 'tease', 'tshirt', 'samsung galaxy', 'canon', 'figure', 'foreo', 'scarf', 'strap', 'marc jacobs', 'bulova', 'fox', 'g', 'kanani', 'teeki', 'head', 'ipsy', 'kyrie', 'shoe', 'shopping', 'vanity', 'save', 'kate', 'mink', 'ps2', 'vita', 'designer', 'gabbana', 'presto', 'solid', 'vionic', 'vlone', 'abh', 'alice', 'color', 'givenchy', 'lacoste', 'print', 'wristlet', '2k17', 'gamecube', 'nail', 'all', 'jbl', 'free ship', 'long', 'smashbox', 'timberlands', 'battlefield', 'dvd', 'glamglow', 'dyson', 'jujube', 'slim', 'ray', 'real', 'mm', 'limelight', 'nyx', 'missing victoria', 'complete', 'duck', 'itworks', 'moroccan', 'ps3', 'vantel', 'full', 'hydro', 'pageant', 'ultimate', 'drunk', 'moncler', 'cosmetics', '360', 'dansko', 'von', 'azure', 'mckenna', 'movies', 'pair', 'capri', 'clarisonic', 'keen', 'melissa', 'rings', 'sdcc', 'snes', 'stamps', 'swim', 'vault', 'birkenstocks', 'herbalife', 'human', 'ibloom', 'month', 'stella', 'tokidoki', 'victoria secret', 'latisse', 'windbreaker', 'homecoming', 'nose', 'printer', 'ban', 'booties', 'overwatch', 'missing free', 'align', 'flats', 'grips', 'jack', 'mer', 'signed', 'soap', ':', 'galaxy', 'tween', 'aztec', 'senegence', 'secret vs', 'wild', 'funko funko', 'lps', 'jim', 'sony', 'buddy', 'protector', 'skulls', 'pink victoria', 'bellami', 'covergirl', 'longchamp', 'outfits', 'rave', 'studio', 'surge', 'underwear', '15', 'michele', 'slippers', 'sweatsuit', 'signature', 'coach coach', 'heels', '4s', 'bnwt', 'missing vs', 'ape', 'bourke', 'kkw', 'sigma', 'swell', 'hilfiger', 'invicta', 'tie', 'colourpop', 'pc', 'tights', 'mossimo', 'pottery', 'remote', 'superstar', 'chi', 'costa', 'diffuser', 'freshly', 'id', 'kavu', 'kd', 'newborn', 'sanuk', 'stylus', 'blouse', 'button', 'grey', 'kerastase', 'ysl', 'lularoe os', 'coupons', 'herschel', 'luggage', 'wildfox', 'cleats', 'doterra', 'earbuds', 'guerlain', 'joyfolie', 'mixing', 'space', 'speck', 'trilogy', 'wet', 'bling', 'charging', 'hoverboard', 'ty', 'kenzo', 'angelus', 'clutch', "women '", 'balmain', 'carat', 'digital', 'ex', 'vinyl', 'mermaid', 'philosophy', 'pie', 'tubular', 'missing baby', 'bong', 'brow', 'christian', 'colors', 'dolls', 'ergo', 'norwex', 'nume', 'tour', 'true', '0', 'cassie', 'rodan', 'keyboard', 'heat', 'gopro', 'lipsticks', 'home', 'accessories', 'aeropostale', 'elgato', 'gtx', 'hollister', 'lipgloss', 'viseart', 'it works', 'bottle', 'insert', 'justin', 'marvel', 'mystery', 'remover', 'jeane', 'cat', 'skull', 'ball', 'bow', 'flip', 'jerseys', 'lamp', 'surface', 'ulta', 'wholesale', 'bears', 'copic', 'ghd', 'grit', 'mackenzie', 'milani', 'sheet', 'studs', 'ties', 'two', 'womens', 'missing 14k', 'heathered', 'l', 'replacement', 'roshes', 'sleep', 'winter', 'air max', 'apple apple', 'dupe', 'engagement', 'grace', 'romper', 'suede', 'ipad mini', 'missing nike', 'balenciaga', 'tea', 'tilbury', 'slip', 'spinner', 'jamberry', 'vera', 'edelman', 'kart', 'kits', 'speed', 'yugioh', 'jaclyn', 'palettes', 'pocketbacs', 'pump', 'shirley', 'sugarpill', 'missing michael', 'nike air', 'ariat', 'capris', 'ergobaby', 'nwot', 'puni', 'youth', 'missing black', 'sonicare', 'car', 'cloth', 'jovani', 'elegant', 'amiibo', 'chargers', 'dooney', 'oribe', '925', 'bomber', 'chapstick', 'manual', 'plate', 'uptempo', 'fitbit charge', 'balm', 'crossfit', 'denali', 'maternity', 'shadow', 'missing kylie', 'astro', 'cable', 'earphones', 'fire', 'fighter', 'film', 'bowls', 'too faced', 'choo', 'gray', 'keranique', 'soft', 'jeffree star', "missing men '", '84', 'cc', 'harlow', 'incense', 'jaanuu', 'mamaroo', 'phone', 'posite', 'wwe', 'ag', 'agnes', 'haan', 'moto', 'apple iphone 6', 'arizona', 'mattes', 'reserve', 'special', 'stick', 'tray', 'vr', 'wheelie', 'accessory', 'bodycon', 'carrier', 'combo', 'gm', 'lightweight', 'pablo', 'pureology', 'shox', 'gameboy', 'peach', 'silicone', 'stila', 'tory', 'fields', 'pyrex', 'garmin', 'mens', 'seiko', 'tattoo', 'teva', 'missing apple', 'bandeau', 'clearance', 'gimmicks', 'justice', 'morgans', 'protectors', 'sleeveless', 'verizon', 'and the', '4x', 'anthropologie', 'bogs', 'bundled', 'dresses', 'gelish', 'heartgold', 'littmann', 'mp3', 'not', 'private', 'ralph', '1 -', 'missing kate', 'bodiez', 'cup', 'furminator', 'monq', 'valentine', '❤', 'missing jordan', 'neck', 'olaplex', 'old', 'cake', 'party', '%', 'coupon', 'highlighter', 'ladies', 'moccasins', 'piyo', 'sandal', 'scrubs', 'missing brand', 'new !', 'bronzer', 'huraches', 'length', 'owls', 'plunder', 'powder', 'racerback', 'strapless', 'system', 'tight', 'torrid', 'xenoverse', 'missing coach', 'board', 'crew', 'iron', 'lebron', 'riley', '18k', 'braun', 'candle', 'paw', 'saltwater', 'cap', 'coffee', 'football', 'order', 'pen', 'pj', 'calendar', 'cutco', 'glitter', 'google', 'ivivva', 'jibbitz', 'madison', 'player', 'popover', 'rosetta', 'sp', 'speedy', 'tattoos', 'affliction', 'elephants', 'stand', 'pink vs pink', 'limited', 'hearts', '3 .', 'missing fidget', '14kt', 'castle', 'dachshund', 'emblem', 'magnetic', 'pajama', 'short', 'snap', 'of 2', 'anatomy', 'bke', 'console', 'cowboy', 'eqt', 'fan', 'flowerbomb', 'keychains', 'miniature', 'pat', 'wolves', 'missing 1', 'hand', 'legends', 'lily', 'months', 'toy', 'missing pink', 'bangle', 'urban decay', 'playstation', 'hp', 'batman', 'bundles', 'deodorant', 'eyelash', 'maxi', 'melville', 'prestos', '3t', 'arc', 'bvlgari', 'chawa', 'cropped', 'ear', 'giftcard', 'magnets', 'miss', 'miu', 'pillows', 'raches', 's4', 'zoomer', 'bathing suit', '256gb', 'adapter', 'andis', 'armband', 'flour', 'jouer', 'low', 'oz', 'popsocket', 'st', 'wigs', 'yellow', 'laurent', 'pocket', '2t', 'campus', 'down', 'flight', 'graphing', 'panda', 'persnickety', 'read', 'stocking', 'yl', 'autographed', 'cami', 'compact', 'giant', 'gymboree', 'lids', 'millers', 'nicole', 'onzie', 'sandles', 'sport', 'superfly', '200', '500gb', 'ana', 'cleaner', 'ferrari', 'hero', 'jackets', 'keds', 'leopard', 'liquid', 'social', 'leather jacket', 'missing nwt', 'natural', 'chanel chanel', 'sold', '11s', '5th', 'diff', 'price', 'primer', 'sanitizer', 'sponges', 'ti', 'wellington', '300', 'airmax', 'australia', 'cans', 'danskin', 'formal', 'headbands', 'iclicker', 'murad', 'paul', 'perricone', 'robin', 'smash', 'thigh', 'yeezys', 'missing 6', 'supreme supreme', 'aux', 'benefit', 'boxers', 'eleanor', 'from', 'nest', 'packets', 'palace', 'speaker', 'thermal', 'voss', 'black and', 'co', 'cuff', 'echo', 'inserts', 'scrub', 'nike jordan', 'it', 'w', 'sign', '50', 'auto', 'castles', 'keona', 'single', '3ft', 'bookbag', 'garnier', 'instyler', 'jeffree', 'keepall', 'little', 'oops', 'pops', 'sampler', 'thrive', 'missing 4', '32ft', '5ml', 'alike', 'avery', 'blaster', 'cabinet', 'facial', 'jade', 'kitchenaid', 'pumpkin', 'tops', 'pink and', 'neon', 'missing women', 'athleta', 'pillow', 'genes', 'labels', 'play', 'apple ipod', 'pink (', 'cpap', 'dylan', 'frontal', 'hoodies', 'infantino', 'look', 'magista', 'nails', 'polaroid', 'selma', 'sheets', 'sock', 'v', 'anklet', 'antenna', 'casemate', 'decals', 'eyebrows', 'gram', 'koko', 'koozie', 'lots', 'lure', 'metcon', 'mosaic', 'water', 'wildlands', 'wristband', 'xv', 'xxs', 'w /', '1tb', 'denim', 'logo', 'paintball', 'renaissance', 'cars', 'fitch', 'h', 'lingerie', 'ornament', 'v2', '10kt', '2x', 'adult', 'artsy', 'bite', 'chocker', 'dark', 'ecotools', 'elf', 'gap', 'horse', 'huf', 'insanity', 'laptop', 'lebrons', 'lighter', 'picked', 'queen', 'rockstud', 'seduction', 'sutton', 't3', 'wax', 'weekender', 'wristbands', 'scott kendra', '5x', 'beach', 'better', 'brazilian', 'campbell', 'fryer', 'helmet', 'hook', 'mirror', 'moroccanoil', 'ovals', 'razer', 'similar', 'skater', 'smocked', 'spandex', 'spectrum', 'support', 'bath bomb', 'bikini top', 'solid black', 'basics', 'butter', 'clip', 'minnetonka', 'textbook', 'secret vs pink', 'anchors', 'clips', 'doppler', 'fs', 'handkerchief', 'infant', 'laces', 'mafia', 'magnet', 'maison', 'mia', 'nfinity', 'pug', 'season', 'selfie', 'shuffle', 'suits', 'topshop', 'ua', 'wrist', 'missing rae', '25', 'cartridge', 'change', 'cheek', 'curling', 'deluxe', 'figurine', 'florentine', 'jump', 'juniors', 'lea', 'needles', 'powerbeats', 'roth', 'toys', 'work', '250gb', 'mate', 'storks', 'missing 5', 'secret victoria', 'listing for', 'rebel', 'sequin', '34b', 'aunt', 'bh', 'contra', 'decor', 'freeshipping', 'hydroflask', 'llr', 'magic', 'opal', 'pads', 'pencil', 'racer', 'stu', 'table', 'thrones', 'train', 'wang', '12s', 'butterfly', 'canvas', 'chokers', 'display', 'empty', 'frankincense', 'iridescent', 'isagenix', 'koi', 'nikki', 'orange', 'ripndip', 'rug', 'stockings', 'sunshine', 'treatment', 'watches', 'wildflower', 'xhilaration', 'lace up', 'beast', 'crops', 'golf', 'heart', 'hudson', 'jane', 'luminess', 'spiritual', 'strawberry', 'tria', 'trunk', 'goose', 'bradley', 'bcbgmaxazria', 'dsi', 'nycc', 'pioneer', 'sunglass', 'apple ipad', 'sony ps4', '2xl', 'beads', 'cleanser', 'clinique', 'duffel', 'filled', 'furby', 'hamilton', 'janie', 'jeremy', 'lulus', 'pantie', 'rope', 'rustic', 'skool', 'twins', 's secret pink', '2tb', 'codes', 'escada', 'fanny', 'gas', 'giggle', 'gobble', 'merlin', 'nipples', 'rain', 'string', 'sweaters', 'swing', 'urban', 'iphone 5', 'kors purse', 'set of', 'blackhead', 'buttons', 'camo', 'candles', 'glue', 'lash', 'republic', 'snowsuit', 'starbucks', 'mary', 'collective', 'polish', 'seat', 'pink vs', '60', 'cookware', 'cp3', 'gallon', 'harry', 'hurache', 'king', 'marmot', 'ora', 'reebok', 'solo', 'tape', 'wraps', 'missing one', 'arm', 'athletic', 'bean', 'cb', 'ce', 'collie', 'colored', 'disposable', 'flower', 'glossier', 'hooks', 'iwatch', 'juvenate', 'mint', 'mix', 'night', 'picks', 'pillowcase', 'popsockets', 'royal', 'shaker', 'stork', 'tickets', 'tracksuit', 'vince', 'zales', 'one size', 'j', 'pantyhose', 'story', 'nike free', 'blender', 'lancome', 'religion', 'pm', 'alphalete', 'anorak', 'citizen', 'dewalt', 'firm', 'forever21', 'markers', 'mugler', 'suitcase', 'urbeats', 'big star', 'for @', 'lularoe tc leggings', 'ankle', 'bandana', 'boxer', 'dōterra', 'eraser', 'floam', 'highlight', 'jessica', 'le', 'lite', 'moondust', 'mukluks', 'nano', 'pattern', 'projector', 'rest', 'scale', 'sim', 'similac', 'stretchy', 'suave', 'teal', 'turtleneck', 'wahl', 'woof', 'missing girls', 'missing size', 'shoulder bag', 'dunn rae dunn', 'birthday', 'bred', 'changing', 'city', 'creepers', 'motherhood', 'graded', 'johnson', 'accu', 'dolce', 'jumbo', 'missing adidas', 'betsey', 'caviar', 'david', 'elisa', 'jeffrey', 'lacefront', 'package', 'power', 'soul', 'swimsuit', '24hr', '34c', 'bassinet', 'bianka', 'birdhouse', 'catchers', 'eyebrow', 'fierce', 'flu', 'gg', 'hare', 'kc', 'maui', 'milwaukee', 'monitor', 'nylon', 'opium', 'osito', 'reusable', 's1', 'sponge', 'x2', 'black &', 'iphone 5c', 'missing 2x', 'bandage', 'erasers', 'gta', 'hi', 'holiday', 'latex', 'parka', 'phantom', 'prismacolor', 'roger', 'sheer', 'suspenders', 'yurman', 'klein', 'turquoise', 'orbit', 'missing rodan', 'dimensions', 'organizer', 'sk8', 'jordan jordan', 'club', 'denona', 'fix', 'maybelline', 'pepper', 'sabo', 'snapchat', 'vineyard vines', '2k15', 'almay', 'americana', 'antique', 'babyliss', 'bun', 'capture', 'cheap', 'crest', 'duo', 'elsa', 'hanes', 'ivy', 'luxe', 'melts', 'modcloth', 'otk', 'polaroids', 'rimmel', 'seawheeze', 'slipper', 'sprint', 'spyder', 'straws', 'swagger', 'sweatpants', 'tent', 'toddlers', 'tracking', 'trial', 'weeknd', 'yeezus', '64gb', 'brew', 'bros', 'canada', 'cord', 'dish', 'jean', 'lipliner', 'moisturizer', 'quantum', 'regimen', 'starter', 'wiiu', 'fitbit fitbit', 'monster high', 'nintendo pokemon', '@', 'marc', 'diaper bag', 'cage', 'condoms', 'duvet', 'edge', 'eno', 'flash', 'master', 's3', 'used', '2oz', '6th', 'azur', 'breastpump', 'bundel', 'calia', 'carrera', 'frames', 'fridge', 'frozen', 'garanimals', 'geometric', 'gigi', 'headphone', 'holo', 'juvias', 'lynnae', 'material', 'matilda', 'prairie', 'receiving', 'ride', 'rods', 'silisponge', 'sims', 'soccer', 'solution', 'stacy', 'swimming', 'trellis', '6 +', 'nintendo 3ds', 'boo', 'bugaboo', 'cd', 'crb', 'dymo', 'glade', 'hammock', 'jacobs', 'lions', 'masks', 'mic', 'octopus', 'pigments', 'pockets', 'rubber', 'samantha', 'simple', 'sonic', 'sports', 'studios', 'wellie', 'lularoe cassie', 'tank top', 'cute', 'paige', 'silhouette', 'bundle !', 'american apparel', 'huda beauty', 'earthbound', '14k gold', 'psp', 'forever 21', 'cize', 'lokai', 'sk', 'xd', 'missing lush', 'arbonne', 'bongo', 'bye', 'chandelier', 'chat', 'comb', 'cross', 'dictionary', 'dragon', 'flyknits', 'gracie', 'graduation', 'grip', 'grovia', 'herrera', 'horizon', 'huda', 'juvia', 'lorac', 'mansion', 'mineral', 'pax', 'pouches', 'prop', 'redefine', 'rosetti', 'sanitizers', 'shampoo', 'sharpener', 'side', 'tb', 'tool', 'tribal', 'tropez', 'vsx', 'was', 'wing', '( on', 'ban ray', 'nyx nyx', '128gb', '35o', 'alexander', 'baker', 'big', 'blackheads', 'cherokee', 'cupcakes', 'dee', 'devacurl', 'dry', 'euc', 'finger', 'fly', 'foam', 'lenovo', 'needle', 'numbers', 'nutribullet', 'person', 'polka', 'shrek', 'steel', 'striped', 'tanjun', 'thin', 'undies', 'wedges', 'cheetah', 'bourke dooney', "pink victoria '", 'erika', 'hermès', 'michael kors', 'balls', 'billionaire', 'goody', 'keyring', 'roar', 'ssd', 'bachelorette', 'bloom', 'cult', 'day', 'defender', 'dose', 'dryer', 'elegance', 'joico', 'josie', 'nintendogs', 'odd', 'papers', 'portion', 'preowned', 'qupid', 'rags', 'sweets', 'tokyo', 'wash', '13s', '4t', 'automatic', 'chair', 'chevron', 'clothing', 'creeper', 'gamma', 'grande', 'homemade', 'lorna', 'maaji', 'pantene', 'poster', 'shining', 'subculture', 'tartiest', 'trick', 'vneck', 'wreath', 'black leggings', 'bundle of', 'lularoe tc', 'of 3', 'humanity', 'lucky', 'pint', 'sperrys', 'x3', 'dunn reserved', 'sony playstation', 'under armour', 'diaper', 'naked', 'samsung samsung', 'airbrush', 'candylipz', 'fantasies', 'gym', 'mustard', 'rc', 'session', 'soulsilver', 'strip', 'venusaur', 'lularoe julia', 'missing high', '1x', 'asap', 'belly', 'bond', 'clarks', 'constellation', 'dressed', 'filter', 'fit', 'flights', 'flynn', 'grateful', 'harvest', 'harvey', 'instax', 'neva', 'outfitters', 'pochette', 'portable', 'premiere', 'products', 'riding', 'rig', 'scentportable', 'shave', 'shellac', 'tongue', '/ xl', 'lime crime', 'makeup bundle', 'lularoe os leggings', '10ft', '32gb', '40', '4moms', 'bake', 'basketball', 'bed', 'boogie', 'canisters', 'converter', 'coral', 'flop', 'flushed', 'formula', 'groove', 'group', 'hanger', 'hard', 'kollection', 'last', 'leg', 'lisa', 'mj', 'mo', 'multicolored', 'natasha', 'nighty', 'notepad', 'olay', 'opi', 'original', 'peppermint', 'plaid', 'plastic', 'reverse', 'rompers', 'saved', 'scrunchies', 'stencils', 'tall', 'thea', 'wipe', 'womans', 'x6', 'size 4', '10x13', 'ed', 'lama', 'mcqueen', 'moon', 'serum', 'snowboard', 'warm', 'webkinz', 'bikini bottoms', 's7', 'missing 20', 'bookmark', '64', 'de', 'gon', 'n', 'quilted', 'chubbies', 'cx', 'no', 'shot', '2b', '2k14', '32', '501', 'camper', 'carmine', 'chocolate', 'cords', 'crazy', 'dusting', 'eyeglass', 'favorites', 'fx', 'he', 'landyard', 'lanie', 'lucy', 'max', 'persona', 'plug', 'remington', 'skort', 'snow', 'standard', 'sunscreen', 'superstars', 'tapers', 'timewise', '6s plus', 'missing boys', 'scrub top', 'water bottle', 'beachbody', 'borax', 'cashmere', 'circo', 'coaster', 'cosmetic', 'curlers', 'deer', 'dice', 'flops', 'fragrance', 'freestyle', 'havana', 'holos', 'htf', 'knock', 'mermaids', 'merona', 'northface', 'patterned', 'picture', 'random', 'requested', 'shoulder', 'skinny', 'stampin', 'straps', 'stretch', 'tanktop', 'tide', 'tiny', 'usb', 'vacation', 'xi', 'b &', 'missing brown', 'never worn', 'silver jeans', 'summer dress', 'asus', 'proactiv', 'raybans', 'seasons', 'shades', 'spanx', 'woman', 'missing men', 'mercier', 'matilda jane', 'bikinis', 'madden', 'minnie', 'bra bundle', 'a1181', 'badge', 'biker', 'bratz', 'cables', 'cookie', 'dickies', 'experience', 'force', 'julie', 'lilo', 'listings', 'mitchell', 'moms', 'native', 'pilaten', 'poshe', 'pregnancy', 'relax', 'scent', 'shade', 'sprays', '/ 4', 'lularoe randy', 'tiffany &', '16', 'alegria', 'angry', 'arctic', 'birthstone', 'blenders', 'ceremony', 'champagne', 'clark', 'clean', 'combat', 'conditioner', 'cotton', 'cutting', 'deal', 'gba', 'groom', 'handles', 'huggies', 'ice', 'macrame', 'merch', 'mount', 'nautica', 'nightie', 'nutcracker', 'phoebe', 'scalp', 'shiny', 'sleepwear', 'socket', 'soffe', 'stones', 'stripped', 'stripper', 'tab', 'tampons', 'tier', 'vspink', 'wheel', 'x1', 'missing 8', '3d', 'chromebook', 'costumes', 'realtree', 'ripka', 'saige', 'stud', 'tournament', 'madden steve', 'rock', 'tresor', 'double', 'auth', 'casio', 'dime', 'honey', 'hue', 'luca', 'mobile', 'purge', 'times', '16g', 'be', 'bronzers', 'bubble', 'colorful', 'direction', 'exchange', 'frame', 'halter', 'mayari', 'medela', 'mezco', 'ops', 'pacifica', 'paco', 'portofino', 'refrigerator', 'refuge', 'revlon', 'sanita', 'sensitive', 'tetris', 'topps', 'missing fashion', 'missing younique', '34a', 'adjustable', 'alpha', 'bead', 'bedding', 'boos', 'boscia', 'bullet', 'casual', 'circles', 'colosseum', 'contouring', 'crowns', 'cubs', 'design', 'drone', 'earpods', 'express', 'fashion', 'guard', 'havaianas', 'hunger', 'jams', 'josh', 'kaya', 'kiss', 'lanyards', 'man', 'measuring', 'mickey', 'mom', 'or', 'princess', 'raiders', 'razor', 'revival', 'shower', 'sleepers', 'sliders', 'smartwatch', 'spf', 'type', 'unisex', 'volume', 'xbox360', '~', 'blue and', 'galaxy s7', 'kylie cosmetics', 'missing halloween', 'missing toddler', 'cooler', 'pacifiers', '6 plus', 'enfamil', 'lounger', 'sally hansen', 'secret bundle', '! ! !', 'alarm', 'aveeno', 'minis', 'brick', 'buxom', 'camisole', 'comfy', 'containers', 'dy', 'earbud', 'electric', 'fishnet', 'flowers', 'handbook', 'kailijumei', 'kleancolor', 'lamps', 'melon', 'neca', 'pez', 'shorthair', 'society', 'makeup bag', 's bundle', 'ahhhsugarsugar', 'att', 'bailey', 'barre', 'betseyville', 'blending', 'contacts', 'culture', 'dvds', 'dynasty', 'earphone', 'ellen', 'euphoria', 'fleming', 'fountain', 'furious', 'gal', 'hawk', 'hdmi', 'hilton', 'iii', 'ink', 'kanken', 'land', 'lipscrub', 'loved', 'lulu', 'notes', 'over', 'packers', 'post', 'promo', 'remotes', 'route', 'sealed', 'shimmer', 'spectra', 'spice', 'stackable', 'starfish', 'tabs', 'terry', 'trinkets', 'warfare', 'wrap', '2 pairs', 'missing 50', 't shirt', '43', 'blooming', 'cowl', 'dane', 'dash', 'lunch', 'pet', 'prime', 'sleeper', 'well', 'old navy', 'bandit', 'cabi', 'nespresso', 'sexy', 'jimmy', 'petsafe', '️', 'estee lauder', 'conair', 'guitar', '36c', 'bat', 'bottles', 'cakes', 'cereal', 'curler', 'en', 'f21', 'fate', 'final', 'handheld', 'herringbone', 'j3', 'mccartney', 'mcdonald', 'meet', 'pineapple', 'playing', 'relic', 'reversible', 'roper', 'seashell', 'sorcerer', 'strings', '3x', '9m', 'accent', 'bordeaux', 'bulldog', 'central', 'collins', 'dak', 'damier', 'decoration', 'despicable', 'drybar', 'eccc', 'erin', 'fifa', 'fits', 'flawed', 'fulton', 'guards', 'hidden', 'i5', 'journal', 'junior', 'les', 'lightbar', 'link', 'links', 'match', 'mittens', 'notebook', 'nuvi', 'parker', 'pasties', 'performance', 'places', 'posh', 'refill', 'reva', 'sandwich', 'seresto', 'spikes', 'sun', 'target', 'tarteist', 'them', 'third', 'usborne', 'vitamin', 'wallets', 'zebra', 'lularoe disney', 'missing green', '12m', '9x12', 'ambiance', 'bone', 'briefs', 'call', 'capcom', 'dana', 'freddy', 'huk', 'juicer', 'nash', 's6', 'wunderbrow', 'physicians formula', 'carnival', 'gb', 'saint laurent', 'army', 'boyshorts', 'leave', 'patrol', 'scooter', 'sega', '36b', 'banjo', 'faja', 'flashlight', 'frieda', 'fusion', 'guide', 'iaso', 'kat', 'laredo', 'lifts', 'munchkin', 'ribbon', 'sparkling', 'stencil', 'thermostat', 'thirty', 'twilly', 'webcam', 'whitening', 'wunder', 'york', 'cross body', 'flip flops', 'missing beauty', '14', '500', '6m', '7plus', '97', 'accents', 'airwalk', 'animals', 'carey', 'cerave', 'coasters', 'collegiate', 'colombian', 'comfortable', 'comforter', 'comics', 'continental', 'crocs', 'dermablend', 'destash', 'dojo', 'dots', 'exercise', 'fimo', 'girdle', 'haul', 'hooters', 'injustice', 'intimately', 'ju', 'manhattan', 'miracle', 'mommy', 'nba', 'nightlight', 'nova', 'plain', 'pocketbook', 'printed', 'ps1', 'recon', 'revolution', 'rocking', 'seconds', 'shakers', 'shep', 'sparkle', 'tartelette', 'three', 'thumb', 'thunder', 'timex', 'tiro', 'triangle', 'valentines', 'vaseline', 'wardrobe', 'zipper', '- free', 'case iphone', 'pandora pandora', 'batiste', 'boyshort', 'castlevania', 'charlie', 'detector', 'dipbrow', 'dollar', 'guerriero', 'holding', 'jumpsuit', 'ninja', 'parade', 'piko', 'sandels', 'shopkins', 'strawberries', 'wallflower', 'wand', 'way', 'zagg', 'sweat pants', '* reserved', 'galaxy s6', 'garcons', 'no boundaries', 'buckle', 'bunny', 'catsuit', 'claus', 'destiny', 'fork', 'foxy', 'friendship', 'knives', 'mercurial', 'metallic', 'packs', 'silence', 'siriano', 'skins', 'sweats', 'vittadini', 'wacom', 'wheels', 'coach wristlet', 'high top', '00g', '7s', '7th', 'agent', 'angled', 'bahama', 'balloons', 'beetlejuice', 'bill', 'blitz', 'brace', 'cannon', 'care', 'cassette', 'children', 'chroma', 'cluster', 'cocoa', 'cowboys', 'creek', 'decorative', 'emoji', 'exfoliating', 'false', 'flexible', 'infrared', 'inspirational', 'inspiron', 'jam', 'kelli', 'koala', 'kohls', 'labs', 'letter', 'loose', 'lowest', 'magazine', 'masters', 'official', 'ollie', 'ovo', 'pave', 'pendleton', 'phones', 'pixel', 'planets', 'pumas', 'savage', 'seeds', 'serve', 'shaped', 'soothe', 'sparkly', 'stamped', 'sundress', 'trefoil', 'trolls', 'zeeland', 'athletica lululemon', 'dri fit', 'it cosmetics', 'missing under', 'screen protector', 'tote bag', 'autograph', 'basic', 'bees', 'denise', 'disneys', 'epic', 'fiesta', 'four', 'justfab', 'micheal', 'minkoff', 'penny', 'ratchet', 'saturn', 'shock', 'skullcandy', 'stuffed', 'tmobile', 'ud', 'young', 'jordan retro', 'credit', 'tumi', 'tsum tsum', 'irmas', 'panasonic', 'pocus', 'bendel', 'carters', 'attachment', 'back', 'baked', 'champ', 'chery', 'colour', 'genuine', 'glowkit', 'heated', 'pjs', 'polishes', 'postcard', 'rifle', 'sense', 'styling', 'tanzanite', 'toothpaste', 'western', 'wheat', 'melville brandy', 'missing custom', 'sterling silver', '1998', '1pc', '250', '72', 'b', 'background', 'bronze', 'bts', 'cement', 'corps', 'curry', 'dockers', 'eileen', 'flex', 'ghost', 'goggles', 'gypsy', 'laser', 'lunchbox', 'mcstuffins', 'melee', 'ordinary', 'parts', 'piercing', 'piggy', 'ping', 'polarized', 'polos', 'portrait', 'ranger', 'rebecca', 'running', 'sak', 'silly', 'tempered', 'tpu', 'trio', 'tuff', 'ufo', 'unlined', 'wish', 'yogi', 'yum', 'russe charlotte', '4th', 'addiction', 'bodysuits', 'chief', 'ck', 'coochy', 'figurines', 'fingerless', 'fitted', 'flannel', 'individual', 'jewelers', 'judge', 'ln', 'pallet', 'plush', 'ripped', 'spider', 'topper', 'tupac', 'velvet', 'wallace', 'wipes', 'zaful', '2 piece', 'disney princess', 'skinny jeans', 'mice', 'storage', 'for [', 'makeup brush', 'secret victoria secret', 'works', 'aucoin', 'harman', 'kinder', 'tan', 'unlock', '18', 'bad', 'evenflo', 'hangers', 'manga', 'pad', 'time', 'make up', '120', '4pcs', '567', 'adhesive', 'baron', 'best', 'blushes', 'caps', 'cheese', 'chunk', 'common', 'count', 'coverup', 'dermalogica', 'fab', 'feel', 'flynit', 'fun', 'ghosts', 'guc', 'henna', 'hiphugger', 'ivanka', 'jojo', 'jordache', 'kirsten', 'leash', 'luxletic', 'lyte', 'macaroon', 'metroid', 'mixer', 'mushroom', 'nuwave', 'ornaments', 'pencils', 'perfectly', 'personal', 'platform', 'pomade', 'protection', 'psychology', 'puffs', 'reduced', 'rollers', 'rue', 'sew', 'silky', 'souls', 'speeds', 'splitter', 'straw', 'tampon', 'travis', '✨', '. tiffany', 'lularoe irma', 'pop !', '5000', 'adorable', 'arrows', 'bin', 'buds', 'bulk', 'cutie', 'decree', 'diapers', 'dreamer', 'elmers', 'extender', 'eyes', 'generation', 'guy', 'illuminator', 'juicy', 'just', 'kt', 'lbs', 'left', 'liz', 'lounge', 'naturals', 'nordstrom', 'omni', 'paperback', 'poison', 'profile', 'scoop', 'septum', 'slices', 'spongebob', 'state', 'steelbook', 'zombie', '& m', 'cosmetics kylie', 'missing floral', 'of the', 'missing rae dunn', 'agenda', 'deck', 'dye', 'halo', 'lee', 'north', 'platter', 'viktor', 'waterproof', 'shoes size', 'helly', 'henri', 'zanotti', 'drive', 'kiehl', 'burton', 'emperor', 'itty', 'jamie', 'maurices', 'papell', 'smok', '12month', '18m', 'ace', 'agate', 'bathbomb', 'camille', 'camouflage', 'cheeky', 'cheryl', 'coloring', 'construction', 'coozie', 'crayon', 'cubes', 'desert', 'disco', 'enzo', 'essentials', 'explore', 'frances', 'gaming', 'hundreds', 'jax', 'jly', 'karen', 'khaki', 'lavender', 'livestrong', 'oils', 'packet', 'profusion', 'psvita', 'puppy', 'roses', 'ruby', 'ski', 'speak', 'splat', 'tessa', 'toilette', 'toothbrush', 'tungsten', 'turtle', 'university', 'unused', 'you', 'zippy', 'half zip', 'kors michael', 'lularoe carly', 'missing grey', '160', 'access', 'any', 'beige', 'bieber', 'compression', 'cork', 'damask', 'deva', 'durable', 'global', 'gun', 'harness', 'hats', 'infinity', 'jergens', 'jordana', 'latch', 'lootcrate', 'midnight', 'mine', 'movie', 'n64', 'pass', 'pigment', 'reduction', 'shopkin', 'stacey', 'tiger', 'trainer', 'unity', '/ 2', 'ray ban', 'tempered glass', 'bbw', 'champion', 'lighters', 'estee', 'california', 'doug', 'gently', 'jwoww', 'mariah', 'siege', 'skyward', 'smores', 'vigoss', 'in 1', 'lego lego', 'yoga pants', '2008', '2y', 'amp', 'av', 'betula', 'bic', 'blackberry', 'blur', 'bm', 'cato', 'corset', 'dodger', 'doom', 'eden', 'everyday', 'fake', 'kotex', 'navajo', 'necklaces', 'non', 'ogx', 'olympus', 'open', 'outlet', 'pajamas', 'patriots', 'peacoat', 'phat', 'pride', 'primal', 'rack', 'regular', 'robot', 'santa', 'shiseido', 'snowboarding', 'solids', 'spiderman', 'spread', 'stain', 'stopper', 'stream', 'synthetic', 'techniques', 'ult', 'nintendo super', 'ugg boots', 'v neck', "victoria '", '10ml', '36dd', '70', 'acg', 'ammo', 'area', 'ashton', 'bass', 'bay', 'biore', 'bird', 'caroline', 'charge', 'combined', 'coverage', 'creams', 'dora', 'easter', 'figures', 'frankie', 'geneva', 'grail', 'heroes', 'ii', 'insulated', 'jellies', 'jolly', 'joni', 'legend', 'lipglosses', 'mademoiselle', 'marilyn', 'monaco', 'oval', 'papaya', 'parallel', 'pit', 'quartz', 'retired', 'rugby', 'shrug', 'storybook', 't25', 'tears', 'therapy', 'tone', 'umgee', 'unicorns', 'utility', 'vineyard', 'xx', 'bikini set', 'bobbi brown', 'in the', 'missing lace', 'missing tc', '4gb', 'comme', 'cuban', 'jansport', 'lg', 'american eagle', 'coach purse', 'long sleeve', 'roads', 'lilly pulitzer', 'onsie', 'paint', 'rapture', '32oz', 'ballet', 'crews', 'durango', 'everlast', 'hugo', 'seamless', 'sg', 'spartina', 'splatoon', 'triple', 'yamaha', '89', 'advance', 'anklets', 'blocks', 'brewers', 'brilliance', 'cartridges', 'chan', 'circle', 'cowgirl', 'crayons', 'dancing', 'dslr', 'ecko', 'egg', 'feeding', 'flowy', 'flux', 'future', 'glove', 'gomez', 'gooseberry', 'isabelle', 'izzy', 'john', 'kim', 'metallica', 'microphone', 'minecraft', 'minion', 'mixr', 'murphy', 'olivia', 'rampage', 'reflex', 'rocky', 's925ss', 'shine', 'trinket', 'visionnaire', 'adidas originals', 'missing authentic', 'missing the', 'one piece', 'smart watch', '09', '34ddd', 'bright', 'bumgenius', 'cell', 'cellphone', 'claire', 'cynthia', 'darts', 'device', 'diana', 'foil', 'forces', 'frenchie', 'gs', 'hamster', 'heeled', 'iphone5', 'lauder', 'novel', 'og', 'pacifier', 'pick', 'pinup', 'please', 'postage', 'purchase', 'ribbed', 'rogue', 's8', 'shark', 'spaghetti', 'temptation', 'tresemme', 'triangles', 'visor', 'ymi', 'iphone 5s', 'missing hot', 'missing it', 'tommy hilfiger', '55mm', 'avenue', 'extreme', 'fast', 'gitd', 'headset', 'marciano', 'messy', 'mi', 'millennium', 'purity', 'sling', 'tablet', 'pottery barn', 'appetite', 'funko funko pop', 'lenny', 'works bbw', 'animal', 'blackmilk', 'bridal', 'doctor', 'dummies', 'lust', 'shearling', 'tobacco', '32c', '400', '999', 'alphabet', 'art', 'assassins', 'avengers', 'avon', 'banned', 'bhm', 'bogo', 'broncos', 'carolyn', 'carter', 'cindy', 'cleaning', 'crosley', 'curve', 'evod', 'extra', 'fergie', 'g5', 'gate', 'hologram', 'how', 'kaki', 'legos', 'libby', 'lipkits', 'luck', 'maise', 'marion', 'meow', 'mercer', 'miyake', 'molly', 'orbeez', 'pacsun', 'petals', 'playsuit', 'pure', 'quarter', 'racers', 'realistic', 'rookie', 'secert', 'shelby', 'skeleton', 'soundtrack', 'still', 'thermoball', 'toshiba', 'unblemish', 'unzipped', 'vacuum', 'vanessa', 'war', 'zippered', '* *', 'missing 10', 'missing anastasia', 'top bundle', 'missing free ship', '32a', '36ddd', '6x9', '8s', 'abercrombie', 'adam', 'apparel', 'balloon', 'benassi', 'benjamin', 'carryall', 'character', 'coastal', 'collapsible', 'condition', 'curly', 'designs', 'energy', 'everest', 'fear', 'goku', 'goldstone', 'i7', 'idole', 'knockout', 'legacy', 'money', 'nfl', 'nightgown', 'nipple', 'pallets', 'pore', 'protective', 'putty', 'razorback', 're', 'screen', 'shannon', 'shin', 'six', 'snack', 'sweat', 'teepee', 'tin', 'w7', 'whales', 'whbm', 'x5', 'lularoe classic', 'perfect tee', 'crystal', 'directions', 'flatback', 'market', 'puzzle', 'stripe', 'strobe', 'stuart', 'toe', 'towel', 'volcom', 'xbox 360', 'dr', 'bathing', 'weitzman', 'beaded', 'bella', 'claiborne', 'cuffs', 'lock', 'pages', 'kors michael kors', '2gb', '30ml', '49ers', '5sos', '7pc', '9s', 'artisan', 'assn', 'batwing', 'ben', 'beyond', 'bins', 'blotting', 'bumper', 'collectible', 'colorblock', 'comic', 'ethernet', 'hooded', 'kaplan', 'kitten', 'ladybug', 'lippies', 'ls', 'medallions', 'mirrors', 'neoprene', 'nunchuck', 'nuud', 'pan', 'pravana', 'pusheen', 'python', 'raquel', 'satchel', 'spark', 'street', 'swamp', 'tint', 'training', 'turtles', '®', '& white', '2 .', 'burberry burberry', 'faded glory', 'missing *', 'missing rose', 'reserved bundle', "woman '", '1995', 'adventures', 'angelica', 'ap', 'apron', 'apt', 'awareness', 'baf', 'batteries', 'battlefront', 'bib', 'bionic', 'blood', 'bulb', 'cambogia', 'charcoal', 'cnd', 'cooking', 'dracula', 'feet', 'flaws', 'folding', 'griffey', 'hey', 'hippie', 'jeggings', 'kiwi', 'legendary', 'lenses', 'lipkit', 'longhorns', 'looking', 'maggie', 'med', 'naturalizer', 'operated', 'piercings', 'ponytail', 'raptors', 'reaper', 'retin', 'sara', 'size10', 'soda', 'test', 'tips', 'touchscreen', 'tunic', 'varsity', 'vice', 'voltage', 'warriors', 'weight', 'wwf', '! new', ': )', 'lauren conrad', 'missing small', 'missing white', 'missing womens', '42mm', 'bff', 'bloomers', 'cabbage', 'coogi', 'cups', 'guardian', 'leon', 'liplicious', 'pique', 'seahawks', 'speedo', 'kay mary', 'missing [', 'missing mini', 'missing on', 'd', 'tv', 'giuseppe', 'guess', 'hocus', 'waist', 'crumble', 'ebene', 'pipe', '3 pairs', '32x30', 'afterglow', 'amope', 'angela', 'bfyhc', 'birchbox', 'cha', 'chewbacca', 'climalite', 'cone', 'dave', 'dressbarn', 'editions', 'funky', 'gorgeous', 'gshock', 'heartsoul', 'ikea', 'lights', 'lippie', 'mua', 'nailpolish', 'offs', 'pimple', 'pounds', 'release', 'runners', 'smacker', 'sounds', 'stack', 'supplement', 'ted', 'trust', 'tweed', 'virginia', 'worth', 'zirconia', '& bourke', "jordan '", 'kylie jenner', 'maybelline maybelline', 'sale !', '01', '1lb', '7y', 'album', 'algenist', 'anti', 'bearpaw', 'blessed', 'break', 'buy', 'caboodle', 'cameo', 'carbon', 'chalcedony', 'chest', 'cm', 'coconut', 'commemorative', 'compare', 'couples', 'cruz', 'cupid', 'drink', 'elastic', 'eos', 'fabric', 'farm', 'flirty', 'hallows', 'hawaiian', 'highs', 'hitter', 'iconic', 'ionic', 'joggers', 'juggernaut', 'kardashian', 'konami', 'liv', 'lopez', 'luxie', 'midi', 'momma', 'nabi', 'nexus', 'nivea', 'ofra', 'onesize', 'pets', 'platinum', 'purpose', 'reading', 'screens', 'shell', 'shield', 'simpsons', 'skylanders', 'slow', 'snuggie', 'speakers', 'stationary', 'study', 'swatch', 'tamer', 'tested', 'tulip', 'universal', 'value', 'villains', 'vinyasa', 'wifi', 'willow', 'gymshark gymshark', "l '", 'missing blue', 'missing cute', 'new black', 'nwt pink', 'polo ralph', 'zip up', '80', 'alexa', 'botanicals', 'cent', 'first', 'livie', 'make', 'microfiber', 'milk', 'olaf', 'omega', 'oreal', 'poppy', 'samba', '. 0', 'ray -', 'rm ]', 'cache', 'defect', 'tahari', 'vivitar', 'dimes', 'breath', 'country', 'fawn', 'run', 'stress', '✳', 'tennis shoes', '36a', 'asia', 'at', 'beverage', 'bryce', 'charlotte', 'cinderella', 'collared', 'curtains', 'feather', 'fruit', 'glow', 'gross', 'hayden', 'heusen', 'janes', 'kirby', 'mane', 'mascaras', 'mos', 'music', 'nasty', 'nature', 'nickelodeon', 'norelco', 'overalls', 'pewter', 'repellent', 'reward', 'sammy', 'scented', 'sham', 'store', 'swan', 'toothbrushes', 'tracer', 'tracy', 'volumizer', 'disney disney', 'jacobs marc', 'perfect t', 'taylor loft', '14g', '4xl', ']', 'anchor', 'aria', 'bandai', 'barbies', 'biofit', 'both', 'bundl', 'calculator', 'catch', 'checks', 'cheekster', 'chick', 'copy', 'cub', 'desktop', 'digimon', 'distance', 'drifit', 'embroidered', 'firming', 'gizeh', 'glossy', 'gudetama', 'jeweled', 'martins', 'missguided', 'morning', 'nautical', 'oak', 'olympic', 'ombré', 'ons', 'oxfords', 'panels', 'patricia', 'phillies', 'photo', 'pinkblush', 'polymailers', 'puffer', 'refills', 'reindeer', 'rex', 'sailor', 'sandra', 'saturday', 'screwdriver', 'seaside', 'shadows', 'sided', 'sized', 'sky', 'sofa', 'superman', 'sweatshirts', 'tennis', 'tomato', 'unionbay', 'unopened', 'velcro', 'walker', '❣', 'missing gray', 'pink victoria secret', '18650', '45', 'brixton', 'bug', 'coins', 'db', 'drill', 'elves', 'energie', 'pierced', 'solo2', 'liquid lipstick', 'missing 30', 'missing lot', 'missing samsung', '* * *', 'bennett', 'fiestaware', 'qui', '350', 'kershaw', 'temporary', 'fossil fossil', '8x10', 'attention', 'ava', 'borderlands', 'brandt', 'breast', 'chow', 'christina', 'citrine', 'dip', 'dreamcatchers', 'drinking', 'dunk', 'g1', 'hoop', 'keys', 'knitted', 'mrs', 'napkins', 'neutrogena', 'nibble', 'nor', 'place', 'retros', 'rudolph', 'saffiano', 'sdhc', 'shea', 'slouch', 'spoon', 'stacking', 'stadium', 'sydney', 'tees', 'try', 'weatherproof', 'decay urban', 'johnson betsey', 'missing 100', 'missing plus', 'nintendo ds', 'rue 21', '007', '160gb', '39', 'actual', 'against', 'akira', 'apex', 'asics', 'baggies', 'bambi', 'blanc', 'bralettes', 'brooch', 'bulbs', 'cali', 'cheerleading', 'chirp', 'cosmic', 'creed', 'cz', 'dipped', 'discontinued', 'don', 'duct', 'fluff', 'gabriella', 'generic', 'george', 'graphics', 'heads', 'healing', 'hollywood', 'inside', 'kinky', 'laguna', 'lasting', 'lux', 'mass', 'medallion', 'muscle', 'nikes', 'nude', 'opened', 'out', 'overall', 'pallete', 'plz', 'pretty', 'purifying', 'rainbows', 'se', 'sequins', 'shape', 'ships', 'skyrim', 'styrofoam', 'tang', 'tankini', 'tripp', '. l', 'and black', 'bundle (', 'independent lularoe', 'nike lebron', 'nintendo nintendo', 'ambient', 'ashley', 'avent', 'babies', 'bell', 'brady', 'c9', 'cut', 'daiso', 'good', 'gourmet', 'honor', 'optimus', 'plumper', 'pressure', 'skateboarding', 'steelers', 'vanilla', 'yoda', 'body mist', 'missing funko', 'missing harry', 'missing little', 'missing makeup', 'harajuku', 'kiri', 'your', 'rose gold', 'soho', '35', 'beaters', 'enfagrow', 'sleeves', '2007', '6x', 'adrienne', 'banana', 'binoculars', 'candlelight', 'catcher', 'draw', 'droid', 'filters', 'franklin', 'freshener', 'furreal', 'goods', 'husky', 'hypervenom', 'jay', 'jenner', 'luna', 'madagascar', 'mw3', 'nickel', 'ninjago', 'ostrich', 'pals', 'pills', 'precision', 'quick', 'rattle', 'religious', 'rising', 'sconces', 'sinful', 'swiftly', 'toro', 'valve', '. .', 'gift set', "girl '", 's under', '07', '2004', '2005', '32b', 'aeo', 'asphalt', 'bamboo', 'boohoo', 'boutique', 'bralett', 'breakfast', 'bubbler', 'bulldogs', 'center', 'championship', 'charming', 'closure', 'cuticle', 'cutter', 'daytrip', 'divided', 'doe', 'dualshock', 'dunks', 'effects', 'eucalyptus', 'exotic', 'extras', 'fp', 'goat', 'hallmark', 'heather', 'honest', 'infuser', 'ka', 'kimono', 'locker', 'mar', 'melted', 'meme', 'moments', 'motocross', 'mousse', 'nip', 'nudes', 'nursing', 'nüüd', 'pacific', 'pamela', 'perfect', 'photocard', 'pinball', 'plan', 'press', 'prince', 'proactive', 'request', 's5', 'sack', 'sc', 'songs', 'spalding', 'stamp', 'thank', 'tip', 'tower', 'trouser', 'tutu', 'ukulele', 'warmers', 'wayfarer', 'wiz', 'bundle -', 'hollister hollister', 'missing os', 'new pink', 'size 11', '2017', 'atari', 'belle', 'catalina', 'flask', 'gilligan', 'hybrid', 'jewell', 'keepsake', 'min', 'ps', 'receipt', 'roll', 'school', 'skates', 'sparrow', 'sub', 'taxi', 'text', 'wool', 'and white', 'beats by', 'people free people', '3g', '58', 'nascar', 'eyeshadows', 'calvin', 'show', '1st', '5w', 'belkin', 'brallete', 'creative', 'critters', 'deadly', 'elaina', 'globe', 'haven', 'indiana', 'is', 'knit', 'muji', 'pakistani', 'pinocchio', 'sands', 'sweet', 'tac', 'tikes', 'towels', 'tree', 'trina', 'worthington', 'x9', 'lularoe xs', 'missing large', 'nwt vs', "' s secret", '27', '4g', 'aa', 'ad', 'alicia', 'aurora', 'aussie', 'bibs', 'bigger', 'braclet', 'brain', 'brownie', 'cart', 'chips', 'cinnamon', 'conceal', 'concert', 'dance', 'daughter', 'dawn', 'deep', 'diy', 'doggy', 'dory', 'eau', 'fila', 'flavored', 'fluffy', 'fob', 'fuzzy', 'gentlease', 'grandma', 'gwen', 'hannah', 'hers', 'hurley', 'injection', 'jars', 'jillian', 'kancan', 'khakis', 'lang', 'leotard', 'lighted', 'lincoln', 'lomo', 'luke', 'margot', 'mlp', 'monroe', 'muse', 'nora', 'osh', 'partial', 'pb', 'pfg', 'philips', 'pomegranate', 'potion', 'prismatic', 'prison', 'radial', 'rayne', 'remastered', 'retool', 'return', 'royale', 'saucer', 'seen', 'shoreline', 'simply', 'smile', 'strength', 'studded', 'supplies', 'underground', 'vaulted', 'vip', 'watt', 'wo', 'wooden', '✅', '/ 7', 'nike men', 'poly mailers', 'size 7', 'burch tory burch', '22', '69', 'assorted', 'chaps', 'crawford', 'dd', 'drake', 'farmhouse', 'festival', 'fjallraven', 'hills', 'hippy', 'horseshoe', 'item', 'kid', 'micro', 'minions', 'moly', 'paracord', 'pastel', 'path', 'sesame', 'smoked', 'spiked', 'spirit', 'tek', 'wave', '/ 8', 'matte lip', 'missing ✨', 'nintendo wii', 'pants size', 'size xl', 'tc lularoe', 'bonus', 'cameras', 'dangle', 'faded', 'kitchen', 'lysol', 'mercedes', 'rumble', 'singer', 'v3', 'missing jamberry', 'moose toys', 'plus case', 'size 10', 'mcdonalds', 'channel', 'cube', 'fr', 'giraffe', 'grand', 'interactive', 'lc', 'loca', 'longboard', 'martian', 'maya', 'megazord', 'optix', 'sat', 'scrap', 'serpentina', 'yu', 'missing faux', 'nail polish', 'xbox xbox', '❤ ️', "men ' s", '150', '2003', '68', '6month', 'aloe', 'amulet', 'angel', 'anne', 'avia', 'barely', 'bianca', 'bodysuit', 'bolero', 'bowie', 'bride', 'brita', 'catherine', 'clock', 'college', 'cooling', 'cupcake', 'desi', 'doggie', 'downshifter', 'embroidery', 'eminem', 'espadrilles', 'favor', 'flirt', 'flow', 'gibson', 'handle', 'heidi', 'hell', 'hyperwarm', 'infamous', 'initial', 'ix', 'jet', 'joan', 'josefina', 'july', 'kandi', 'ku', 'lansinoh', 'laugh', 'lis', 'lp', 'mainstays', 'marisa', 'miscellaneous', 'monkey', 'monokini', 'mont', 'need', 'pearl', 'pixar', 'playtex', 'potter', 'prestige', 'pugs', 'roche', 'russ', 'service', 'shapes', 'skincare', 'skip', 'spoons', 'sportswear', 'stethoscope', 'styled', 'tamagotchi', 'taz', 'thankful', 'uv', 'vehicles', 'vi', 'virgin', 'whale', 'wire', 'words', 'ya', 'yasss', '- -', 'converse all', 'l .', 'little pony', 'maxi dress', 'missing long', 'missing vintage', 'secret free', 'vans vans', '128', '2014', 'boss', 'chambray', 'chic', 'chip', 'container', 'coolpix', 'dad', 'dallas', 'davids', 'davidson', 'desire', 'dodgers', 'freddys', 'freeman', 'gordon', 'hardy', 'houndstooth', 'inch', 'lacey', 'luxury', 'muppets', 'my', 'organic', 'pea', 'r1', 'report', 'root', 'scrapbooking', 'security', 'statue', 'storm', 'stüssy', 'summers', 'teenage', 'tude', 'turbo', 'wander', 'e .', 'eagle shorts', 'me miss', 'missing nwot', 'pulitzer lilly', 'kat von d', '38ddd', 'allsaints', 'banks', 'conrad', 'dear', 'element', 'gravity', 'pod', 'spencer', 'lip kit', 'missing disney', "victoria ' s"]
desc_terms = ['.', ',', 'and', 'new', '!', 'in', 'for', 'the', 'size', 'with', 'a', '-', 'is', 'to', 'of', 'box', 'condition', "'", 'authentic', ':', 'on', 'missing', 'brand', 'all', 'free', 'used', 'it', '2', 'i', '[', '1', 'shipping', 'no', 'worn', '/', 'this', 'you', 'or', 'brand new', 'bundle', 'never', 'are', ')', 'price', '3', 's', 'not', 'black', 'pink', 'great', '(', 'but', '&', 'one', '"', '5', 'comes with', 'will', 'color', 'only', 'charger', 'set', 'from', 'small', '4', 'bag', 'have', 'my', 'has', 'as', 'good', "' s", '6', 'your', 'leather', 'be', 'comes', 'like', '! !', 'tags', 'can', 'free shipping', 'very', 'large', 'gold', '*', '8', 'rare', 'top', 'lularoe', 'medium', '7', 'that', 'firm', 'white', 'dress', 'each', 'out', 'items', 'rm', 'beautiful', 'cute', 'up', 'condition .', 'me', 'once', 'both', 'blue', 'nwt', 'please', 'some', 'perfect', '%', 'just', '10', 'includes', 'nike', 'by', 'original', 'these', 'so', 'full', ']', 'still', 'case', 'wear', 'retail', 't', 'included', '. .', 'more', '. i', 'oz', 'works', 'other', 'times', 'missing brand', 'if', 'save', 'silver', 'new with', ', and', 'been', 'sealed', 'ship', 'any', 'at', 'great condition', 'back', 'lululemon', 'fit', '. size', 'dust bag', 'never used', 'like new', 'm', 'missing new', 'two', 'none', 'super', 'this is', 'missing none', 'they', 'plus', 'jacket', 'leggings', 'sterling', 'on the', 'selling', 'body', 'sold out', 'lot', '. no', 'is a', 'of the', 'opened', 'good condition', 'home', 'card', 'new in', 'them', 'red', 'colors', '. 5', 'never worn', 'pairs', 'for [', 'price is', 'long', '#', '9', 'hair', 'in the', 'pair', 'an', 'excellent', 'light', 'made', 'day', 'edition', 'secret', 'vintage', '1 .', '12', 'see', 'shirt', 'x', 'soft', 'women', 'was', '14k', 'shoes', 'apple', 'missing brand new', 'bought', 'victoria', ': )', 'get', 'cards', 'watch', 'xs', 'everything', '. the', 'sold', "' t", 'nintendo', 'fast', 'fits', 'quality', 'palette', 'funko', 'item', 'with tags', 'necklace', '3 .', 'inside', 'pants', 'zip', 'gift', 'unlocked', 'also', 'high', 'use', '100', 'mini', 'total', 'travel', '. this', 'over', 'listing', 'sale', 'battery', 'inches', '. it', 'little', 'vs', 'make', 'off', '+', 'game', '! ! !', 'scratches', 'sample', 'for a', 'green', 'skin', 'xl', 'iphone', 'dunn', 'worn once', 'disney', 'sephora', 'men', 'months', 'american', 'new .', 'front', 'love', 'jeans', 'brown', 'games', 'ring', 'with a', 'gorgeous', 'in box', 'diamond', 'new ,', 'louis', 'unicorn', 'girl', '•', 'dust', '0', 'offers', 'collection', 'tag', 'size small', 'check', 'baby', '* *', 'need', '] .', 'style', 'stains', 'available', 'strap', 'flaws', 'lace', 'i have', 'paid', 'bracelet', 'jordan', 'smoke', 'if you', 'stickers', 'do', 'purchase', '20', 'ask', 'about', 'bottom', 'missing size', 'wallet', 'pieces', 'controller', 'material', 'retails', "i '", 'zipper', 'pictures', 'real', 'face', 'will be', 'shirts', 'extra', 'for the', '2 .', 'pockets', 'used .', 'new !', 'package', 'smoke free', 'well', 'l', 'excellent condition', 'nice', '[ rm', 'few', 'left', 'cream', 'picture', '. comes', 'free ship', 'hoodie', 'os', 'tracking', 'sizes', 'time', 'pop', 'purple', 'when', 'lip', 'piece', 'questions', 'is for', 'box .', 'tote', 'phone', 'complete', 'coach', "it '", 'exclusive', 'boxes', '16', '. all', '. . .', 'look', 'custom', 'length', 'ships', 'don', '11', 'grey', 'big', '$', 'chain', 'bags', 'navy', 'brush', 'book', 'in good', 'and a', 'outfit', 'purse', 'buy', 'condition !', 'shorts', 'sleeve', 'before', 'slime', 'print', 'size medium', '18', 'bottles', 'limited', 's secret', 'adidas', 'too', 'makeup', 'in great', 'mint', 'screen', 'sell', 'sets', 'boots', 'great for', 'clean', 'any questions', 'chanel', 'new never', '21', 'full size', 'has a', 'new with tags', 'shipping .', 'listings', 'tc', 'in a', 'side', 'size :', 'packs', 'there', 'set of', 'brand new with', 'shown', ', but', 'can be', 'perfume', 'come', '. price', 'kit', '] each', '15', 'bottle', 'ml', 'receipt', '30', 'kylie', 'needs', 'super cute', 'she', 'it is', 'shade', 'pet', 'tops', '100 %', 'under', '10k', 'protector', 'new in box', 'condition ,', 'as a', 'tested', 'no box', 'hard to find', 'jewelry', 'hand', 'perfect condition', 'have a', 'size 6', 'is firm', 'senegence', 'pokemon', 'missing this', '% authentic', 'offer', 'never been', 'thank', 'no free', 'sweater', '1 -', 'bra', 'books', 'nwot', "women '", 'kendra', 'new and', 'this is a', 'lipstick', 'plastic', 'foundation', '1 /', 'airpods', 'pocket', 'brand new in', '925', 'gucci', 'handmade', 'eyeshadow', 'bag .', 'i will', 'cotton', 'tarte', '14', 'free home', 'rose', '24', 'without', 'lot of', 'youth', 'shoe', '0 .', 'bnip', 'w', 'pictured', 'they are', 'size large', 'band', 'eye', 'unopened', 'sz', 'washed', 'because', 'perfect for', 'mist', 'photos', 'diamonds', 'tank', 'conditioner', 'tags :', 'gap', 'fast shipping', 'polo', 'size 8', '. comes with', 'these are', 'shipping !', 'camera', 'tee', 'would', 'and the', '50', 'waist', '. will', 'pack', 'adjustable', 'samples', 'orange', 'earrings', 'controllers', 'gloss', ', no', 'bras', 'yellow', 'gray', 'euc', 'may', 'reserved', 'a size', 'it .', 'button', 'a few', 'find', 'same', 'does', 'price firm', 'want', 'home .', 'my other', '. new', 'purchased', 'out of', 'worn .', 'message', '13', 'know', 'low', 'kids', 'bar', ', i', 'pretty', 'pro', 'sheets', '. never', 'shape', 'is in', 'design', 'go', '. 7', 'wig', 'used condition', 'brand new ,', 'store', 'comfortable', 'lv', 'come with', 'sterling silver', 'with the', 'armour', ', never', 'christmas', 'great condition .', '[ rm ]', 'secret pink', 'her', ', size', 'to bundle', 'than', 'from the', 'brand new .', 'around', 'really', ';', '15ml', 'color :', 'with box', 'stone', 'bundle of', 'stamped', 'had', 'skirt', "it ' s", 'last', '. 4', 'pairs of', 'water', 'straps', 'doll', 'thanks', 'open', 'bundles', 'solid', 'inch', 'what', 'brushes', 'product', 'blush', 'american girl', 'you can', 'never opened', 'hard', 'cover', 'fur', 'wore', 'missing 2', ') .', 'no free shipping', 'cans', 'shipped', 'smoke free home', '5ml', 'ugg', '2018', 'prices', 'old', 'lego', 'pic', 'forever', 'shampoo', 'grams', 'days', 'heart', 'box and', 'halloween', 'mug', '5 .', 'sexy', 'amazing', 'its', 'retired', '. brand', 'good condition .', '️', 'accessories', 'stretchy', 'charms', 'kors', 'tea', 'serum', '2x', 'only worn', 'half', '. great', 'boy', 'gently', 'girls', 'batteries', 'originally', 'metal', 'feel', 'work', 'bath', 'made in', 'skinny', 'most', 'beats', 'xbox', 'hot', 'series', 'original box', 'am', 'mask', 'shoulder', 'together', '. they', "men '", 'got', 'ps4', 'not included', 'used for', 'packaging', 'bluetooth', 'minor', 'i can', 'times .', '. has', 'charm', 'mac', 'air', 'heavy', '00', 'coat', 'pics', 'unused', 'plated', 'only used', 'rm ]', 'in great condition', 'will ship', '. free', 'base', 'campus', 'classic', 'of 2', 'free people', 'bundle for', 'in excellent', 'envelope', 'different', 'one size', 'tall', 'missing i', '. worn', 'beauty', "' m", 'logo', 'all in', 'has been', 'bubble', 'including', 'rips', 'will bundle', 'closet', 'sure', '. please', 'been used', 'tags .', 'wedding', 'long sleeve', 'at the', 'new condition', 'right', 'clear', 'to save', 'travel size', 'brand new never', 'been worn', 'down', 'so i', 'twice', 'authenticity', 'very good', 'tieks', 'holes', 'as shown', 'figures', 'factory', '25', 'canister', 'huge', 'photo', 'missing bundle', 'small .', 'buckle', 'hardware', 'include', 'naked', '6 .', 'holds', 'samsung', 'still in', 'size 7', 'no flaws', 'matte', 'best', 'looks', 'wireless', 'cheap', 'separate', '. only', 'comfy', 'd', 'glass', 'bottoms', 'mario', 'parts', 'trades', 'willing', '* * *', 'short', 'lens', 'figure', 'oil', 'dolls', 'vuitton', 'which', 'first', '. very', 'the price', 'high quality', 'closure', 'three', 'but i', 'bracelets', 'month', 'human hair', 'shades', 'new never used', 'scott', 'discontinued', 'faced', 'a little', 'shown in', 'belt', 'sticker', 'forever 21', 's size', 'lipsticks', 'show', 'all new', 'thanks for', 'measures', '] for', 'expires', '. bundle', 'looking', 'abh', 'brandy', 'display', 'marks', 'scent', 'gently used', 'backpack', 'code', 'matching', 'natural', 'the box', 'out my', 'paper', 'other items', 'faux', ': 1', 'in good condition', 'primer', 'fabric', 'are in', '8 .', 'nude', 'no scratches', 'cable', 'dress .', ', the', 'nail', 'spandex', 'us', '4 .', 'all are', 'vs pink', 'binder', 'hat', 'outfits', 'to the', 'negotiable', 'rise', 'free shipping !', 'line', 'strips', 'fl', 'swatched', 'younique', 'retails [', 'summer', 'wear .', 'ds', 'double', 'manual', 'today', '18k', 'missing lularoe', 'deals', 'machine', 'and it', 'on shipping', 'damage', 'lotion', 'rae', 'hours', 'of wear', 'size 2', 'scentsy', 'many', 'polyester', 'another', 'nyx', 'cut', 'pandora', 'take', 'up to', 'sleeves', 'fine', 'to be', 'mascara', 'guitar', 'deluxe', 'next', 'priority', 'is the', 'key', 'life', 'brand new !', 'star', 'yurman', '! i', 'we', 'dvd', 'jersey', 'order', 'shopping', 'smells', 'wood', '. perfect', 'to ship', 'nylon', 'do not', 'brand :', 'for sale', 'stretch', 'size 10', 'and is', 'missing all', 'dark', 'floral', 'size 5', 'chase', 'wide', 'includes :', 'much', '. 00', 'human', 'listed', 'spray', 'firm .', 'pouch', 'reversible', 'black and', '. price is', 'in excellent condition', 'michael', 'people', 'over [', 'to sell', '. this is', 'excellent condition .', 'feet', 'wheels', '10 %', 'palettes', 'for more', 'designer', '40', 'corset', 'powder', 'from a', 'awesome', 'glow', 'kitchen', 'read', 'try', '1 )', '. brand new', 'suit', 'working', 'bape', 'power', 'bling', 'decal', 'true', 'fl oz', '=', 'empty', 'verizon', 'size xs', 'the back', 'use it', 'push', 'you are', 'cord', 'as well', 'deal', 'could', '22', '22k', 'brands', '3 -', 'will not', '90', 'carrier', 'day shipping', 'here', 'keep', 'minnie', 'missing 1', 'to save on', 'and i', 'dress ,', 'winter', 'barbie', 'htf', 'number', 'bye', 'comment', 'remote', 'warmer', 'condition :', 'australia', 'bombshell', 'lightweight', '5 "', '16gb', 'bow', 'clothes', 'stamp', 'candy', 'fitbit', 'i do', 'missing the', 'never used .', 'padding', 'pattern', 'rings', 'retails for', '. if', 'with tracking', 'canvas', 'ultimate', 'neck', 'please check', 'date', 'fashion', 'protectors', 'replacement', 'suede', 'ua', '. just', 'never been used', 'bnwt', 'cloth', '. a', '2016', 'turquoise', 'you will', 'after', 'lauren', 'blanket', 'crystal', 'h', 'queen', 'konami', 'mesh', 'money', 'console', 'couple', 'safe', 'all size', 'size 9', 'new , never', 'season', 'sized', 'year', '. [', 'message me', 'free home .', 'dog', 'pad', 'scratch', 'one of', 'multiple', 'sweatshirt', '. 2', '. smoke', 'on hold', 'heat', 'signed', 'silpada', '. 99', 'colored', 'size is', 'tank top', 'med', 'peach', 'storage', 'nose', 'body works', 'firm please', 'missing great', 'blu', 'compatible', 'n', 'seat', 'thick', '34', '3x', 'bowl', 'into', 'jewelers', 'priced', 'sony', 'all brand', 'with tags .', 'coin', 'easy', 'led', 'free shipping .', 'treatment', 'new sealed', 'of a', 'used ,', 'a new', 'removed', 'laptop', 'maybe', 'scuffs', '2 "', 'discussed', 'keychain', 'let', 'play', 'stain', 'target', '5 -', 'played with', '2017', 'adorable', 'pin', 'toy', 'tula', 'in original', 'size 4', '32', 'cup', 'paisley', 'way', 'and save', 'factory sealed', 'once .', 'price :', 'even', 'beads', 'ready', 'price is firm', 'burberry', 'you have', 'note', 'ultra', 'warm', 'items .', 'msrp', 'socks', 'stack', 'system', '14k gold', 'it has', 'came', 'dustbag', '/ 17', 'boys', 'happy', 'included .', 'like a', 'asking', 'have the', 'crossbody', 'brand new and', 'tears', 'not sure', ". it '", 'mugs', 'urban', 'i ship', 'four', 'k', 'lipsense', 'striped', 'sugar', 'with tag', 'works brand', 'yellow gold', "you '", 'eagle', 'gb', '. these', 'wash', 'this is the', 'denim', 'polishes', 'clasp', 'final', 'our', 'removable', 'cleaning', 'hollister', 'stones', ', 1', 'pet free', 'bars', 'cracked', 'victoria secret', '. *', 'wrap', '- [', 'supply', 'size xl', 'bronzer', 'mk', 'teal', 'bundle with', 'of 3', 'boutique', 'discoloration', 'online', 'tax', 'type', 'vguc', '✅', 'tiffany', '10 .', 'barely', 'cords', 'eyeliner', 'years', '! this', '. smoke free', 'missing this is', '?', 'fun', 'limbs', 'harley', 'slip', 'bag ,', 'disc', 'exp', 'with everything', 'bralette', 'lights', 'mixing', 'mobile', '✨', 'must go', 'care', 'nib', 'robe', '36', 'combo', 'pillow', 'features', 'better', 'crossfit', 'discounts', 'supreme', '8oz', 'broken', 'cosmetics', 'freddy', 'maxi', ', 2', 'at all', '200', '3oz', '500', 'holder', 'night', 'otherwise', 'separately', 'value', 'washable', 'chocolate', 'discount', '! size', '. plus', 'for me', 'anastasia', 'car', 'hole', 'deep', 'virgin', 'shows', 'steel', '. one', 'set includes', 'paperback', 'website', 'i am', 'blouse', 'fading', 'instructions', 'nars', 'plates', 'plays', 'put', 'i need', 'pair of', 'bright', 'owner', 'pjs', 'torrid', 'tour', 'were', 'ipsy', 'missing never', "victoria '", 'wall', 'boot', 'eau', 'always', 'gel', 'mat', 'six', 'someone', 'weight', 'original price', 'add', 'dry', 'hilfiger', 'house', 'imei', 'laces', 'loved', 'orders', 'no stains', 'xbox one', 'bikini', 'extensions', 'justice', 'scrub', 'conditions', '" x', 'special', ', so', 'reasonable', 'acrylic', 'booster', 'packets', 'pallet', 'sheer', 'stainless', 'a great', 'all items', '60', 'irma', 'romper', 'has some', 'gymboree', 'maternity', 'provocateur', 'yeti', ', or', 'there are', 'boden', 'details', 'gildan', 'sprint', 'wifi', 'womens', 'charger ,', 'for 1', 'is not', 'zip up', 'ferragamo', 'iridescent', 'miss', 'now', 'vaulted', '. 25', 'does not', 'on it', 'rubber', 'says', 'firm !', 'funko pop', 'of them', 'dresses', 'genuine', 'names', 'profile', ': [', 'bought for', 'missing handmade', 'ages', 'coupons', 'gone', 'lined', 'seen', 'wax', '2 -', 'defects', 'fragrance', 'fugitive', 'man', 'synthetic', 'tf', 'and has', '1tb', '. in', 'paid [', 'me .', 'retail [', 'headband', 'normal', 'for it', 'description', 'cases', 'pads', '. both', 'color is', 'other listings', 'save on', 'works great', 'drawstring', 'graded', 'mod', 'mouse', 'shower', 'unicorns', 'vera', '. retails', '/ 2', 'in perfect', 'old navy', 'fees', 'machines', 'pages', 'scented', '- 6', '3 )', 'both size', ". i '", 'pigment', 'u', 'check out', 'first class', 'you get', 'perfectly', 'but it', 'outside', 'star wars', 'de', 'while', '- 3', '. it is', 'bangle', 'frames', 'stunning', 'of my', 'bundle and save', 'bombs', 'formal', 'xxl', 'a small', 'missing for', 'ready to', '2oz', 'athletic', 'cartridges', 'cleaned', 'coins', 'cubes', 'diaper', 'hdmi', 'height', 'insurance', 'monitor', 'oversized', 'pen', 'seeds', '/ 16', 'dollars', 'ice', 'lowest', 'proof', 'signature', 'ties', 'tower', '. 1', 'mint condition', 'missing black', 'set .', '3 / 4', 'case .', 'retail price', 'business', 'heel', 'llr', 'leather .', 'begonia', 'candles', 'disk', 'major', 'platter', 'psa', 'trial', 'ulta', '- small', 'includes 2', 'anti', 'combine', 'containers', 'frame', 'loot', 'mumu', 'ounce', 'stila', 'trim', '( 2', '. -', 'on .', 'for [ rm', 'contour', 'lancome', 'meet', 'seasons', 'sleepers', 'slim', 'only .', 'collectors', '. i have', 'tube', 'belly', 'melville', '. super', 'no holes', 'lock', 'must', 'rue', 'and white', 'size 0', 'pet free home', 'buyer', 'lush', 'onesie', 'stitch', 'if you have', 'charging', 'grade', 'mic', 'serious', 'squash', 'and 1', 'bundle to', 'each .', 'duffle', 'icloud', 'lanyard', 'onyx', 'pops', 'salon', 'similar', 'violette', 'for your', 'size .', 'top .', 'cuff', 'liquid', 'og', 'sheet', 'kylie cosmetics', '100 % authentic', 'bundled', 'pendant', 'pre', 'price .', 'with charger', 'vinyl', '®', 'hold', 'glossy', '. you', '. includes', 'very good condition', 'beige', 'carat', 'crew', 'lemons', 'letter', 'pokémon', '* bundle', 'missing small', 'buttons', 'contains', 'ex', 'i7', 'individually', 'kay', 'mattel', 'mostly', 'opi', 'pump', 'sensors', 'sequins', 'tan', 'who', '( 1', 'as is', 'hold for', 'size 3', 'dragon', 'extremely', 'luggage', 'light up', 'light weight', 'missing 3', 'no longer', 'so cute', 'knee', 'longer', 'mitchell', 'authentic .', 'mix', 'll', 'polish', 'head', '38', 'bundling', 'engagement', 'through', 'and one', 'new with tag', 'avon', 'charge', 'inseam', 'ks', 'louboutin', 'pave', 'playstation', 'twilly', '. like', '. made', 'top ,', 'except', 'lining', 'noticeable', 'products', 'shopkins', 'missing 100', 'very sexy', 'size 8 .', '28', 'absolutely', 'bleached', 'flour', 'generic', 'jamberry', 'mailers', 'model', 'mori', 'packet', 'their', 'xxs', 'yugioh', '~', '. (', '. box', '. some', 'are size', 'great price', 'this bundle', 'am selling', 'it was', 'under armour', '6s', 'baseball', 'lots', 'spell', 'warranty', 'wraps', '. and', 'box ,', 'dress up', '. great condition', 'with tags !', 'layered', 'silk', 'week', 'feel free', '38dd', 'bnib', 'drawers', 'glue', 'marc', 'north', 'older', 'others', 'ozs', 'receive', 'tubes', '7 .', '] value', '. [ rm', '500gb', 'condoms', 'distressed', 'elastic', 'give', 'how', 'lorac', 'pearl', 'regimen', 'stitched', 'wallets', 'zella', '& t', 'find .', 'new authentic', '32gb', 'beach', 'cc', 'glitter', 'runs', 'may have', 'not buy', 'the other', 'you want', 'alloy', 'zuni', 'cat', 'per', 'doterra', 'in my', 'fringe', 'joggers', 'nothing', 'price is for', 'carter', 'hidrocor', 'measurements', '9 .', 'all my', 'missing .', 'missing beautiful', '38ddd', 'added', 'ball', 'count', 'detachable', 'ftp', 'kickee', 'kind', 'mcdonald', 'princess', 'rf', 'spf', 'teapot', 'vantel', '- free', 'bag and', 'same day', 'worn a', 'expiration', 'filled', 'kane', 'stores', 'wii', 'wool', '2 pairs', 'case for', 'my closet', 'please ask', 'which is', 'without tags', '. i will', 'essential', 'maybelline', 'usb', 'purchasing', '- new', 'stars', 'wars', 'bundle includes', 'oz .', 'manufacturing', '. 3', 'card .', 'board', 'clinique', 'jar', 'kate', 'legging', 'mind', 'test', 'free to', 'price !', 'with it', '. 5 "', 'bebe', 'comics', 'crafts', 'diapers', 'fitted', 'gta', 'page', 'rebecca', 'video', 'within', 'and have', 'fair condition', 'light blue', 'not lularoe', 'still have', 'size small .', 'automatic', 'butter', 'framing', 'poly', 'ratings', 'returns', 'roses', 'scarf', 'shakeology', 'weekender', 'and get', 'with any', 'etc', 'randy', 'resistant', 'rue21', 'scratches on', 'size m', 'total of', '. free shipping', ', in', 'birthday', 'lunch', 'revival', 'tiny', 'gps', 'part', 'worth', '❤', 'earrings .', 'have been', 'one pair', '10kt', '150', '17', '2ml', 'affliction', 'beast', 'blonde', 'celine', 'change', 'costume', 'maroon', 'mists', 'separating', 'tanks', 'tshirt', 'tshirts', 'version', 'but still', 'can bundle', 'missing 6', 'no holds', 'worn and', '19', '34dd', 'attached', 'displayed', 'mcm', 'pigmented', 'platform', 'purses', 'sales', 'seatbelt', 'simple', 'split', 'taupe', 'vases', 'wildfox', 'workout', '/ 10', 'case and', "don '", 'missing -', 'purse .', 't come', 'want to', 'worn twice', 'con', 'cz', 'lps', 'nm', 'since', 'tunic', ', 4', 's a', 'selling as', 'shirt .', '. these are', 'ct', 'owned', 'mansion', 'stock', 'for 10', 'checkout', 'gym', 'obo', 'speaker', 'unless', ', 3', '50 %', 's .', 'new ! !', '06', 'burgundy', 'comments', 'contact', 'duffel', 'edp', 'grace', 'imitation', 'insoles', 'laser', 'longchamp', 'pitcher', 'pullover', 'unstoppables', 'unsure', 'world', '% full', 'missing *', 'selling for', 'with other', 'conair', 'dd', 'king', 'nfl', 'plaid', 'rated', 'restorative', 'sanitized', 'stick', 'teddy', 'views', 'money back', 'one is', 'out other', 'pic 4', 'questions feel', 'and pet free', '5y', 'bit', 'candle', 'crystals', 'flowers', 'ink', 'pencil', 'zara', 'all 3', 'for my', 'the description', 'new without tags', 'tokyo', 'all the', 'works with', 'fresh', 'there is', 'comforter', 'leopard', 'monogram', 'i don', 'is [', '2t', 'buddy', 'era', 'foot', 'tape', '& body', '. good', 'on a', 'missing new with', 'bins', 'bronze', 'condo', 'jars', 'petite', 'renaissance', 'rhinestones', 'strappy', 'tory', 'variety', ', kylie', 'brand .', 'for exposure', 'pet and', 'priced to', 'see all', 'shoes are', 'with gold', 'with one', 'xbox 360', 's secret pink', 'additional', 'canisters', 'capsules', 'collector', 'else', 'flaw', 'honey', 'ladies', 'moisture', 'reflected', 'roshe', 'sdcc', 'tees', 'wi', 'x2', '! no', 'a used', 'also includes', 'available in', 'shipping ,', 'this listing', 'any questions please', '84', 'altered', 'cookware', 'vr', '* no', 'and 2', 'bag !', 'worn for', 'to find', 'used a', 'to ask', '] !', 'envelopes', 'strapless', 'black leather', ', never used', 'acacia', 'auto', 'bowls', 'container', 'digital', 'phillip', 'n wild', 'only [', 'shape .', 'used but', '750', 'abercrombie', 'basic', 'cabochon', 'decay', 'ends', 'eyebrow', 'jogger', 'lavender', 'lowball', 'opal', 'payless', 'played', 'wreath', '- one', 'ask .', 'for christmas', 'secret new', 'some are', 'tried on', 'very hard', 'like new .', '4oz', 'alterations', 'biggest', 'booklet', 'brought', 'cib', 'clock', 'disneyland', 'drone', 'keyboard', 'mermaid', 'mikoh', 'release', 'return', 'solo', 'tons', 'zebra', '. also', 'color .', 'in color', '• no', 'aluminum', 'flavored', 'hp', 'lillebaby', 'liner', 'lol', 'models', 'single', 'train', '. thanks', 'chain .', 'with all', 'working condition', 'bloomingdale', 'name', ', only', '. there', 'cake', 'guaranteed', 'pure', 'dress with', 'long lasting', 'this bag', 'buds', 'cpu', 'finish', 'gameboy', 'gunmetal', 'mega', 'signs', 'taken', '! *', '. not', 'be used', '64gb', '9311', 'apt', 'forever21', 'intact', 'kleancolor', 'lamp', 'list', '. retail', '. •', 'firm price', 'iphone 4', 'lularoe tc', 'missing one', 'must have', 'ships in', 'in box .', '128gb', 'ads', 'agate', 'alex', 'brazilian', 'circulated', 'clip', 'cracks', 'crayon', 'dead', 'goddess', 'insert', 'lootcrate', 'lowballs', 'melissa', 'perc', 'samon', 'similac', 'sprays', 'titanium', 'trays', 'vhtf', 'washing', 'waterproof', 'zales', 'air jordan', 'are a', 'back of', 'better than', 'mary kay', 'no low', 'no trades', 'of it', 'to get', 'to make', '❤ ️', 'anything', 'asap', 'bomb', 'carrying', 'flash', 'holiday', 'inserts', 'medela', 'sarah', 'silicone', 'tried', 'where', ', a', 'great shape', 'in package', 'size 4t', 'herbivore', 'cold', 'spend', 'apply', 'coupon', 'formula', 'kinect', 'sandals', 'ysl', '3 oz', 'in plastic', 'material :', 'will fit', '7 . 5', 'retails for [', '12m', 'child', 'chirp', 'handbag', 'hook', 'lenses', 'lilly', 'lingerie', 'master', 'percent', 're', 'regular', 'sequin', 'snap', 'tj', 'vanilla', 'won', 'wrong', '; )', 'good for', 'i bundle', 'you stickers', 'this is for', '42ct', 'blind', 'bnwot', 'ce4', 'certificate', 'chopsticks', 'clings', 'destroyed', 'gelcolor', 'gels', 'goodwill', 'pant', 'slightly', 'snapchat', 'spring', 'thermal', 'unboxed', 'usps', 'xhilaration', 'yards', 'zelda', '! free', ', two', '. top', 'add on', 'black .', 'full .', 'i bought', 'missing 5', 'missing reserved', 'top and', 'bons', 'flea', 'megazord', 'otterbox', 'pajama', 'papers', 'sells', ', black', '2 pair', 'not come', 'of life', 'shower gel', 'please check out', 'squishy', 'bundle and', 'thank you', 'used it', '. if you', 'pink brand', 'non', 'japan', 'brand is', 'too faced', 'bandage', 'commenting', 'dogs', 'korean', 'spots', 'hard to', 'large .', 'next day', ', never worn', 'out my other', 'boyfriend', 'celebrities', 'elite', 'farmhouse', 'given', 'heads', 'michele', 'mimi', 'sperrys', 'stand', 'tpu', 'collection .', 'mark on', 'new -', 'price includes', 'watch .', '100ct', '64', 'alarm', 'arbonne', 'aritzia', 'bust', 'centerpieces', 'conditioners', 'donated', 'dreaming', 'facial', 'healthtex', 'lead', 'lid', 'medusa', 'mj', 'parfum', 'pest', 'rompers', 'scrubs', 'second', 'softcover', 'taking', 'weighs', '|', '- price', '/ 12', '2 piece', 'a gift', 'used twice', 'my other items', 'converse', 'fossil', 'mcfarlane', 'mickey', 'music', 'rechargeable', '☆', '. still', '12 /', '5 oz', '7 /', 'and never', 'charlotte russe', 'combined shipping', 'for looking', 'of two', 'plus tax', 'ship out', 'shoes size', 'them .', '. thanks for', 'f', 'lab', 'ship !', 'use for', 'own', 'rhodium', '. bought', 'cherokee', 'place', 'post', 'sports', '5s', 'boohoo', 'carly', 'shining', 'thin', 'diameter', 'dyed', 'evolve', 'foam', 'manduka', 'nails', 'olive', 'pacific', 'paint', 'released', 'say', 'sweatsuit', 'testers', 'toe', 'whipped', '- 12', '. brown', 'all over', 'bath bomb', 'bottle of', 'get 1', 'missing used', 'more than', 'tags -', '05', '1x', '38d', 'barnes', 'bobbi', 'broke', 'chap', 'choker', 'coa', 'crib', 'cvs', 'dojo', 'donate', 'donating', 'flower', 'goose', 'jean', 'prevent', 'protection', 'rd', 'retailed', 'sakura', 'salvage', 'screwdriver', 'seperate', 'sight', 'silky', 'simpsons', 'slight', 'theora', 'think', 'tvs', 'vnds', '- white', '24 "', '3 /', '6 months', '] or', 'available :', 'free and', 'let me', 'listing is', 'medium .', 'more pictures', 'new &', 'new unused', 'os leggings', 'slow rising', 'sports bra', 'the picture', 'new . never', 'breath', 'buying', 'charizard', 'doesn', 'drop', 'highlighter', 'jungle', 'links', 'mixed', 'necklaces', 'poles', 'random', 'suspension', 'add to', 'appearance of', 'james avery', 'limited edition', 'one time', 'rose gold', 'this shirt', 'heels', 'moisturizer', 'several', 'support', '×', 'and black', 'swiss', 'toner', '’', 'curl', 'deck', 'oval', 'topic', 'for one', 'in this', 'included in', 'missing comes', '42', 'caps', 'detail', 'ear', 'however', 'keen', 'kpop', 'largest', 'milanese', 'platinum', 'quilt', 'ray', 'though', '2 for', 'apple brand', 'on all', 'shipping included', 'with case', 'x 5', 'ags', 'cancelled', 'durable', 'ebay', 'fenton', 'flavor', 'gobble', 'guc', 'homemade', 'hood', 'hoodies', 'moisturizing', 'revolve', 'rooted', 'shimmer', 'stencil', 'storm', 'topaz', 'tubs', 'vspink_forsale', 'wave', 'wick', 'young', 'free .', 'full sized', 'great with', 'hot pink', 'missing includes', 'missing only', 'ship .', 'shoulder bag', 'size 18', 'still works', 'will include', '14kt', '3ds', '^', 'actually', 'co', 'concealer', 'damaged', 'exclusives', 'gain', 'heater', 'https', 'lisse', 'marled', 'okay', 'pipe', 'private', 'sad', 'thankful', 'v', 'before buying', 'new free', 'nike new', 'as a gift', 'cap', 'clarisonic', 'tarnish', ', it', 'easy to', 'has the', 'no rips', 'that i', 'flap', 'handles', 'a smoke', '34a', 'application', 'cartridge', 'smooth', 'body wash', 'that is', 'view all', 'children', 'coral', 'garland', 'ninjas', 'patent', 'shadow', ', just', ', please', '. everything', 'a -', 'made of', 'missing no', 'you like', '1ml', '36dd', '80', 'alexis', 'balls', 'balm', 'benefit', 'checks', 'cheer', 'clearance', 'cm', 'colourpop', 'cropped', 'ctw', 'dream', 'due', 'dupe', 'dōterra', 'hanging', 'herbalife', 'idea', 'marvel', 'microphone', 'napkins', 'nova', 'orig', 'paperwork', 'posh', 'ps3', 'puma', 'rest', 'row', 'shep', 'shopkin', 'sweet', 'unknown', 'vanity', 'vision', 'wristlet', 'york', '♡', '- not', '. but', 'all for', 'for 3', 'in size', 'or holes', 'shipping and', 'silver .', '1pcs', '39', 'alone', 'babies', 'bandeau', 'bed', 'currently', 'enjoy', 'globe', 'horror', 'iron', 'keychains', 'klein', 'la', 'madden', 'movies', 'neverfull', 'ones', 'receiving', 'satchel', 'textbook', 'waterford', 'whole', 'x13', 'xsmall', 'yoga', 'a lot', 'a month', 'other listing', 'sale !', 'the tags', 'missing new in', 'angel', 'bts', 'cardigan', 'monster', 'pay', 'iphone 7', 'than one', 'never worn .', '4 -', 'cleanser', 'plate', 'stylus', 'hello', 'making', 'tone', ', shipping', 'my lowest', 'of these', 'great condition !', 'help', 'hidden', 'island', 'probably', 'sparkly', 'taxes', 'tear', 'then', '. most', '13 .', 'a [', 'a good', 'charger .', 'listings .', 'questions you', 'the inside', 'to a', 'waist .', '1500', '4x6', 'bigger', 'block', 'brick', 'cacique', 'control', 'gg', 'handy', 'hanes', 'hardcover', 'holo', 'i5', 'ipod', 'italy', 'jadore', 'kt', 'lightly', 'living', 'lounge', 'miniature', 'mx', 'notes', 'overalls', 'plush', 'previously', 'rulu', 'sided', 'smashbox', 'stripes', 'tahitian', 'tsums', 'vans', 'vial', 'web', 'yield', '. sz', 'eye shadow', 'items are', 'kors michael', 'originally [', 'sale is', 'smoke /', 'to try', '♡ ♡', '250', '80s', 'accessory', 'biker', 'breathtaking', 'calvin', 'christian', 'diffuser', 'fair', 'his', 'julia', 'kitchenaid', 'manga', 'mm', 'murad', 'namebrand', 'nasal', 'oils', 'pills', 'pls', 'professional', 'project', 'ram', 'serial', 'smile', 'standard', 'straight', 'sweaters', 'tartiest', 'tulle', ', great', '4 )', ': -', 'bikini top', 'box with', 'few times', 'if it', 'a few times', 'rm ] .', 'bulbs', 'headphones', 'panties', 'shell', 'why', ', very', '3 for', 'picture .', 'good condition ,', 'frida', 'loose', 'tie', '( 3', 'like to', 'sound', 'until', 'velcro', 'an offer', 'my other listings', 'animal', 'cleansing', 'franchise', 'hippie', 'jingle', 'lm', 'luxury', 'phillips', 'pole', 'sponge', 'spot', 'titles', ': (', 'a couple', 'body lotion', 'bundled with', 'half zip', 'made by', 'nike nike', 'missing like new', '12s', '2014', 'accordingly', 'airbrush', 'autograph', 'basketball', 'bends', 'bf', 'cart', 'cereal', 'cheaper', 'closeout', 'db', 'edt', 'eyewear', 'faded', 'foundations', 'functional', 'gallon', 'halfway', 'hi', 'horn', 'journals', 'panda', 'perfumes', 'pouchy83', 'rarely', 'riding', 'run', 'sought', 'tease', 'tree', 'volume', 'zagg', '- medium', '12 months', 'a set', 'can use', 'foundation in', 'it comes', 'leggings .', 'missing [', 'not include', 'of 6', 'or [', 'perfect .', 'size 3t', 'w /', 'with original', '125', 'agreed', 'amiibo', 'avia', 'background', 'becca', 'bruce', 'cleansers', 'collectible', 'compliments', 'crop', 'fee', 'flexible', 'hardly', 'inspired', 'interior', 'maidenform', 'might', 'min', 'ninascloset', 'prom', 'pusher', 'rabbit', 'refurbished', 'snags', 'splitting', '! never', '- -', 'for them', 'one .', 'reasonable offers', 'shirt ,', 'stretch .', 'the screen', 'used once', 'prada', 'rc', 'unique', 'usa', '( 4', 'because i', 'for all', 'new without', 'questions please', 'perfect condition .', 'tool', 'viewing', 'vive', '- 1', 'iphone 6', '300', 'advance', 'artis', 'enamel', 'every', 'eyes', 'going', 'nikon', 'official', 'origins', 'satin', 'sorry', 'valentino', 'back to', 'missing a', 'to [', '9k', 'artist', 'barley', 'blackmilk', 'brushed', 'cheetah', 'chi', 'defect', 'doses', 'fleece', 'george', 'guns', 'hundred', 'interested', 'kkw', 'laced', 'liter', 'mens', 'pedal', 'percolator', 'pilling', 'pinkie', 'preworn', 'primitive', 'quart', 'quilted', 'sapphire', 'seraphine', 'snaps', 'sole', 'swarovski', 'tmobile', 'toys', 'twin', 'unofficial', 'veuc', '！', '1 for', '3 "', '7 "', '] ,', 'a bit', 'all black', 'black background', 'came with', 'love this', 'missing these', 'phone is', 'ring .', 'see photos', 'shipping is', 't -', 'the bottom', 'the plastic', 'with 2', '6 . 5', 'all brand new', '1pc', '31', '37', '7oz', 'approximately', 'arms', 'assorted', 'bleach', 'chronograph', 'convention', 'cowl', 'f21', 'fe', 'gamestop', 'infinite', 'itworks', 'jegging', 'jumbo', 'jumpers', 'keywords', 'le', 'minkpink', 'movie', 'osito', 'perfectlyposh', 'petal', 'port', 'pristine', 'pulitzer', 'rainbow', 'reduced', 'sherpa', 'sleepy', 'things', 'touched', 'tumbler', 'unif', 'versace', 'waistband', '4 /', 'i want', 'items and', 'lightly used', 'love pink', 'lularoe brand', 'missing good', 'packaging !', 'small -', 'the package', 'to any', 'usually ship', 'comes with a', 'missing great condition', "secret victoria '", 'used condition .', 'changing', 'harvey', 'panty', 'party', 'patina', 'speck', 'vita', '. 0', '/ 4', 'can be worn', 'is in good', 'didn', 'goes', 'lipgloss', 'prana', 'you !', 'settings', 'is brand', 'missing please', '8 . 5', "i ' ll", 'above', 'bumgenius', 'cables', 'clothing', 'dye', 'fall', 'minifigure', 'mirror', 'parka', 'rollerball', '- 10', '] off', 'and bundle', 'got it', 'includes shipping', 'missing bnwt', 'some wear', 'the color', 'very cute', '( 2 )', 'from a smoke', '350', '35mm', '70', '7ml', '999', 'android', 'athleta', 'bdsm', 'bone', 'boston', 'boundaries', 'clearblue', 'concepts', 'dc', 'decor', 'decorative', 'deodorants', 'detangling', 'dishwasher', 'downloaded', 'dsi', 'duct', 'dvds', 'eeuc', 'eyelashes', 'frank', 'g', 'geneva', 'herringbone', 'kikki', 'mediums', 'nightgown', 'obsession', 'oyster', 'popclick', 'programmable', 'ps2', 'reed', 'remotes', 'roughly', 'scrapbook', 'shoedazzle', 'snow', 'sparkles', 'sperry', 'suits', 'toddler', 'tri', 'wunder', '°', '% cotton', '8 /', 'a .', 'charger and', 'closet for', 'for two', 'great to', 'i just', 'is my', 'missing 4', 'on front', 'on me', 'pink new', 'ship same', 'socks .', 'the shirt', 'wear ,', 'x 4', '1 / 2', 'no free ship', 'alive', 'blues', 'brooch', 'castle', 'cato', 'concentrate', 'continental', 'craft', 'cross', 'decided', 'ella', 'emperor', 'flowerbomb', 'football', 'jon', 'jujube', 'lite', 'locked', 'pumpkin', 'rave', 'revlon', 'rush', 'shaped', 'shop', 'tarnished', 'triple', '. 8', '6 -', '7 oz', 'all of', 'back .', 'clean ,', 'everything in', 'for 2', 'home decor', 'life left', 'lip gloss', 'missing women', 'offer .', 'once for', 'over the', 'rae dunn', 'soft and', 't have', 'to bottom', 'check out my', 'me know if', '30oz', 'braun', 'comb', 'embroidered', 'primark', 'sleeveless', 'swim', '. thank', 'mint green', 'missing •', 'more !', 'pants .', 'pink and', 'the front', 'to you', 'creamiicandy', 'bundle to save', 'are brand', 'lips', 'mary', 'outfitters', '% off', '- 7', 'have some', '14g', 'approx', 'guard', 'gymshark', 'kits', 'lg', 'locket', 'revo', 'sign', 'unworn', ', still', 'american eagle', 'bubble mailers', 'in packaging', 'pink pink', '180', '1990', '72', 'aio', 'armani', 'art', 'bake', 'bnew', 'bryant', 'copic', 'dent', 'episodes', 'glittery', 'guardian', 'london', 'lookalike', 'medicine', 'nobo', 'pallets', 'paracord', 'qty', 'reign', 'rejuvenator', 'sent', 'sequined', 'smitten', 'space', 'ssd', 'statement', 'teen', 'tint', 'trunk', 'turntable', '* free', '] +', 'a pair', 'and shipping', 'bag for', 'be a', 'gold plated', 'missing cute', 'spandex .', 'the pictures', 'up for', 'wear on', 'with black', 'in the box', '125ml', '3d', 'beaters', 'bench', 'cb', 'cinnamon', 'crafting', 'cranberry', 'dimensions', 'donruss', 'fireball', 'fr', 'garter', 'headbands', 'henry', 'lang', 'legos', 'lotions', 'mamaroo', 'mophie', 'notrades', 'pillowcase', 'props', 'rams', 'saber', 'seems', 'songs', 'sorel', 'string', 'talking', 'timex', 'toaster', 'ton', 'trina', 'types', 'width', 'wonderful', 'wrought', 'yeezy', ', you', '. black', '. used', 'each additional', 'for free', 'for you', 'in one', 'iphone 5', 'lace up', 'low price', 'missing in', 'palette ,', 'to size', 'wear it', 'shipping ! !', 'bands', 'ever', 'ii', 'leg', 'person', 'ranger', 'something', 'tripod', 'webkinz', '2 )', 'weight :', '2 . 5', 'tv', 'x20', 'that it', 'boost', 'tilbury', 'went', '. from', 'comes from', '. new with', 'along', 'compartments', 'heartgold', 'hope', 'inglot', 'jade', 'james', 'marble', 'piling', 'sweats', 'topps', 'tortoise', 'yum', '- bundle', 'dunn brand', 'fit .', 'just ask', 'missing vintage', 'new price', 'works perfectly', 'ask any questions', 'excellent condition ,', 'aaa', 'bookmarks', 'buns', 'common', 'directly', 'dj', 'espresso', 'exact', 'expensive', 'faucet', 'fix', 'fp', 'fully', 'gauranteed', 'gravy', 'hairline', 'hd', 'hottie', 'hourglass', 'knockout', 'kohls', 'megan', 'oxblood', 'pc', 'pillows', 'pixi', 'plucked', 'qt', 'retractable', 'rock', 'rugby', 'spent', 'sticky', 'strings', 'strobing', 'thong', 'turned', 'usable', 'wine', 'yellowed', 'yoshi', '/ smoke', '1 time', 'are new', 'be shipped', 'everything is', 'fast !', 'for any', 'get rid', 'it for', 'jacket .', 'never wore', 'please read', 'plus size', 'size fits', 'tag .', 'top is', '. size small', '1993', '256gb', '2day', '34ddd', 'aeropostale', 'anymore', 'bella', 'claire', 'curlers', 'delectable', 'delta', 'dillon', 'earth', 'electro', 'flat', 'ghz', 'irmas', 'lost', 'markers', 'mutant', 'nebraska', 'neca', 'nuby', 'pounds', 'reel', 'reflect', 'starring', 'step', 'sunday', 'sunglasses', 'sweatshirts', 'trilogy', 'unlock', 'vasanti', 'victorian', 'vitamin', 'wefts', "' d", "5 '", '@ [', 'also have', 'and 3', 'apple watch', 'final price', 'for this', 'lularoe none', 'missing blue', 'nine west', 'package .', 'push up', 'the children', 'to let', 'wash wear', 'will come', '. feel free', '. they are', 'new in package', '35', 'adapter', 'applicator', 'either', 'incredible', 'link', 'piercing', 'straw', 'touch', 'vegan', 'vigoss', 'a free', 'cute with', 'does have', 'not used', 'oz each', 'surgical steel', 'uv', 'listing for', 'trx', 've', 'bradley', 'spade', 'family', 'hats', 'skyn', 'studs', 'thrive', 'towel', ', new', '1 pair', '20 %', 'purchased from', 'questions or', 'was a', 'will sell', '. never used', '31st', '6fl', 'alike', 'authentication', 'bandanas', 'bathing', 'blackhead', 'circo', 'class', 'crocs', 'designed', 'equestrian', 'fleo', 'fraying', 'helmet', 'interchangeable', 'jakedasnake', 'knotted', 'label', 'mop', 'peacocks', 'playset', 'polybag', 'postcards', 'processor', 'remember', 'scrunchies', 'snowsuit', 'southern', 'stent', 'talk', 'tartelette', 'tilly', 'toploader', 'umbro', 'updated', 'vsx', 'walgreens', 'warmest', '/ m', 'but the', 'do bundles', 'in its', 'just the', 'missing super', 'on my', 'red and', 'slightly used', 'the original', 'time .', '295', '322', '60w', 'backings', 'bakery', 'batman', 'bead', 'beautifully', 'blouses', 'boxed', 'brass', 'bridal', 'burch', 'cameras', 'capri', 'chamber', 'comparable', 'daily', 'deadstock', 'decorating', 'enfamil', 'figurines', 'findings', 'flings', 'france', 'fx', 'garanimals', 'hotty', 'intel', 'jane', 'joy', 'kimono', 'liquidation', 'lowballing', 'moving', 'named', 'packages', 'pcs', 'pendleton', 'plugs', 'receipts', 'robot', 'saige', 'sanitize', 'shark', 'sprayed', 'tail', 'tattoos', 'ticket', 'tier', 'walmart', 'wolverine', '. so', '] with', 'a bundle', 'and more', 'barely used', 'bottoms are', 'box is', 'bracelet .', 'brown leather', 'cheetah print', 'condition but', 'cute and', 'jacket ,', 'jewelry .', 'liquid lipstick', 'missing authentic', 'missing nwt', 'need to', 'new from', 'please don', 'questions ,', 'ring size', 'secret victoria', 'silver tone', 'smoking home', 't know', 'tee .', 'the game', 'try on', '. no flaws', 'and save on', 'have any questions', '40dd', 'adhesive', 'autumn', 'bulls', 'chunky', 'cleaner', 'collective', 'croft', 'holographic', 'kappa', 'mam', 'massager', 'pricing', 'question', 'sport', 'tarnishing', 'ts', 'unisex', 'wolves', ', blue', '. color', 'bottle ,', 'bought it', 'case ,', 'condition with', 'fits all', 'games ,', 'include :', 'nail polish', 'sold as', 'taken out', 'brand new -', 'hermès', 'partum', ', one', 'get it', 'to see', 'damier', 'league', 'punch', '1 oz', 'are the', '( 1 )', 'below', 'blend', 'crease', 'nmd', 'points', 'seller', "' re", '11 .', 'look at', 'lululemon lululemon', 'with free', 'i have a', '199', '2002', '2b', '_', 'apex', 'aquarium', 'authenic', 'blizzard', 'bodysuit', 'close', 'cyborg', 'fixed', 'gag', 'geo', 'giftcard', 'grail', 'gtx', 'gun', 'handle', 'heathered', 'hookah', 'iphone5', 'italian', 'kds', 'leave', 'limelight', 'livie', 'middleton', 'missguided', 'norwex', 'ocelot', 'octopus', 'pot', 'protect', 'razer', 'room', 'saddle', 'sanctuary', 'scentportables', 'shiseido', 'spacers', 'specs', 'straws', 'uag', 'unauthentic', 'victorias', 'wooden', 'zippers', '. for', 'a perfect', 'built in', 'condition size', 'days .', 'in it', 'in price', 'orange and', 'price [', 'size 11', 'used and', '0s', '1st', '3ml', '585', '6pc', 'barn', 'bm', 'bn', 'boss', 'ccm', 'cleanse', 'clutch', 'cognac', 'credit', 'cupcake', 'dipped', 'fedex', 'g1', 'garnier', 'giant', 'gomez', 'hanger', 'ibloom', 'immediate', 'intensity', 'jackets', 'jeweler', 'kai', 'lambskin', 'manicure', 'medallion', 'mike', 'multiples', 'nvr', 'orchid', 'paring', 'popular', 'ppg', 'prints', 'published', 'puffer', 'repro', 'resized', 'runway', 'saffiano', 'shock', 'sites', 'smell', 'smokers', 'stadium', 'starbucks', 'strip', 'subwoofer', 'surfer', 'sweetie', 'symbol', 'teaches', 'venti', 'watt', 'wedges', 'wizard', 'wrinkled', 'zodiac', '✈', '✔', '( not', '- xl', '. 75', '. high', '8 oz', 'at my', 'by victoria', 'care of', 'cleaning out', 'comes in', 'condition (', 'free pet', 'have it', 'includes a', 'listings and', 'listings for', 'me with', 'no tags', 'only swatched', 'original packaging', 'pictured .', 'stains ,', 'sweater .', 't see', 'tested and', 'to wear', 'great condition ,', 'it is a', 'like new condition', 'on the back', '1990s', 'anthropologie', 'demi', 'egg', 'extras', 'faja', 'instyler', 'kinder', 'molly', 'ounces', 'phones', 'shipments', 'stripper', 'yes', '. gently', '4 oz', '] plus', 'come out', 'is from', 'lipstick in', 'listing includes', 'of times', 'offers .', 'on back', 'oz )', 'please see', 'shorts .', 'size l', 'used only', 'work .', '. only worn', 'can be used', 'on the front', 'pink brand new', 'hoppy', 'i got', 'new never worn', 'worn once .', 'what you', 'for apple', 'legend', 'malaysian', 'microfiber', 'omega', 'rachel', 'setting', 'underwire', '- used', '2 oz', 'piece .', 'shirt size', 'the day', '- [ rm', '25oz', '45', 'anywhere', 'apparel', 'c', 'cast', 'closed', 'cookies', 'crowd', 'david', 'dope', 'eeeuc', 'expenses', 'found', 'gamecube', 'gown', 'hack', 'honda', 'hung', 'jolteon', 'june', 'lightning', 'loungefly', 'mag', 'mayari', 'mercurial', 'merry', 'mg', 'neckline', 'nickel', 'nipple', 'pebble', 'popsugar', 'prescription', 'purchases', 'quick', 'rasberry', 'santa', 'scoopneck', 'sfpf', 'shoelaces', 'skeletor', 'sleep', 'stations', 'took', 'virginity', 'weeks', 'ziploc', '! :', '! comes', '. firm', 'boots .', 'fabric .', 'flip flops', 'it in', 'me know', 'navy blue', 'or next', 'pack of', 'part of', 'size you', 'super soft', 'to all', 'to my', 'wear and', 'you .', 'your own', "' s in", '10ml', '1991', '2m', '7pc', 'anais', 'barrels', 'chavonne11', 'coffee', 'core', 'coverage', 'cuttings', 'decals', 'donkey', 'dorothy', 'dragonball', 'drawing', 'drill', 'embellished', 'eyeglass', 'fairytale', 'fanny', 'fishnet', 'giving', 'hmu', 'jessie', 'kelly', 'knives', 'kohl', 'lebrons', 'lipliner', 'loaf', 'logan', 'lucy', 'mercier', 'mink', 'morganite', 'mtg', 'org', 'pd', 'pellets', 'pores', 'prepared', 'pretend', 'pros', 'recive', 'regardless', 'seal', 'shimmery', 'soles', 'spaghetti', 'stocking', 'succulent', 'table', 'wrapping', 'wrist', 'yves', '●', '❄', '! 1', '# 3', ') )', '. she', '18 months', 'and in', 'bought from', 'christmas gift', 'fits like', 'hardly used', 'keep it', 'kind of', 'low ball', 'of each', 'offers !', 'open back', 'pink with', 'quality .', 'secret size', 'see through', 'silver plated', 't .', 'very soft', 'washed and', 'welcome to', 'worn ,', 'all items are', 'nike brand new', 'anyone', 'baskets', 'chair', 'crossbones', 'dozen', 'laundered', 'mr', 'pageant', 'plain', 'polarized', 'speed', 'themed', 'troopers', 'ud', 'vince', 'xv', 'zippered', '& white', ', white', '. 9', '. fast', '. see', '. with', ': one', 'as pictured', 'before purchasing', 'before you', 'blue and', 'brand -', 'everything you', 'missing girls', 'on your', 'only the', 'open to', 'please view', 'see pictures', 'used this', 'you bundle', 'you for', 'in my closet', 'like new !', 'size medium .', 'away', 'comfort', 'getting', 'micro', 'patagonia', 'pocus', 'rookie', 'favorite', 'x 3', 'colorful', 'decks', 'lauder', 'lf', 'self', 'and comes', 'see my', 'so you', '128', '201', '33', '3gs', 'airwalk', 'becky', 'bell', 'cookie', 'eccc', 'filtration', 'jumpsuit', 'klorane', 'obsessed', 'planner', 'programmed', 'protein', 'puppy', 'ralph', 'ravewithmigente', 'reserve', 'tb', 'zoom', '" tall', ', authentic', ', comes', '. check', '. ✨', '0 -', 'black ,', 'game .', 'light pink', 'light wash', 'missing it', 'phone case', 'reserved for', 'strap .', 'the phone', 'the top', 'tiffany &', '160g', '18months', '360', '4ml', 'analog', 'b', 'bac', 'ban', 'biore', 'birthstone', 'bloom', 'boop', 'bud', 'cardboard', 'chargers', 'chic', 'coloring', 'confetti', 'cove', 'creams', 'deer', 'designs', 'diesel', 'dior', 'donut', 'drive', 'dyson', 'eat', 'elephants', 'elk', 'episode', 'erica', 'everywhere', 'feedback', 'fingers', 'frequent', 'galaxy', 'ge', 'gifts', 'gitd', 'glucose', 'gypsy', 'handcrafted', 'harlow', 'hershey', 'hospital', 'hostess', 'ibiza', 'international', 'jeffery', 'juniors', 'lea', 'licensed', 'lint', 'loom', 'milwaukee', 'mystery', 'napa', 'nb', 'oranges', 'orignal', 'parade', 'pixie', 'pochette', 'poncho', 'raven', 'reciept', 'research', 'responsible', 'ribbed', 'sailor', 'sanitizer', 'sesame', 'sewed', 'shaker', 'shears', 'shift', 'shops', 'shortie', 'snapback', 'soaked', 'spark', 'teeth', 'third', 'thrift', 'tub', 'unprocessed', 'uppers', 'valued', 'votive', 'wavy', 'whisper', 'wipes', 'wirelessly', 'wouldn', 'wrapped', 'z', 'zippy', '▶', '( see', '. 50', '. washed', '10 /', '5 pairs', 'all 4', 'also comes', 'cards .', 'gold .', 'how to', 'in .', 'it cosmetics', 'mail .', 'on them', 'perfect to', 'please note', 'push -', 'really good', 'slip on', 'the dress', 'this item', '9 . 5', 'bke', 'california', 'certain', 'covergirl', 'elegant', 'exelent', 'ikea', 'inbox', 'keurig', 'knit', 'lashes', 'lower', 'oo', 'pablo', 'playing', 'shaft', 'skulls', 'sonic', 'syringes', '* size', 'a box', 'authentic ,', 'eyeshadow palette', 'in length', 'is only', 'listings !', 'nwt size', 'sale .', 'solid black', 'still on', 'will receive', 'been used .', 'free to ask', 'bebop', 'adult', 'view', 'come in', 'you for looking', '26', 'donutella', 'hands', 'isn', 'matc', 'peel', 'plaster', 'recommendations', 'slides', '] 2', 'it to', 'save !', 'so it', 't like', 'thanks !', 'and save !', 'will be shipped', '1080p', '1900', '1w', '4x', '600d', 'accepting', 'age', 'aswell', 'bale', 'bookmark', 'champion', 'chips', 'copper', 'corrector', 'dirty', 'du', 'eos', 'grit', 'guide', 'hookups', 'hoped', 'interview', 'jaybird', 'jefree', 'kept', 'miranda', 'modesty', 'nighter', 'parallel', 'persian', 'pg', 'pocketbac', 'silversmith', 'smokefree', 'steelbook', 'stinks', 'ted', 'thirty', 'timewise', 'tsp', 'upgrading', 'whirl', 'wholesale', 'writing', '! you', '5 )', 'at &', 'listings to', 'lularoe lularoe', 'missing both', 'once !', 'one side', 'one you', 'sure if', 'with your', 'if you want', '11c', '1d', '32oz', '50in', '5fl', 'almost', 'angora', 'aussie', 'autographed', 'beary', 'beauties', 'bellami', 'blaster', 'boxycharm', 'bucket', 'bulldogs', 'bullet', 'cancellations', 'cannot', 'carabiner', 'clarifying', 'clarity', 'collar', 'colorstay', 'dashiki', 'davidson', 'ding', 'diy', 'erased', 'exam', 'exfoliator', 'feb', 'fibers', 'fin', 'fisheye', 'flyknit', 'geek', 'giveaway', 'glide', 'grabs', 'headpiece', 'hem', 'highlight', 'inexpensive', 'killstar', 'levi', 'madewell', 'mewtwo', 'moonstone', 'neiman', 'nerd', 'newest', 'opaque', 'padded', 'pebbled', 'pistol', 'pouchette', 'projects', 'prongs', 'pumps', 'raw', 'reads', 'requested', 'resurfacing', 'rhinestone', 'roof', 'ross', 'rowley', 'sandal', 'shoreline', 'siberian', 'sim', 'siri', 'skye', 'smaller', 'sooo', 'sooooo', 'souvenir', 'soy', 'statue', 'sterilized', 'suprise', 'surface', 'tossing', 'tramp', 'treatments', 'tribe', 'twilight', 'varsity', 'volu', 'windbreaker', '! so', '. can', '. excellent', '. paid', ': black', 'black /', 'but in', 'great quality', 'high rise', 'high waist', 'items :', 'jeans .', 'love these', 'make sure', 'missing excellent', 'more .', 'my daughter', 'non -', 'ships fast', 'shirt is', 'shown .', 'small stain', 'the fabric', 'to use', 'very nice', 'with dust', 'without tag', '• brand', '34c', '5ft', 'auth', 'birkenstock', 'bridesmaids', 'campaign', 'clay', 'comic', 'cost', 'crack', 'deco', 'donna', 'earring', 'eggs', 'elton', 'featherweight', 'figurine', 'hence', 'irons', 'lane', 'loss', 'magnetic', 'margiela', 'morphe', 'nes', 'origami', 'program', 'reflective', 'rolex', 'secrets', 'sneakers', 'start', 'trying', 'twists', 'weekend', '% polyester', '& co', "' oreal", ', like', ', pet', '- neck', '. selling', '/ 6', '30 "', 'and charger', 'apple iphone', "carter '", 'for samsung', 'jeans size', 'retails at', 'shades of', 'skin ,', 'what i', 'within 24', 'yoga pants', '. never worn', "pink victoria '", 'without tags .', 'discs', 'near', 'provided', 'sc', 'tsum', 'wrappers', "levi '", 'lots of', 'olaplex', 'invicta', 'arrows', 'bose', 'jenner', 'teeki', '% brand', 'all prices', 'are not', 'as seen', 'been opened', 'pet friendly', 'with no', '23', '30x30', '36ddd', 'amazingly', 'angels', 'bang', 'buttered', 'calculator', 'camille', 'cutting', 'duster', 'electric', 'elevated', 'exterior', 'fantastic', 'gaming', 'heroes', 'instruction', 'jouer', 'mojave', 'patricia', 'payton', 'perla', 'ponytail', 'rad', 'shake', 'sharpie', 'shatter', 'sour', 'technology', 'totaling', 'veronica', 'words', 'xp', '! the', '. each', '2 x', '] per', 'a clean', 'day .', 'items come', 'lilly pulitzer', 'listed .', 'pink logo', 'suitable for', 'washed once', 'with leggings', 'wore once', 'your size', 'condition . size', 'free shipping *', 'please do not', '148', '2004', '27', '2x2', '3cm', '4t', '6y', 'aerosoles', 'alcohol', 'ant', 'athletics', 'badge', 'beverly', 'bisque', 'blog', 'canopy', 'capris', 'cellular', 'cetaphil', 'cindy', 'clearly', 'concert', 'contacts', 'creek', 'dallas', 'dictionary', 'directions', 'fenty', 'films', 'filofax', 'firming', 'five', 'flowered', 'geometric', 'guides', 'individuals', 'jeweled', 'juicer', 'kinda', 'labeled', 'lash', 'lilkittylady', 'linda', 'mainstays', 'marl', 'microsoft', 'minimum', 'mother', 'motivational', 'naturalizer', 'northface', 'platters', 'process', 'reborn', 'reusable', 'round', 'russe', 'sculpt', 'searched', 'seresto', 'servings', 'sitting', 'smartwatch', 'stabilizer', 'stardust', 'strapback', 'suki', 'tagged', 'teens', 'tin', 'uses', 'vachetta', 'wet', 'workshop', 'yru', ', medium', '3 items', ': 0', 'and they', 'bundle &', 'bundle .', 'could be', 'dress size', 'firm on', 'includes all', 'item is', 'lularoe .', 'micro usb', 'missing you', 'sephora brand', 'signs of', 'small hole', 'the side', 'very comfortable', 'worn one', '! check out', ', etc .', 'of life left', 'axis', 'being', 'blueberries', 'cassie', 'cement', 'circles', 'colognes', 'colorblock', 'dance', 'evil', 'flag', 'flattering', 'frontline', 'gardening', 'givenchy', 'handwritten', 'insured', 'jedi', 'kaya', 'kodi', 'kyliner', 'marcelle', 'marquise', 'outer', 'photocard', 'pods', 'polaroids', 'posite', 'preowned', 'quartz', 'recipes', 'securely', 'spiders', 'tac', 'thrasher', 'ty', 'vote', ', fashion', '2 in', 'and washed', 'asking [', 'black with', 'down the', 'lime green', 'missing large', 'missing two', 'not noticeable', 'pink ,', 'pink vs', 'size -', 'them are', 'twice .', 'wallet .', '. price firm', 'i will not', 'save on shipping', 'true to size', 'carton', 'cowboys', 'curtain', 'engineering', 'express', 'sega', 'tryons', 'condition no', 'good shape', 'x 1', ', perfect', ': 2', 'is a size', '3in', 'casual', 'dollar', 'eyebrows', 'subculture', 'too many', '34b', '7days', 'absolue', 'ambiance', 'bam', 'calendar', 'clips', 'corner', 'denims', 'fire', 'horse', 'imperfections', 'koozies', 'nordstrom', 'quickly', 'sack', 'shedding', 'sonicare', 'sown', 'straighten', 'tangled', 'tracker', 'undervisor', 'upgraded', 'zx', '! -', '" d', ', nwt', 'are [', 'for bundles', 'hair .', 'leggings ,', 'less than', 'long .', 'pieces of', 'sorry no', 'this .', '2010', 'aid', 'att', 'bailey', 'batmobile', 'benzoyl', 'bfyhc', 'bloomingdales', 'bread', 'breyer', 'bush', 'camo', 'cave', 'cd', 'certificates', 'chaco', 'charges', 'chenille', 'chestnut', 'colorway', 'commercial', 'cons', 'credits', 'cutters', 'dies', 'digitizer', 'dont', 'dot', 'earphone', 'earpieces', 'existing', 'fioni', 'foamposites', 'folk', 'fresheners', 'fuentes', 'glory', 'goku', 'hang', 'homecoming', 'homepage', 'i3', 'kansas', 'laura', 'lettering', 'methods', 'mets', 'mi', 'microdermabrasion', 'moth', 'ne', 'notifications', 'oud', 'pains', 'pennies', 'pique', 'players', 'pouches', 'powershot', 'putter', 'resellers', 'ruby', 'sand', 'seeker', 'shipment', 'sides', 'sizing', 'slide', 'slot', 'smelling', 'spain', 'sphere', 'stuffer', 'swaddles', 'ten', 'unbranded', 'upto', 'vapor', 'vichy', 'village', 'widescreen', 'wording', '* i', ', all', '. oz', '. small', '1 black', '8 "', 'great used', 'have never', 'i could', 'is an', 'it looks', 'large but', 'large size', 'leather ,', 'love it', 'of jewelry', 'opened .', 'pink size', 'save more', 'straps .', 'the last', 'these have', 'to buy', 'used or', 'v neck', '. * *', 'condition . no', '04', '18month', '190', 'ag', 'alligator', 'anderson', 'bahama', 'bassinet', 'baublebar', 'belted', 'bordeaux', 'boxer', 'clubs', 'cod', 'coveralls', 'dad', 'dicks', 'drusy', 'field', 'foams', 'ghouls', 'glacier', 'greetings', 'heard', 'hubby', 'initial', 'intimately', 'jfc1010', 'khaki', 'landing', 'lifeproof', 'mineral', 'morning', 'needed', 'needle', 'oreal', 'owl', 'peppermint', 'pies', 'placement', 'poppin', 'retro', 'robin', 'rubbermaid', 'sampler', 'shadows', 'shorter', 'sleeved', 'smart', 'tactical', 'textbooks', 'tick', 'tot', 'trefoil', 'ultron', 'venom', 'wearing', 'wot', 'yarn', '! please', '& girl', ', pink', '. inside', '3 times', 'all natural', 'can fit', 'get one', 'great deal', 'handful of', 'have to', 'in your', 'lightweight ,', 'lot .', 'missing lot', 'new 100', 'no offers', 'once ,', 'pics for', 'pre -', 'purse ,', 's -', 'see is', 'set is', 'shoulder strap', 'such a', 'take a', 'this price', 'to take', 'to your', 'ugg australia', 'you would', 'alexander', 'amelia', 'customized', 'device', 'fidget', 'hermes', 'itty', 'kaws', 'midi', 'pez', 'redefine', 'wild', 'design .', 'time to', 'would like', 'like new ,']


def rec_time(title):
    if mode:
        time_records[title] = int(time())


def print_info(title, message=None):
    if mode:
        last_time = time_records.get(title, 0)
        cur_time = int(time())
        if message is None:
            print('【%s】【%ds】【%s】' % (current_process().name, cur_time - last_time, title))
        else:
            print('【%s】【%ds】【%s】【%s】' % (current_process().name, cur_time - last_time, title, message))
            time_records[0] = cur_time


def read_data(data_dir='../input'):
    def fillna(df):
        df['brand_name'].fillna('missing', inplace=True)
        df['item_description'].fillna('None', inplace=True)
        df['category_name'].fillna('Other', inplace=True)
        df['name'].fillna('Unknown', inplace=True)
        df['item_condition_id'] = df.item_condition_id.astype(np.int16).fillna(0)
        df['shipping'] = df.shipping.astype(np.uint8).fillna(0)

    rec_time('read data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', engine='c')
    train_df['price'] = np.log1p(train_df.price)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', engine='c')
    print_info('read data', '%s %s' % (train_df.shape, test_df.shape))

    # test_df = test_df.append(test_df).append(test_df).append(test_df).append(test_df).append(test_df, ignore_index=True)

    rec_time('remove item with 0 price')
    train_df = train_df.loc[train_df.price > 1].reset_index(drop=True)
    print_info('remove item with 0 price', train_df.shape)

    rec_time('fillna')
    fillna(train_df)
    fillna(test_df)
    print_info('fillna', '%s %s' % (train_df.shape, test_df.shape))

    return train_df, test_df


def extract_cats(col):
    na_val = 'Other'
    cats = col.str.split('/')
    cat_len = cats.str.len()
    cat1 = cats.str.get(0)
    cat2 = cats.str.get(1)
    cat2.fillna(na_val, inplace=True)
    cat3 = cats.str.get(2)
    cat3.fillna(na_val, inplace=True)
    cat_entity = cats.str.get(-1)
    cat_n = cat_entity.copy()
    cat_n.loc[cat_len <= 3] = na_val
    return cat_len, cat1, cat2, cat3, cat_entity, cat_n


def embed_target_agg(col, statistical_size=5, k=5, random_state=0):
    if col.shape[0] < statistical_size:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        means = []
        stds = []
        mins = []
        q1s = []
        medians = []
        q3s = []
        maxes = []
        for _, vind in KFold(n_splits=k, shuffle=True, random_state=random_state).split(col):
            v_col = col.take(vind)
            means.append(np.mean(v_col))
            stds.append(np.std(v_col))
            mins.append(np.min(v_col))
            q1s.append(np.percentile(v_col, 25))
            medians.append(np.median(v_col))
            q3s.append(np.percentile(v_col, 75))
            maxes.append(np.max(v_col))
        return np.round(np.mean(means)), np.round(np.std(means)), np.round(np.mean(stds)), np.round(
            np.std(stds)), np.round(np.mean(mins)), np.round(np.std(mins)), np.round(np.mean(q1s)), np.round(
            np.std(q1s)), np.round(np.mean(medians)), np.round(np.std(medians)), np.round(np.mean(q3s)), np.round(
            np.std(q3s)), np.round(np.mean(maxes)), np.round(np.std(maxes))


def embed_thread(train_df, test_df, key_col_name, extra_cols, random_state=5000):
    rec_time('%s embed target' % key_col_name)
    cols = extra_cols + [key_col_name]
    gp_embed = train_df[cols + ['price']].groupby(cols).agg(embed_target_agg, **{'random_state': random_state})
    embed_col_suffix = '_' + key_col_name + '_embed'
    tr_embed = np.array(list(train_df.join(gp_embed, on=cols, rsuffix=embed_col_suffix)['price' + embed_col_suffix]),
                        dtype=np.float32)
    ts_embed = test_df.join(gp_embed, on=cols)['price']
    ts_embed.loc[ts_embed.isnull()] = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] * np.sum(ts_embed.isnull())
    ts_embed = np.array(list(ts_embed), dtype=np.float32)
    print_info('%s embed target' % key_col_name)

    return tr_embed, ts_embed


def encode_text(col):
    def count_chars(txt):
        _len = 0
        digit_cnt, number_cnt = 0, 0
        lower_cnt, upper_cnt, letter_cnt, word_cnt = 0, 0, 0, 0
        char_cnt, term_cnt = 0, 0
        conj_cnt, blank_cnt, punc_cnt = 0, 0, 0
        sign_cnt, marks_cnt = 0, 0

        flag = 10
        for ch in txt:
            _len += 1
            if ch in string.ascii_lowercase:
                lower_cnt += 1
                letter_cnt += 1
                char_cnt += 1
                if flag:
                    word_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 0
            elif ch in string.ascii_uppercase:
                upper_cnt += 1
                letter_cnt += 1
                char_cnt += 1
                if flag:
                    word_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 0
            elif ch in string.digits:
                digit_cnt += 1
                char_cnt += 1
                if 1 != flag:
                    number_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 1
            elif '_' == ch:
                conj_cnt += 1
                char_cnt += 1
                if flag > 2:
                    term_cnt += 1
                flag = 2
            elif ch in string.whitespace:
                blank_cnt += 1
                flag = 3
            elif ch in string.punctuation:
                punc_cnt += 1
                flag = 4
            else:
                sign_cnt += 1
                if flag != 5:
                    marks_cnt += 1
                    flag = 5

        return (
            _len, digit_cnt, number_cnt, digit_cnt / (1 + number_cnt), lower_cnt, upper_cnt, letter_cnt, word_cnt,
            letter_cnt / (1 + word_cnt), char_cnt, term_cnt, char_cnt / (1 + term_cnt), conj_cnt, blank_cnt,
            punc_cnt, sign_cnt, marks_cnt, sign_cnt / (1 + marks_cnt))

    return np.array(list(col.apply(count_chars)), dtype=np.uint16)


def desc_cnt_thread(tr_desc_col, ts_desc_col):
    col_name = tr_desc_col.name
    rec_time('%s encode_text' % col_name)
    tr_desc_cnts = encode_text(tr_desc_col)
    ts_desc_cnts = encode_text(ts_desc_col)
    print_info('%s encode_text' % col_name)

    return tr_desc_cnts, ts_desc_cnts


def cat_extract_thread(tr_cat_col, ts_cat_col):
    rec_time('extract cats')
    tr_cat_len, tr_cat1, tr_cat2, tr_cat3, tr_cat_entity, tr_cat_n = extract_cats(tr_cat_col)
    ts_cat_len, ts_cat1, ts_cat2, ts_cat3, ts_cat_entity, ts_cat_n = extract_cats(ts_cat_col)
    print_info('extract cats')
    rec_time('calc cat cnt')
    cnts = tr_cat_entity.append(ts_cat_entity).value_counts()
    tr_cat_entity_cnt = tr_cat_entity.to_frame().join(cnts, on='category_name', rsuffix='_cnt')[
        'category_name_cnt']
    ts_cat_entity_cnt = ts_cat_entity.to_frame().join(cnts, on='category_name', rsuffix='_cnt')[
        'category_name_cnt']
    print_info('calc cat cnt')

    return (np.vstack([tr_cat_len, tr_cat_entity_cnt]).T, np.vstack([ts_cat_len, ts_cat_entity_cnt]).T), (
        tr_cat1, tr_cat2, tr_cat3, tr_cat_entity, tr_cat_n), (ts_cat1, ts_cat2, ts_cat3, ts_cat_entity, ts_cat_n)


def item_cond_thread(tr_cond_col, ts_cond_col):
    col_name = tr_cond_col.name
    rec_time('%s process' % col_name)
    tr_x_cond = pd.get_dummies(tr_cond_col).values
    ts_cond_col = ts_cond_col.clip(1, 5)
    ts_x_cond = pd.get_dummies(ts_cond_col).values
    print_info('%s process' % col_name)
    return tr_x_cond, ts_x_cond


def ship_thread(tr_ship_col, ts_ship_col):
    col_name = tr_ship_col.name
    rec_time('%s process' % col_name)
    tr_x_ship = pd.get_dummies(tr_ship_col).values
    ts_x_ship = pd.get_dummies(ts_ship_col).values
    print_info('%s process' % col_name)
    return tr_x_ship, ts_x_ship


def nn_cat_thread(tr_cat_col, ts_cat_col):
    col_name = tr_cat_col.name
    rec_time('%s label encode' % col_name)
    ler = LabelEncoder()
    ler.fit(np.hstack([tr_cat_col, ts_cat_col]))
    nn_tr_x_cat = ler.transform(tr_cat_col).astype(np.uint32)
    nn_ts_x_cat = ler.transform(ts_cat_col).astype(np.uint32)
    print_info('%s label encode' % col_name)
    return nn_tr_x_cat, nn_ts_x_cat, ler.classes_.shape[0]


def nn_brand_thread(tr_brand_col, ts_brand_col):
    col_name = tr_brand_col.name
    rec_time('%s label encode' % col_name)
    ler = LabelEncoder()
    ler.fit(np.hstack([tr_brand_col, ts_brand_col]))
    nn_tr_x_brand = ler.transform(tr_brand_col).astype(np.uint32)
    nn_ts_x_brand = ler.transform(ts_brand_col).astype(np.uint32)
    print_info('%s label encode' % col_name)
    return nn_tr_x_brand, nn_ts_x_brand, ler.classes_.shape[0]


def brand_thread(tr_brand_col, ts_brand_col):
    col_name = tr_brand_col.name
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern='.+', min_df=10, binary=True, dtype=np.uint8)
    tr_x_brand = vr.fit_transform(tr_brand_col)
    print_info('train %s vectorize' % col_name)
    rec_time('test %s vectorize' % col_name)
    ts_x_brand = vr.transform(ts_brand_col)
    print_info('test %s vectorize' % col_name)
    return tr_x_brand, ts_x_brand


def nn_name_thread(tr_name_col, ts_name_col):
    column_name = tr_name_col.name
    rec_time('%s tokenizer' % column_name)
    tkr = Tokenizer()
    tkr.fit_on_texts(tr_name_col)
    nn_tr_x_name = np.array(tkr.texts_to_sequences(tr_name_col))
    nn_ts_x_name = np.array(tkr.texts_to_sequences(ts_name_col))
    print_info('%s tokenizer' % column_name)
    rec_time('%s pad_sequences' % column_name)
    nn_tr_x_name = pad_sequences(nn_tr_x_name, maxlen=max_len_dic[column_name], truncating='post', padding='post')
    nn_ts_x_name = pad_sequences(nn_ts_x_name, maxlen=max_len_dic[column_name], truncating='post', padding='post')
    print_info('%s pad_sequences' % column_name)
    return nn_tr_x_name, nn_ts_x_name, len(tkr.word_index) + 1


def name_thread(tr_name_col, tr_brand_col, ts_name_col, ts_brand_col, name_queue):
    col_name = tr_name_col.name
    rec_time('train %s process' % col_name)
    tr_name_col = tr_name_col.str.replace(r'\s\s+', ' ')
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 3), vocabulary=name_terms, dtype=np.uint8)
    tr_x_name = vr.fit_transform(tr_brand_col + ' ' + tr_name_col)
    print_info('train %s vectorize' % col_name, 'name vocabulary size is %d.' % len(name_terms))
    rec_time('train %s encode_text' % col_name)
    tr_x_name_cnts = encode_text(tr_name_col)
    print_info('train %s encode_text' % col_name)
    print_info('train %s process' % col_name)

    rec_time('test %s process' % col_name)
    ts_name_col = ts_name_col.str.replace(r'\s\s+', ' ')
    rec_time('test %s vectorize' % col_name)
    ts_x_name = vr.transform(ts_brand_col + ' ' + ts_name_col)
    print_info('test %s vectorize' % col_name)
    rec_time('test %s encode_text' % col_name)
    ts_x_name_cnts = encode_text(ts_name_col)
    print_info('test %s encode_text' % col_name)
    print_info('test %s process' % col_name)

    nn_tr_x_name, nn_ts_x_name, name_len = nn_name_thread(tr_name_col, ts_name_col)
    del tr_name_col, ts_name_col
    gc.collect()

    col_name = tr_brand_col.name
    rec_time('%s process' % col_name)
    cnts = tr_brand_col.append(ts_brand_col).value_counts()
    tr_brand_cnt = tr_brand_col.to_frame().join(cnts, on=col_name, rsuffix='_cnt')[col_name + '_cnt']
    ts_brand_cnt = ts_brand_col.to_frame().join(cnts, on=col_name, rsuffix='_cnt')[col_name + '_cnt']
    tr_brand_len = tr_brand_col.str.len()
    ts_brand_len = ts_brand_col.str.len()
    tr_x_brand_cnts = np.vstack([tr_brand_len, tr_brand_cnt]).T
    ts_x_brand_cnts = np.vstack([ts_brand_len, ts_brand_cnt]).T
    print_info('%s process' % col_name)
    del cnts, tr_brand_len, tr_brand_cnt, ts_brand_len, ts_brand_cnt
    gc.collect()

    tr_x_brand, ts_x_brand = brand_thread(tr_brand_col, ts_brand_col)

    nn_tr_x_brand, nn_ts_x_brand, brand_len = nn_brand_thread(tr_brand_col, ts_brand_col)
    del tr_brand_col, ts_brand_col
    gc.collect()

    name_queue.put((tr_x_name, tr_x_name_cnts, ts_x_name, ts_x_name_cnts, nn_tr_x_name, nn_ts_x_name, name_len,
                    tr_x_brand_cnts, ts_x_brand_cnts, tr_x_brand, ts_x_brand, nn_tr_x_brand, nn_ts_x_brand, brand_len))


def nn_desc_thread(tr_desc_col, ts_desc_col, nn_desc_queue):
    column_name = tr_desc_col.name
    rec_time('train %s tokenizer' % column_name)
    tkr = Tokenizer()
    tkr.fit_on_texts(tr_desc_col)
    nn_tr_x_desc = np.array(tkr.texts_to_sequences(tr_desc_col))
    nn_tr_x_desc = pad_sequences(nn_tr_x_desc, maxlen=max_len_dic[column_name], truncating='post', padding='post')
    print_info('train %s tokenizer' % column_name)
    del tr_desc_col
    gc.collect()

    rec_time('test %s tokenizer' % column_name)
    nn_ts_x_desc = np.array(tkr.texts_to_sequences(ts_desc_col))
    nn_ts_x_desc = pad_sequences(nn_ts_x_desc, maxlen=max_len_dic[column_name], truncating='post', padding='post')
    print_info('test %s tokenizer' % column_name)
    del ts_desc_col
    gc.collect()

    nn_desc_queue.put((nn_tr_x_desc, nn_ts_x_desc, len(tkr.word_index) + 1))


def desc_vectorize_test_thread(data_id, vr, ts_desc_col, desc_queue):
    col_name = 'item_description'
    rec_time('test part%d %s vectorize transform' % (data_id, col_name))
    ts_x_desc = vr.transform(ts_desc_col)
    print_info('test part%d %s vectorize transform' % (data_id, col_name))
    desc_queue.put((data_id, ts_x_desc))


def desc_vectorize_thread(tr_desc_col, tr_brand_col, ts_desc_col, ts_brand_col, desc_queue):
    col_name = tr_desc_col.name
    rec_time('train %s process' % col_name)
    tr_desc_col = tr_desc_col.str.replace(r'\s\s+', ' ')
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 3), vocabulary=desc_terms, dtype=np.uint8)
    tr_x_desc = vr.fit_transform(tr_brand_col + ' ' + tr_desc_col)
    print_info('train %s vectorize' % col_name, 'desc vocabulary size is %d.' % len(desc_terms))
    print_info('train %s process' % col_name)
    del tr_desc_col, tr_brand_col
    gc.collect()

    rec_time('test %s process' % col_name)
    ts_desc_col = ts_desc_col.str.replace(r'\s\s+', ' ')
    rec_time('test %s vectorize' % col_name)
    ts_col = ts_brand_col + ' ' + ts_desc_col
    ts_part_data_len = int(0.32 * ts_col.shape[0])
    rec_time('test %s vectorize transform' % col_name)
    del ts_brand_col, ts_desc_col
    gc.collect()

    ts_desc_queue = Queue()
    worker1 = get_worker(func=desc_vectorize_test_thread, name='desc_vectorize_thread_worker1',
                         args=(1, vr, ts_col[ts_part_data_len: 2 * ts_part_data_len], ts_desc_queue))
    worker1.start()
    worker2 = get_worker(func=desc_vectorize_test_thread, name='desc_vectorize_thread_worker2',
                         args=(2, vr, ts_col[2 * ts_part_data_len:], ts_desc_queue))
    worker2.start()

    ts_x_desc0 = vr.transform(ts_col[:ts_part_data_len])
    del ts_col
    gc.collect()

    data_id_12, ts_x_desc_12 = ts_desc_queue.get()
    data_id_21, ts_x_desc_21 = ts_desc_queue.get()
    ts_desc_queue.close()
    if 1 == data_id_12:
        ts_x_desc = vstack((ts_x_desc0, ts_x_desc_12, ts_x_desc_21))
    else:
        ts_x_desc = vstack((ts_x_desc0, ts_x_desc_21, ts_x_desc_12))
    worker1.join()
    worker2.join()
    print_info('test %s vectorize transform' % col_name)
    print_info('test %s process' % col_name)
    del worker1, worker2, ts_desc_queue, ts_x_desc_12, ts_x_desc_21
    gc.collect()

    desc_queue.put((tr_x_desc, ts_x_desc))


def cat_vectorize_thread(col_name, tr_cat_col, ts_cat_col, cat_queue):
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat = vr.fit_transform(tr_cat_col)
    ts_x_cat = vr.transform(ts_cat_col)
    print_info('%s vectorize' % col_name)
    cat_queue.put((tr_x_cat, ts_x_cat))


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def get_data():
    train_df, test_df = read_data()
    y = train_df.price.copy().values
    submission = test_df[['test_id']]

    nn_x = {}
    nn_ts_x = {}
    len_dic = {}

    col_name = 'item_description'
    rec_time('%s normalize' % col_name)
    train_df[col_name].replace('No description yet', 'None', inplace=True)
    test_df[col_name].replace('No description yet', 'None', inplace=True)
    print_info('%s normalize' % col_name)

    desc_queue = Queue()
    desc_worker = get_worker(desc_vectorize_thread, name='desc_vectorize_thread',
                             args=(train_df.item_description, train_df.brand_name, test_df.item_description,
                                   test_df.brand_name, desc_queue))
    desc_worker.start()

    nn_desc_queue = Queue()
    nn_desc_worker = get_worker(nn_desc_thread, name='nn_desc_thread',
                                args=(train_df.item_description, test_df.item_description, nn_desc_queue))
    nn_desc_worker.start()

    name_queue = Queue()
    name_worker = get_worker(name_thread, args=(train_df['name'], train_df.brand_name, test_df['name'],
                                                test_df.brand_name, name_queue), name='name_thread')
    name_worker.start()
    del train_df['name'], test_df['name']
    gc.collect()

    col_name = 'item_condition_id'
    tr_x_cond, ts_x_cond = item_cond_thread(train_df[col_name], test_df[col_name])
    nn_x[col_name] = tr_x_cond
    nn_ts_x[col_name] = ts_x_cond

    col_name = 'shipping'
    tr_x_ship, ts_x_ship = ship_thread(train_df[col_name], test_df[col_name])
    nn_x[col_name] = tr_x_cond
    nn_ts_x[col_name] = ts_x_cond

    tr_desc_cnts, ts_desc_cnts = desc_cnt_thread(train_df.item_description, test_df.item_description)
    del train_df['item_description'], test_df['item_description']
    gc.collect()

    tr_x_brand_embeds, ts_x_brand_embeds = embed_thread(train_df, test_df, 'brand_name',
                                                        ['item_condition_id', 'shipping'], random_state=5001)
    del train_df['brand_name'], test_df['brand_name']
    gc.collect()

    col_name = 'category_name'
    x_col_name = col_name + '_ind'
    nn_tr_x_cat, nn_ts_x_cat, len_dic[x_col_name] = nn_cat_thread(train_df[col_name], test_df[col_name])
    nn_x[x_col_name] = nn_tr_x_cat
    nn_ts_x[x_col_name] = nn_ts_x_cat
    del nn_tr_x_cat, nn_ts_x_cat
    gc.collect()

    (tr_cat_cnts, ts_cat_cnts), (tr_cat1, tr_cat2, tr_cat3, train_df['cat_entity'], tr_cat_n), (
        ts_cat1, ts_cat2, ts_cat3, test_df['cat_entity'], ts_cat_n) = cat_extract_thread(train_df.category_name,
                                                                                         test_df.category_name)
    tr_x_cat_embeds, ts_x_cat_embeds = embed_thread(train_df, test_df, 'cat_entity', ['item_condition_id', 'shipping'],
                                                    random_state=5000)
    del train_df, test_df
    gc.collect()

    (tr_x_name, tr_x_name_cnts, ts_x_name, ts_x_name_cnts, nn_x['name_seq'], nn_ts_x['name_seq'], len_dic['name_seq'],
     tr_x_brand_cnts, ts_x_brand_cnts, tr_x_brand, ts_x_brand, nn_x['brand_name_ind'], nn_ts_x['brand_name_ind'],
     len_dic['brand_name_ind']) = name_queue.get()
    name_queue.close()
    name_worker.join()
    del name_queue, name_worker
    gc.collect()

    col_name = 'item_description'
    x_col_name = col_name + '_seq'
    nn_tr_x_desc, nn_ts_x_desc, len_dic[x_col_name] = nn_desc_queue.get()
    nn_desc_queue.close()
    nn_desc_worker.join()
    nn_x[x_col_name] = nn_tr_x_desc
    nn_ts_x[x_col_name] = nn_ts_x_desc
    del nn_tr_x_desc, nn_ts_x_desc, nn_desc_queue, nn_desc_worker
    gc.collect()

    rec_time('combine cnt features')
    x_cnts = np.hstack([tr_x_brand_embeds, tr_x_brand_cnts, tr_x_cat_embeds, tr_cat_cnts, tr_x_name_cnts,
                        tr_desc_cnts]).astype(np.float32)
    ts_x_cnts = np.hstack([ts_x_brand_embeds, ts_x_brand_cnts, ts_x_cat_embeds, ts_cat_cnts, ts_x_name_cnts,
                           ts_desc_cnts]).astype(np.float32)

    scaler = MinMaxScaler()
    scaled_x_cnts = scaler.fit_transform(x_cnts).astype(np.float32)
    scaled_ts_x_cnts = scaler.transform(ts_x_cnts).astype(np.float32)
    nn_x['cnts'] = scaled_x_cnts
    nn_ts_x['cnts'] = scaled_ts_x_cnts
    print_info('combine cnt features')
    del scaled_x_cnts, scaled_ts_x_cnts
    gc.collect()

    tr_x_desc, ts_x_desc = desc_queue.get()
    desc_queue.close()
    desc_worker.join()
    del desc_worker, desc_queue
    gc.collect()

    rec_time('combine features 0')
    x = hstack([x_cnts, tr_x_ship, tr_x_cond, tr_x_brand, tr_x_name, tr_x_desc], dtype=np.float32)
    del x_cnts, tr_x_ship, tr_x_cond, tr_x_brand, tr_x_name, tr_x_desc
    gc.collect()

    if ts_x_cnts.shape[0] < max_data_size:
        ts_x = hstack([ts_x_cnts, ts_x_ship, ts_x_cond, ts_x_brand, ts_x_name, ts_x_desc], dtype=np.float32)
    else:
        ts_x = combine_features([ts_x_cnts, ts_x_ship, ts_x_cond, ts_x_brand, ts_x_name, ts_x_desc])
    del ts_x_cnts, ts_x_ship, ts_x_cond, ts_x_brand, ts_x_name, ts_x_desc
    gc.collect()
    print_info('combine features 0')

    cat_queue2 = Queue()
    cat_worker2 = get_worker(cat_vectorize_thread, name='cat_vectorize_thread2',
                             args=('cat2', tr_cat2, ts_cat2, cat_queue2))
    cat_worker2.start()
    del tr_cat2, ts_cat2
    gc.collect()

    cat_queue3 = Queue()
    cat_worker3 = get_worker(cat_vectorize_thread, name='cat_vectorize_thread3',
                             args=('cat3', tr_cat3, ts_cat3, cat_queue3))
    cat_worker3.start()
    del tr_cat3, ts_cat3
    gc.collect()

    cat_queue_n = Queue()
    cat_worker_n = get_worker(cat_vectorize_thread, name='cat_vectorize_thread_n',
                              args=('cat_n', tr_cat_n, ts_cat_n, cat_queue_n))
    cat_worker_n.start()
    del tr_cat_n, ts_cat_n
    gc.collect()

    col_name = 'cat1'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat1 = vr.fit_transform(tr_cat1)
    ts_x_cat1 = vr.transform(ts_cat1)
    print_info('%s vectorize' % col_name)
    del tr_cat1, ts_cat1
    gc.collect()

    tr_x_cat2, ts_x_cat2 = cat_queue2.get()
    cat_queue2.close()
    cat_worker2.join()
    del cat_worker2, cat_queue2
    gc.collect()

    tr_x_cat3, ts_x_cat3 = cat_queue3.get()
    cat_queue3.close()
    cat_worker3.join()
    del cat_worker3, cat_queue3
    gc.collect()

    tr_x_cat_n, ts_x_cat_n = cat_queue_n.get()
    cat_queue_n.close()
    cat_worker_n.join()
    del cat_worker_n, cat_queue_n
    gc.collect()

    rec_time('combine features 1')
    x = hstack([x, tr_x_cat1, tr_x_cat2, tr_x_cat3, tr_x_cat_n], dtype=np.float32).tocsr()
    del tr_x_cat1, tr_x_cat2, tr_x_cat3, tr_x_cat_n
    gc.collect()

    if ts_x.shape[0] < max_data_size:
        ts_x = hstack([ts_x, ts_x_cat1, ts_x_cat2, ts_x_cat3, ts_x_cat_n], dtype=np.float32).tocsr()
    else:
        ts_x = combine_features([ts_x, ts_x_cat1, ts_x_cat2, ts_x_cat3, ts_x_cat_n])
    del ts_x_cat1, ts_x_cat2, ts_x_cat3, ts_x_cat_n
    gc.collect()
    print_info('combine features 1', '%s %s' % (x.shape, ts_x.shape))

    return submission, y, x, ts_x, nn_x, nn_ts_x, len_dic


def get_worker(func, args, name=None):
    worker = Process(target=func, args=args, name=name)
    return worker


class LgbTrainer:
    def __init__(self, n_round, params):
        self.num_boost_round = n_round
        self.params = params
        self.model = None

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, x, y):
        self.model = lgb.train(self.params, lgb.Dataset(x, label=y), num_boost_round=self.num_boost_round)
        return self

    def predict(self, x):
        return self.model.predict(x)


def measure_handler(target, pred):
    return metrics.mean_squared_error(target, pred) ** 0.5


param_cache = {'LgbTrainer-learning_rate': {"{'alpha'- 0.9, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'feature_fraction'- 0.9, 'lambda_l1'- 0, 'lambda_l2'- 0}": {0.1: (0.428283333695869, 0.00022696602809427202), 0.15625: (0.4243262071703957, 0.00015019371604202414), 0.2: (0.42459300596211297, 6.413822909057121e-06), 0.1140625: (0.42668043678626166, 6.45624498306907e-05), 0.128125: (0.42561759657517306, 0.00018487136148029326), 0.1421875: (0.4249251656380728, 0.000245961610084916), 0.1703125: (0.4241691748892791, 7.553672844401449e-05), 0.184375: (0.42449427293029335, 5.691593135986883e-05), 0.1984375: (0.42463905900269666, 0.0002778444675942682), 0.2125: (0.424764260494135, 0.00018230798130525194)}}, 'LgbTrainer-alpha': {"{'learning_rate'- 0.1703125, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'feature_fraction'- 0.9, 'lambda_l1'- 0, 'lambda_l2'- 0}": {0.2: (0.4403266489094749, 0.0005798368751046556), 0.4: (0.42851933045353396, 0.0005337929809554609), 0.8: (0.4241867258754856, 0.00034266703335156246), 0.9: (0.42428909889918837, 0.0001269412994774899), 0.5: (0.42609153332145555, 0.00037897883528006937), 0.6: (0.42508681216071625, 0.0005198334536579696), 0.7: (0.4245099126814109, 0.0005254943183174154)}}, 'LgbTrainer-num_leaves': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'feature_fraction'- 0.9, 'lambda_l1'- 0, 'lambda_l2'- 0}": {56: (0.4243610609071705, 0.0003220739734068889), 60: (0.4241938333486882, 0.00044856503608534104), 64: (0.42389721230957045, 0.00012092804011873404)}}, 'LgbTrainer-max_depth': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'num_leaves'- 64, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'feature_fraction'- 0.9, 'lambda_l1'- 0, 'lambda_l2'- 0}": {11: (0.4257390879967675, 0.0006422866920184622), 12: (0.4246789618409418, 0.0006315935416283114), 13: (0.424159160857272, 0.0006530573680005036)}}, 'LgbTrainer-bagging_fraction': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_freq'- 1, 'feature_fraction'- 0.9, 'lambda_l1'- 0, 'lambda_l2'- 0}": {0.8: (0.42417495421635476, 0.0001872781570237403), 0.9: (0.42385412724007565, 2.0527364912370505e-05), 1.0: (0.42476646803241863, 9.510713795479742e-05)}}, 'LgbTrainer-bagging_freq': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'feature_fraction'- 0.9, 'lambda_l1'- 0, 'lambda_l2'- 0}": {1: (0.42399735607082634, 0.00027509578637527343)}}, 'LgbTrainer-feature_fraction': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'lambda_l1'- 0, 'lambda_l2'- 0}": {0.8: (0.4242678279828749, 1.499764146498106e-05), 0.9: (0.4243530908297009, 0.0002342838982892781), 1.0: (0.4242279490534157, 9.352662910044884e-05)}}, 'LgbTrainer-lambda_l1': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'feature_fraction'- 0.8, 'lambda_l2'- 0}": {0.0: (0.423974386229127, 0.0002285715967729196), 0.4: (0.4236787280042236, 0.00017822919601950815), 0.8: (0.4235600805746734, 0.0002505222408797636), 2.0: (0.4237444073633829, 0.00023569570435114096)}}, 'LgbTrainer-lambda_l2': {"{'learning_rate'- 0.1703125, 'alpha'- 0.9, 'num_leaves'- 64, 'max_depth'- 13, 'bagging_fraction'- 0.9, 'bagging_freq'- 1, 'feature_fraction'- 0.8, 'lambda_l1'- 0.8}": {0.0: (0.42346338866450839, 0.00024325305753644666)}}}


def run():
    data_dir = '../input'
    submission, y, x, test_x, nn_x, nn_test_x, len_dic = get_data()
    del submission, test_x, nn_x, nn_test_x, len_dic
    gc.collect()

    num_boost_round = 3000
    origin_params = {'objective': 'huber', 'metric': 'rmse', 'verbose': -1, 'nthread': 5, 'alpha': 0.9,
                     'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': 0, 'bagging_fraction': 0.8, 'bagging_freq': 1,
                     'feature_fraction': 0.8}
    lgb_model = LgbTrainer(n_round=num_boost_round, params=origin_params)

    # init_param = [('learning_rate', 0.15625), ('num_leaves', 64), ('max_depth', 13), ('bagging_fraction', 0.9), ('bagging_freq', 1),
    #             ('feature_fraction', 0.9), ('lambda_l1', 2.0), ('lambda_l2', 0.01)]
    init_param = [('learning_rate', 0.15625), ('alpha', 0.9), ('num_leaves', 64), ('max_depth', 13), ('bagging_fraction', 0.9), ('bagging_freq', 1),
                ('feature_fraction', 0.9), ('lambda_l1', 0), ('lambda_l2', 0)]
    param_dic = {'learning_rate': [.1, .2],
                 'alpha': [.2, .4, .8],
                 'num_leaves': ['grid', 56, 60, 64],
                 'max_depth': ['grid', 11, 12, 13],
                 'min_data': [16, 32, 64],
                 'bagging_fraction': ['grid', .8, .9, 1.0],
                 'bagging_freq': ['grid', 1],
                 'feature_fraction': ['grid', .8, .9, 1.0],
                 'lambda_l1': [0.0, .4, .8, 2.0],
                 'lambda_l2': [0.0, .01, .02, .04]}

    tune(lgb_model, x, y, init_param, param_dic, measure_func=measure_handler, detail=True, kc=(2, 1),
         random_state=3000, max_optimization=False, score_min_gain=1e-3, factor_cache=param_cache)


if __name__ == '__main__':
    run()