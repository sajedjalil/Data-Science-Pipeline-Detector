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

from keras.layers import Input, Dropout, Dense, concatenate, Embedding, Flatten, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import copy
import gc
import os
import string
import warnings
from multiprocessing import current_process, Process, Queue
from time import time

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold

import inspect
import re
import threading

from pandas import Series

warnings.filterwarnings('ignore')


# -------------------------------------------util---------------------------------------
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
        train_x = {}
        test_x = {}
        for col_name, col in x.items():
            train_x[col_name] = col[train_index]
            test_x[col_name] = col[test_index]
        train_y = y[train_index]
        test_y = y[test_index]

        learning_model.fit(train_x, train_y)
        test_p = learning_model.predict(test_x)
        local_cv_scores.append(measure_func(test_y, test_p))

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
                               fit_params=None, group_bounds=None):
    if data_dir is not None:
        score_cache = read_cache(learning_model, factor_key, data_dir, factor_table)
    else:
        score_cache = {}

    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num

    cv_score_means = []
    cv_score_stds = []
    last_time = int(time())
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
                cv_score_std = large_num

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

        if data_dir is not None:
            cur_time = int(time())
            if cur_time - last_time >= 300:
                last_time = cur_time
                write_cache(learning_model, factor_key, score_cache, data_dir, factor_table)
                print(param_cache)

    if data_dir is not None:
        write_cache(learning_model, factor_key, score_cache, data_dir, factor_table)

    best_factor_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                              max_optimization=max_optimization)
    best_factor = factor_values[best_factor_index]
    print('--best factor: ', factor_key, '=', best_factor, ', mean=', cv_score_means[best_factor_index], ', std=',
          cv_score_stds[best_factor_index])

    return best_factor, cv_score_means[best_factor_index], cv_score_stds[best_factor_index]


def read_cache(model, factor_key, data_dir, factor_table):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    factor_cache_key = type(model).__name__ + '-' + factor_key
    if factor_cache_key in param_cache:
        cache = param_cache[factor_cache_key]
        cache_key = round_float_str(str(factor_table).replace(':', '-'))
        if cache_key in cache:
            return cache[cache_key]

    return {}


def write_cache(model, factor_key, cache_map, data_dir, factor_table):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    if cache_map:
        cache = {}
        factor_cache_key = type(model).__name__ + '-' + factor_key
        if factor_cache_key in param_cache:
            cache = param_cache[factor_cache_key]

        cache_key = round_float_str(str(factor_table).replace(':', '-'))
        if cache_key in cache:
            cache[cache_key].update(cache_map)
        else:
            cache[cache_key] = cache_map
        param_cache[factor_cache_key] = cache


def probe_best_factor(learning_model, x, y, factor_key, factor_values, get_next_elements, factor_table, detail=False,
                      cv_repeat_times=1, kc=None, score_min_gain=1e-4, measure_func=metrics.accuracy_score,
                      balance_mode=None, random_state=0, mean_std_coeff=(1.0, 1.0), max_optimization=True, nthread=1,
                      data_dir=None, inlier_indices=None, holdout_data=None, fit_params=None, group_bounds=None):
    int_flag = all([isinstance(ele, int) for ele in factor_values])
    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num
    last_best_score = bad_score

    if data_dir is not None:
        score_cache = read_cache(learning_model, factor_key, data_dir, factor_table)
    else:
        score_cache = {}

    last_time = int(time())
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
                    cv_score_std = large_num

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

            if data_dir is not None:
                cur_time = int(time())
                if cur_time - last_time >= 300:
                    last_time = cur_time
                    write_cache(learning_model, factor_key, score_cache, data_dir, factor_table)
                    print(param_cache)

        if data_dir is not None:
            write_cache(learning_model, factor_key, score_cache, data_dir, factor_table)

        best_factor_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                                  max_optimization=max_optimization)
        if abs(cv_score_means[best_factor_index] - last_best_score) < score_min_gain:
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
         random_state=0, detail=True, kc=None, inlier_indices=None, holdout_data=None, nthread=1, group_bounds=None):
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
                fit_params=fit_params, group_bounds=group_bounds)


def tune_factor(model, x, y, init_factor, factor_dic, get_next_elements, update_factors, cv_repeat_times=1, kc=None,
                measure_func=metrics.accuracy_score, balance_mode=None, max_optimization=True, score_min_gain=1e-4,
                mean_std_coeff=(1.0, 1.0), data_dir=None, random_state=0, detail=True, inlier_indices=None,
                holdout_data=None, nthread=1, fit_params=None, group_bounds=None):
    optional_factor_dic = {'measure_func': measure_func, 'cv_repeat_times': cv_repeat_times, 'detail': detail,
                           'max_optimization': max_optimization, 'kc': kc, 'inlier_indices': inlier_indices,
                           'mean_std_coeff': mean_std_coeff, 'score_min_gain': score_min_gain, 'data_dir': data_dir,
                           'holdout_data': holdout_data, 'balance_mode': balance_mode, 'nthread': nthread,
                           'fit_params': fit_params, 'group_bounds': group_bounds}

    best_factors = init_factor
    seed_dict = {}
    for i, (factor_key, factor_val) in enumerate(best_factors):
        seed_dict[factor_key] = random_state + i
        factor_values = factor_dic[factor_key]
        if factor_val not in factor_values:
            factor_values.append(factor_val)
            factor_dic[factor_key] = sorted(factor_values)
    last_best_factors = backup(best_factors)

    tmp_hold_factors = []
    last_best_score = 1e10
    cur_best_score = last_best_score
    while True:
        update_factors(best_factors)

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

name_terms = ['bundle', 'for']
desc_terms = ['.', ',', 'and']


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


def embed_target_agg(col, statistical_size=30, k=5, random_state=0):
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


def desc_cnt_thread(tr_desc_col, ts_desc_col, desc_queue):
    col_name = 'item_description'
    rec_time('%s encode_text' % col_name)
    tr_desc_cnts = encode_text(tr_desc_col)
    ts_desc_cnts = encode_text(ts_desc_col)
    print_info('%s encode_text' % col_name)

    desc_queue.put((tr_desc_cnts, ts_desc_cnts))


def desc_vectorize_thread(tr_desc_col, tr_brand_col, desc_queue):
    col_name = 'item_description'
    rec_time('train %s process' % col_name)
    tr_col = tr_desc_col.str.replace(r'\s\s+', ' ')
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 3), vocabulary=desc_terms, dtype=np.uint8)
    tr_x_desc = vr.fit_transform(tr_brand_col + ' ' + tr_col)
    print_info('train %s vectorize' % col_name, 'desc vocabulary size is %d.' % len(desc_terms))
    print_info('train %s process' % col_name)

    rec_time('test %s process' % col_name)
    ts_col = desc_queue.get()
    rec_time('test %s vectorize' % col_name)
    ts_x_desc = vr.transform(ts_col)
    print_info('test %s vectorize' % col_name)
    print_info('test %s process' % col_name)

    desc_queue.put((tr_x_desc, ts_x_desc))


def name_thread(tr_name_col, tr_brand_col, ts_name_col, ts_brand_col, name_queue):
    col_name = 'name'
    rec_time('train %s process' % col_name)
    tr_col = tr_name_col.str.replace(r'\s\s+', ' ')
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 3), vocabulary=name_terms, dtype=np.uint8)
    tr_x_name = vr.fit_transform(tr_brand_col + ' ' + tr_col)
    print_info('train %s vectorize' % col_name, 'name vocabulary size is %d.' % len(name_terms))
    rec_time('train %s encode_text' % col_name)
    tr_name_cnts = encode_text(tr_col)
    print_info('train %s encode_text' % col_name)
    print_info('train %s process' % col_name)

    rec_time('test %s process' % col_name)
    ts_col = ts_name_col.str.replace(r'\s\s+', ' ')
    rec_time('test %s vectorize' % col_name)
    ts_x_name = vr.transform(ts_brand_col + ' ' + ts_col)
    print_info('test %s vectorize' % col_name)
    rec_time('test %s encode_text' % col_name)
    ts_name_cnts = encode_text(ts_col)
    print_info('test %s encode_text' % col_name)
    print_info('test %s process' % col_name)

    name_queue.put((tr_x_name, tr_name_cnts, ts_x_name, ts_name_cnts))


def brand_thread(tr_brand_col, ts_brand_col):
    col_name = 'brand_name'
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern='.+', min_df=10, binary=True, dtype=np.uint8)
    tr_x_brand = vr.fit_transform(tr_brand_col)
    print_info('train %s vectorize' % col_name)
    rec_time('test %s vectorize' % col_name)
    ts_x_brand = vr.transform(ts_brand_col)
    print_info('test %s vectorize' % col_name)

    return tr_x_brand, ts_x_brand


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


def cat_thread(tr_cat1, tr_cat2, tr_cat3, tr_cat_n, ts_cat1, ts_cat2, ts_cat3, ts_cat_n, cat_queue):
    rec_time('category_name process')
    col_name = 'cat1'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat1 = vr.fit_transform(tr_cat1)
    ts_x_cat1 = vr.transform(ts_cat1)
    print_info('%s vectorize' % col_name)
    col_name = 'cat2'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat2 = vr.fit_transform(tr_cat2)
    ts_x_cat2 = vr.transform(ts_cat2)
    print_info('%s vectorize' % col_name)
    col_name = 'cat3'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat3 = vr.fit_transform(tr_cat3)
    ts_x_cat3 = vr.transform(ts_cat3)
    print_info('%s vectorize' % col_name)
    col_name = 'cat_n'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat_n = vr.fit_transform(tr_cat_n)
    ts_x_cat_n = vr.transform(ts_cat_n)
    print_info('%s vectorize' % col_name)
    rec_time('combine cats')
    tr_x_cats = hstack((tr_x_cat1, tr_x_cat2, tr_x_cat3, tr_x_cat_n))
    ts_x_cats = hstack((ts_x_cat1, ts_x_cat2, ts_x_cat3, ts_x_cat_n))
    print_info('combine cats')
    print_info('category_name process')

    cat_queue.put((tr_x_cats, ts_x_cats))


def embed_thread(train_df, test_df, key_col_name, extra_cols, random_state=5000):
    rec_time('%s embed target' % key_col_name)
    cols = extra_cols + [key_col_name]
    gp_embed = train_df[cols + ['price']].groupby(cols).agg(embed_target_agg, **{'random_state': random_state})
    embed_col_suffix = '_' + key_col_name + '_embed'
    tr_embed = np.array(list(train_df.join(gp_embed, on=cols, rsuffix=embed_col_suffix)['price' + embed_col_suffix]))
    ts_embed = test_df.join(gp_embed, on=cols)['price']
    ts_embed.loc[ts_embed.isnull()] = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] * np.sum(ts_embed.isnull())
    ts_embed = np.array(list(ts_embed))
    print_info('%s embed target' % key_col_name)

    return tr_embed, ts_embed


def item_cond_thread(tr_cond_col, ts_cond_col):
    col_name = 'item_condition_id'
    rec_time('%s process' % col_name)
    tr_x_cond = pd.get_dummies(tr_cond_col).values
    ts_cond_col = ts_cond_col.clip(1, 5)
    ts_x_cond = pd.get_dummies(ts_cond_col).values
    print_info('%s process' % col_name)
    return tr_x_cond, ts_x_cond


def ship_thread(tr_ship_col, ts_ship_col):
    col_name = 'shipping'
    rec_time('%s process' % col_name)
    tr_x_ship = pd.get_dummies(tr_ship_col).values
    ts_x_ship = pd.get_dummies(ts_ship_col).values
    print_info('%s process' % col_name)
    return tr_x_ship, ts_x_ship


def read_data(data_dir='../input'):
    def fillna(df):
        df['brand_name'].fillna('missing', inplace=True)
        df['item_description'].fillna('None', inplace=True)
        df['category_name'].fillna('Other', inplace=True)
        df['name'].fillna('Unk.', inplace=True)
        df.fillna(0, inplace=True)

    rec_time('read data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', engine='c')
    train_df['price'] = np.log1p(train_df.price)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', engine='c')
    print_info('read data', '%s %s' % (train_df.shape, test_df.shape))

    rec_time('remove item with 0 price')
    train_df = train_df.loc[train_df.price > 1].reset_index(drop=True)
    print_info('remove item with 0 price', train_df.shape)

    rec_time('fillna')
    fillna(train_df)
    fillna(test_df)
    print_info('fillna', '%s %s' % (train_df.shape, test_df.shape))

    return train_df, test_df


def get_lgb_data(train_df, test_df, share_queue):
    col_name = 'item_description'
    rec_time('train %s normalize' % col_name)
    train_df.item_description.replace('No description yet', 'None', inplace=True)
    print_info('train %s normalize' % col_name)

    desc_queue = Queue()
    desc_worker = get_worker(desc_vectorize_thread, name='desc_vectorize_thread',
                             args=(train_df.item_description, train_df.brand_name, desc_queue))
    desc_worker.start()

    other_queue = Queue()
    other_worker = get_worker(name_thread, name='name_thread',
                              args=(train_df['name'], train_df.brand_name, test_df['name'], test_df.brand_name,
                                    other_queue))
    other_worker.start()

    col_name = 'item_description'
    rec_time('test %s normalize' % col_name)
    test_df.item_description.replace('No description yet', 'None', inplace=True)
    desc_queue.put(test_df.brand_name + ' ' + test_df.item_description.str.replace(r'\s\s+', ' '))
    print_info('test %s normalize' % col_name)

    (tr_cat_cnts, ts_cat_cnts), (tr_cat1, tr_cat2, tr_cat3, train_df['cat_entity'], tr_cat_n), (
        ts_cat1, ts_cat2, ts_cat3, test_df['cat_entity'], ts_cat_n) = cat_extract_thread(train_df.category_name,
                                                                                         test_df.category_name)

    tr_x_cat_embeds, ts_x_cat_embeds = embed_thread(train_df, test_df, 'cat_entity', ['item_condition_id', 'shipping'],
                                                    random_state=5000)
    tr_x_brand_embeds, ts_x_brand_embeds = embed_thread(train_df, test_df, 'brand_name',
                                                        ['item_condition_id', 'shipping'], random_state=5001)

    tr_x_cond, ts_x_cond = item_cond_thread(train_df.item_condition_id, test_df.item_condition_id)

    tr_x_name, tr_name_cnts, ts_x_name, ts_name_cnts = other_queue.get()
    other_worker.join()
    other_worker = get_worker(cat_thread, name='cat_thread',
                              args=(tr_cat1, tr_cat2, tr_cat3, tr_cat_n, ts_cat1, ts_cat2, ts_cat3, ts_cat_n,
                                    other_queue))
    other_worker.start()

    col_name = 'brand_name'
    rec_time('%s process' % col_name)
    cnts = train_df[col_name].append(test_df[col_name]).value_counts()
    train_df = train_df.join(cnts, on=col_name, rsuffix='_cnt')
    test_df = test_df.join(cnts, on=col_name, rsuffix='_cnt')
    train_df[col_name + '_len'] = train_df[col_name].str.len()
    test_df[col_name + '_len'] = test_df[col_name].str.len()
    print_info('%s process' % col_name)
    del cnts

    tr_x_ship, ts_x_ship = ship_thread(train_df.shipping, test_df.shipping)

    cols = ['brand_name_cnt', 'brand_name_len']
    rec_time('extract brand numeric feature')
    tr_x_brand_cnts = train_df[cols]
    ts_x_brand_cnts = test_df[cols]
    print_info('extract brand numeric feature', cols)
    tr_x_brand, ts_x_brand = brand_thread(train_df.brand_name, test_df.brand_name)

    share_queue.put((train_df.category_name, test_df.category_name, train_df['name'], test_df['name']))

    tr_x_cats, ts_x_cats = other_queue.get()
    other_worker.join()
    other_worker = get_worker(desc_cnt_thread, name='desc_cnt_thread',
                              args=(train_df.item_description, test_df.item_description, other_queue))
    other_worker.start()

    tr_desc_cnts, ts_desc_cnts = other_queue.get()
    other_queue.close()
    other_worker.join()
    del other_worker, other_queue
    gc.collect()

    tr_x_desc, ts_x_desc = desc_queue.get()
    desc_queue.close()
    desc_worker.join()
    del desc_worker, desc_queue
    gc.collect()

    rec_time('combine features')
    x_cnts = np.hstack([tr_x_brand_embeds, tr_x_brand_cnts, tr_x_cat_embeds, tr_cat_cnts, tr_name_cnts, tr_desc_cnts])
    ts_x_cnts = np.hstack([ts_x_brand_embeds, ts_x_brand_cnts, ts_x_cat_embeds, ts_cat_cnts, ts_name_cnts,
                           ts_desc_cnts])
    scaler = MinMaxScaler()
    scaled_x_cnts = scaler.fit_transform(x_cnts)
    scaled_ts_x_cnts = scaler.transform(ts_x_cnts)
    x = hstack([tr_x_ship, tr_x_cond, tr_x_brand, tr_x_cats, tr_x_name, tr_x_desc])
    ts_x = hstack([ts_x_ship, ts_x_cond, ts_x_brand, ts_x_cats, ts_x_name, ts_x_desc])
    print_info('combine features')

    return x, train_df.price.copy().values, ts_x, x_cnts, ts_x_cnts, scaled_x_cnts, scaled_ts_x_cnts


def get_nn_data(train_df, test_df, nn_queue):
    y = train_df.price.values
    x = {}
    ts_x = {}
    len_dic = {}

    cols = ['category_name', 'brand_name']
    for col_name in cols:
        rec_time('%s label encode' % col_name)
        ler = LabelEncoder()
        tr_col = train_df[col_name]
        ts_col = test_df[col_name]
        ler.fit(np.hstack([tr_col, ts_col]))
        ind_col_name = col_name + '_ind'
        x[ind_col_name] = ler.transform(tr_col)
        ts_x[ind_col_name] = ler.transform(ts_col)
        len_dic[ind_col_name] = ler.classes_.shape[0]
        print_info('%s label encode' % col_name)
        del tr_col, ts_col, ler

    def to_sequence(tr_column, ts_column):
        column_name = tr_column.name
        rec_time('%s tokenizer' % column_name)
        tkr = Tokenizer()
        tkr.fit_on_texts(tr_column)
        tr_column = np.array(tkr.texts_to_sequences(tr_column))
        ts_column = np.array(tkr.texts_to_sequences(ts_column))
        print_info('%s tokenizer' % column_name)
        return tr_column, ts_column, len(tkr.word_index) + 1

    def cat_name_sequence(share_queue):
        tr_cat_col, ts_cat_col, tr_name_col, ts_name_col = share_queue.get()
        share_queue.put(to_sequence(tr_cat_col, ts_cat_col))
        share_queue.put(to_sequence(tr_name_col, ts_name_col))

    max_len_dic = {'category_name_seq': 10, 'item_description_seq': 65, 'name_seq': 20}

    def process_sequence(column_name, tr_column, ts_column):
        rec_time('%s pad_sequences' % column_name)
        x[column_name] = pad_sequences(tr_column, maxlen=max_len_dic[column_name], truncating='post', padding='post')
        ts_x[column_name] = pad_sequences(ts_column, maxlen=max_len_dic[column_name], truncating='post', padding='post')
        print_info('%s pad_sequences' % column_name)

    sequence_worker = get_worker(cat_name_sequence, name='cat_name_sequence', args=(nn_queue,))
    sequence_worker.start()

    tr_x_desc, ts_x_desc, len_dic['item_description_seq'] = to_sequence(train_df.item_description,
                                                                        test_df.item_description)

    tr_x_cat, ts_x_cat, len_dic['category_name_seq'] = nn_queue.get()
    process_sequence('category_name_seq', tr_x_cat, ts_x_cat)
    del tr_x_cat, ts_x_cat
    gc.collect()

    process_sequence('item_description_seq', tr_x_desc, ts_x_desc)
    del tr_x_desc, ts_x_desc
    gc.collect()

    test_df['item_condition_id'] = test_df.item_condition_id.astype(np.uint8).clip(1, 5)

    cols = ['item_condition_id', 'shipping']
    for col_name in cols:
        rec_time('onehot %s' % col_name)
        x[col_name] = pd.get_dummies(train_df[col_name]).values
        ts_x[col_name] = pd.get_dummies(test_df[col_name]).values
        print_info('onehot %s' % col_name)

    tr_x_name, ts_x_name, len_dic['name_seq'] = nn_queue.get()
    process_sequence('name_seq', tr_x_name, ts_x_name)
    del tr_x_name, ts_x_name
    gc.collect()

    nn_queue.put((x, y, ts_x, len_dic))


def get_worker(func, args, name=None):
    worker = Process(target=func, args=args, name=name)
    return worker


def measure_handler(target, pred):
    return metrics.mean_squared_error(target, pred) ** 0.5


class CNNTrainer:
    def get_cnn_model(self, tx):
        name_seq = Input(shape=[tx['name_seq'].shape[1]], name='name_seq')
        desc_seq = Input(shape=[tx['item_description_seq'].shape[1]], name='item_description_seq')
        brand = Input(shape=[1], name='brand_name_ind')
        cat = Input(shape=[1], name='category_name_ind')
        cond = Input(shape=[tx['item_condition_id'].shape[1]], name='item_condition_id')
        ship = Input(shape=[tx['shipping'].shape[1]], name='shipping')
        cnts = Input(shape=[tx['cnts'].shape[1]], name='cnts')

        emb_name_seq = Embedding(self.len_dic['name_seq'], 20)(name_seq)
        emb_desc_seq = Embedding(self.len_dic['item_description_seq'], 25)(desc_seq)
        emb_brand = Embedding(self.len_dic['brand_name_ind'], 10)(brand)
        emb_cat = Embedding(self.len_dic['category_name_ind'], 10)(cat)

        cnn_name_seq = Flatten()(Conv1D(16, kernel_size=3, activation='relu')(emb_name_seq))
        cnn_desc_seq = Flatten()(Conv1D(20, kernel_size=3, activation='relu')(emb_desc_seq))

        cluster = concatenate([Flatten()(emb_brand), Flatten()(emb_cat), cond, ship])
        cluster = Dropout(0.2)(Dense(64, activation='relu')(cluster))

        flat = concatenate([cnn_name_seq, cnn_desc_seq, cluster, Flatten()(emb_brand), Flatten()(emb_cat),
                            cond, ship, cnts])

        flat = Dropout(0.25)(Dense(256, activation='relu')(flat))
        output = Dense(1, activation='relu')(flat)

        cnn = Model([name_seq, desc_seq, brand, cat, cond, ship, cnts], output)
        cnn.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return cnn

    def __init__(self, lr, batch_size, len_dic):
        self.lr = lr
        self.batch_size = batch_size
        self.len_dic = len_dic
        self.model = None

    def set_params(self, **params):
        if 'lr' in params:
            self.lr = params['lr']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']

    def fit(self, x, y):
        np.random.seed(10001)

        self.model = self.get_cnn_model(x)
        self.model.fit(x, y, epochs=2, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, x):
        return self.model.predict(x, batch_size=70000)


param_cache = {'CNNTrainer-lr': {"{'batch_size'- 4500}": {0.001: (0.45807475583586493, 0.0003961161432851442), 0.002: (0.4537851043253783, 0.0006510877742435312), 0.004: (0.45631588556868224, 0.0008746366814931695), 0.008: (0.45285420733016674, 0.001688124318790245), 0.009: (0.4577256568738349, 0.0008472916667529902), 0.01: (0.4542599606363605, 0.0008250561643224163), 0.0015: (0.4553525453554625, 0.0007223097073233652), 0.0025: (0.4530257962021747, 0.0006849271060432203), 0.003: (0.4555084181277304, 0.0006668763225220024), 0.0035: (0.4541009267600464, 0.000874331422235011), 0.0045: (0.4538851557815569, 0.0007593309156552609), 0.005: (0.45607567640412777, 0.00044521964594387804), 0.0055: (3.07273029178655, 0.0012535435352808895), 0.006: (3.07273029178655, 0.0012535435352808895), 0.0065: (3.07273029178655, 0.0012535435352808895), 0.007: (3.07273029178655, 0.0012535435352808895), 0.0075: (0.457702652495074, 0.0070317506450510705), 0.0085: (0.4556248788172284, 0.002627594460182187)}, "{'batch_size'- 750}": {0.001: (0.4569189692814611, 0.003523495970458538), 0.002: (0.46091831973195746, 0.010367997417095744), 0.0025: (0.46804921678656336, 0.013803827063609058), 0.004: (0.4897611881713428, 0.04385022538661613), 0.008: (3.07273029178655, 0.0012535435352808895), 0.009: (3.07273029178655, 0.0012535435352808895), 0.01: (3.07273029178655, 0.0012535435352808895), 0.0005: (0.45788057481165495, 0.006955658044461991), 0.00096875: (0.4591273504745755, 0.005220496807764637), 0.0014375: (0.45635593855113904, 0.007598862358964691), 0.00190625: (0.4597587720025045, 0.011665586967591153), 0.002375: (0.46269222588343234, 0.00787177370131946), 0.00284375: (0.47509830160301886, 0.020533151108439698), 0.0033125: (0.49727831732315364, 0.05145120306993353), 0.00378125: (0.5086832488613724, 0.07995067399795272), 0.00425: (0.48480255475240996, 0.041338316519567785), 0.00471875: (0.47718725962661407, 0.02223095824395283), 0.0051875: (0.49518554335098736, 0.03644135778673489), 0.00565625: (3.07273029178655, 0.0012535435352808895), 0.006125: (3.07273029178655, 0.0012535435352808895), 0.00659375: (3.07273029178655, 0.0012535435352808895), 0.0070625: (3.07273029178655, 0.0012535435352808895), 0.00753125: (3.07273029178655, 0.0012535435352808895)}}, 'CNNTrainer-batch_size': {"{'lr'- 0.0025}": {1000: (0.4507490961323735, 0.00057263083800068), 2000: (0.45089697405558965, 0.00012520224073370937), 3000: (0.45510785514996166, 0.004412495523522447), 4000: (0.4532403573769783, 0.0006130024718145545), 4500: (0.4543437534892043, 0.0021537143203683353), 5000: (0.4549975958886881, 0.001736014145914465), 6000: (0.45665157200156337, 0.0010903354472231958), 7000: (0.45812783156767645, 0.0023370026884816128), 8000: (0.4568458589188265, 0.0006973372059255521), 500: (0.45286692560818403, 0.005990035241529842), 750: (0.44792832823392353, 0.0004158374473212197), 1250: (0.4500211691368556, 0.0010428559083440017), 1500: (0.45618086121094154, 0.004805846212772535), 1750: (0.4554715968045824, 0.00665077159109395), 2250: (0.453684424021066, 0.0010252042103870169), 2500: (0.4596504435526945, 0.006809590825104781), 2750: (0.4546410731642067, 0.00119900202893726), 250: (0.45352511574609, 0.010831081877659717), 375: (0.44878013353488794, 0.004638566805512469), 625: (0.45804677293586976, 0.006626310611840112), 875: (0.45136870232597986, 0.004382264695414411)}, "{'lr'- 0.001}": {500: (0.4537673106180193, 0.002510508936199259), 750: (0.4514077382170983, 0.0010171154688484892), 1000: (0.45163124633844537, 0.0018147568417195049), 2000: (0.45367215756185769, 0.00093041749296632445), 3000: (0.4561892362695375, 0.0017949159809918418), 4000: (0.45785500300202314, 0.00023963453343228328)}}}


def run():
    data_dir = '../input'
    train_df, test_df = read_data()

    nn_queue = Queue()
    nn_worker = get_worker(func=get_nn_data, args=(train_df, test_df, nn_queue), name='get_nn_data')
    nn_worker.start()

    x, y, test_x, x_cnts, ts_x_cnts, scaled_x_cnts, scaled_ts_x_cnts = get_lgb_data(train_df, test_df, nn_queue)
    del train_df, test_df, x, y, test_x, x_cnts, ts_x_cnts
    gc.collect()

    nn_x, nn_y, nn_test_x, len_dic = nn_queue.get()
    nn_queue.close()
    nn_worker.join()
    nn_x['cnts'] = scaled_x_cnts
    nn_test_x['cnts'] = scaled_ts_x_cnts
    del nn_worker, nn_queue, nn_test_x, scaled_x_cnts, scaled_ts_x_cnts
    gc.collect()

    cnn_model = CNNTrainer(lr=9e-3, batch_size=4500, len_dic=len_dic)

    init_param = [('lr', 9e-3), ('batch_size', 4500)]
    param_dic = {'lr': [1e-3, 2e-3, 4e-3, 8e-3, 1e-2],
                 'batch_size': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]}

    tune(cnn_model, nn_x, nn_y, init_param, param_dic, measure_func=measure_handler, detail=True, kc=(3, 1),
         random_state=5000, data_dir=data_dir, max_optimization=False, score_min_gain=1e-3)


if __name__ == '__main__':
    run()