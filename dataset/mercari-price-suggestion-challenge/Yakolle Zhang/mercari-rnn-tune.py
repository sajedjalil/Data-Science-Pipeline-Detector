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

from keras.initializers import glorot_normal
from keras.layers import Input, Dense, concatenate, Embedding, Flatten, GRU
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
from scipy.sparse import hstack, vstack
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

max_len_dic = {'item_description': 65, 'name': 10}
max_data_size = 800000

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


def measure_handler(target, pred):
    return metrics.mean_squared_error(target, pred) ** 0.5


class RNNTrainer:
    def get_rnn_model(self, tx, epochs=2):
        name_seq = Input(shape=[tx['name_seq'].shape[1]], name='name_seq')
        desc_seq = Input(shape=[tx['item_description_seq'].shape[1]], name='item_description_seq')
        brand = Input(shape=[1], name='brand_name_ind')
        cat = Input(shape=[1], name='category_name_ind')
        cond = Input(shape=[tx['item_condition_id'].shape[1]], name='item_condition_id')
        ship = Input(shape=[tx['shipping'].shape[1]], name='shipping')
        # cnts = Input(shape=[tx['cnts'].shape[1]], name='cnts')

        emb_name_seq = Embedding(self.len_dic['name_seq'], 20)(name_seq)
        emb_desc_seq = Embedding(self.len_dic['item_description_seq'], 25)(desc_seq)
        emb_brand = Embedding(self.len_dic['brand_name_ind'], 10)(brand)
        emb_cat = Embedding(self.len_dic['category_name_ind'], 10)(cat)

        rnn_name_seq = GRU(8)(emb_name_seq)
        rnn_desc_seq = GRU(10)(emb_desc_seq)

        flat = concatenate([rnn_name_seq, rnn_desc_seq, Flatten()(emb_brand), Flatten()(emb_cat), cond, ship])

        flat = Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=540))(flat)
        flat = Dense(64, activation='relu', kernel_initializer=glorot_normal(seed=541))(flat)
        output = Dense(1, activation='relu')(flat)

        rnn = Model([name_seq, desc_seq, brand, cat, cond, ship], output)
        steps = int(tx['shipping'].shape[0] / self.batch_size) * epochs
        decay = (self.lr / self.lr_fin - 1) / steps
        rnn.compile(loss='mse', optimizer=Adam(lr=self.lr, decay=decay))
        return rnn

    def __init__(self, lr, lr_fin, batch_size, len_dic, seed):
        self.lr = lr
        self.lr_fin = lr_fin
        self.batch_size = batch_size
        self.len_dic = len_dic
        self.seed = seed
        self.model = None

    def set_params(self, **params):
        if 'lr' in params:
            self.lr = params['lr']
        if 'lr_fin' in params:
            self.lr_fin = params['lr_fin']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'seed' in params:
            self.seed = params['seed']

    def fit(self, x, y, epochs=2):
        np.random.seed(self.seed)

        self.model = self.get_rnn_model(x, epochs)
        self.model.fit(x, y, epochs=epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, x):
        return self.model.predict(x, batch_size=70000)


param_cache = {'RNNTrainer-lr': {"{'lr_fin'- 0.002, 'batch_size'- 4096}": {0.002: (0.4923308801624199, 0.0002901905519820114), 0.004: (0.490692267622896, 0.0007090625398504802), 0.007: (0.4905193315242964, 0.0001333918903389797), 0.008: (0.4910183201070959, 0.00035465260148645483), 0.01: (0.493979963848848, 0.0004038156990377739), 0.003: (0.4901190195826883, 0.0005171958839255697), 0.005: (0.4911195716661978, 0.0007196443275195585), 0.006: (0.4878928697045093, 0.002120509416880495), 0.00525: (0.49085542834839546, 0.0007376655698223267), 0.0055: (0.4896861128767863, 0.0008840388312164604), 0.00575: (0.489184402273256, 0.0007029434207432572), 0.00625: (0.484860773132946, 0.004912762254583758), 0.0065: (0.48996368265541046, 0.0003847636779255592), 0.00675: (0.49077560348539967, 0.0007522730610152184), 0.0060625: (0.4893929678594224, 0.0003988355179838088), 0.006125: (0.4893245984519724, 0.0003976074327927126), 0.0061875: (0.4893547826990467, 0.0004877046874587343), 0.0063125: (0.489545407747763, 0.0004958497599018841), 0.006375: (0.489663980813236, 0.0003693192969385639), 0.0064375: (0.48973552824857736, 0.0005397065258932743), 0.00609375: (0.48934832596934646, 0.0004020856604545764), 0.00615625: (0.4894188484642849, 0.0005146633125753808), 0.00621875: (0.48940489340813764, 0.00043664009900512557), 0.00628125: (0.4895330225123385, 0.00044058672851793323)}, "{'lr_fin'- 0.0055, 'batch_size'- 4096}": {0.002: (0.49217103607572504, 0.00013034646528714555), 0.004: (0.47274065777667273, 6.244188536119744e-05), 0.006125: (0.4689257420245654, 0.00026540299265739864), 0.007: (0.4702150194503759, 0.0008719244102404489), 0.008: (0.4684053368011687, 0.00018927772845225843), 0.01: (0.4748815618320771, 1.0454389078956572e-05), 0.0075: (0.4692241345218966, 0.00010209226719290476), 0.0085: (0.4682691261907437, 0.0011815359598793818), 0.009: (0.4707123852591124, 0.00029288976266542144), 0.0095: (0.47326533614188104, 0.000205333227380311)}, "{'lr_fin'- 0.00725, 'batch_size'- 4096}": {0.002: (0.49217103607572504, 0.00013034646528714555), 0.004: (0.47274065777667273, 6.244188536119744e-05), 0.006125: (0.4682577085468676, 0.0003817526135779792), 0.007: (0.4688461032789638, 0.00028663374846757406), 0.008: (0.4670962500706004, 0.0002496779899096857), 0.01: (0.4706888432134023, 0.0007636901911588778), 0.0075: (0.4675174441724066, 0.0003100300407117562), 0.0085: (0.466074644998247, 0.00023274454863522753), 0.009: (0.4687158180707591, 0.0004899319688113357), 0.0095: (0.4722540732165874, 0.0007286425383197792), 0.008125: (0.46595276318930334, 0.0012691470479447853), 0.00825: (0.4666172736958992, 0.0003365473554153564), 0.008375: (0.46769090835500327, 0.00045613245907386024), 0.008625: (0.46854888359464, 0.0010952576511133738), 0.00875: (0.46592963544999155, 0.0021474695558514056), 0.008875: (0.46788141570452424, 0.000182974446262113)}}, 'RNNTrainer-lr_fin': {"{'lr'- 0.006125, 'batch_size'- 4096}": {0.001: (0.4958226799001243, 0.0004095628894873471), 0.002: (0.48955011852414143, 0.0007139229177283436), 0.004: (0.46934779055230763, 0.00020930991776679786), 0.0025: (0.47866692488436136, 0.0008614606482017229), 0.003: (0.47393066647458826, 0.00033427462947685505), 0.0035: (0.4703174250200496, 0.00019138858447312335), 0.0045: (0.4691201271492118, 5.331170128150542e-05), 0.005: (0.46824916092103097, 0.0005095392468291715), 0.0055: (0.4669597474562735, 0.0001301899311608734), 0.006: (0.4675786367471061, 0.00025203000588300273), 0.005125: (0.46815899826897944, 0.00023242260709632756), 0.00525: (0.46798765809251713, 0.0005193804004149927), 0.005375: (0.46776226731229587, 0.0006052120839137587), 0.005625: (0.46754715254205004, 0.0005753256788322003), 0.00575: (0.4671943857424433, 0.0006697812167607287), 0.005875: (0.4668053484976241, 0.0007211753853962888)}, "{'lr'- 0.008, 'batch_size'- 4096}": {0.001: (0.5021776922151178, 0.0009298067601469784), 0.002: (0.48235932548482086, 0.002163477075357051), 0.004: (0.4679352732535516, 0.0006746390587660245), 0.0055: (0.4669072230766401, 0.001676169890279966), 0.006: (0.4664068000453412, 0.001435051110200175), 0.00575: (0.4665170126555932, 0.0015538941742075651), 0.00625: (0.4655835231351363, 0.0013379022557370168), 0.0065: (0.46570518217895074, 0.0010360092866193715), 0.00675: (0.4655558705179486, 0.0007537565816710945), 0.007: (0.465343363551601, 0.0010536055020860802), 0.00725: (0.4651477721413223, 0.0010076654453422818), 0.0075: (0.4652083871141721, 0.001673836479367724), 0.00775: (0.46650791034984695, 0.0019460545434370757), 0.008: (0.4662993495965487, 0.002047846194616121), 0.00825: (0.4662993495965487, 0.002047846194616121), 0.0085: (0.4662993495965487, 0.002047846194616121), 0.00875: (0.4662993495965487, 0.002047846194616121), 0.009: (0.4662993495965487, 0.002047846194616121), 0.007125: (0.4659285114651194, 0.0015062933373956866), 0.007375: (0.46570743194507846, 0.0014723178315733387), 0.007625: (0.4655049495728496, 0.0018419918271019298)}, "{'lr'- 0.0085, 'batch_size'- 4096}": {0.001: (0.5041633406183621, 0.0003491573876203691), 0.002: (0.48925038498058726, 0.0025706183794843573), 0.004: (0.47129869982757155, 0.00068659571564103072), 0.0055: (0.46946552629458621, 0.0014256804104691434), 0.006: (0.46929121414015662, 0.0013859971552994121), 0.00725: (0.46690840409875767, 0.00043647360019521964), 0.009: (0.46796950108516211, 0.0017192822244817274), 0.006625: (0.46733553997702565, 6.4758410824317103e-05)}}, 'RNNTrainer-batch_size': {"{'lr'- 0.006125, 'lr_fin'- 0.0055}": {4096: (0.4697434787203686, 0.00022649789480411187), 4224: (0.47045651430605007, 0.000667342881422589), 4352: (0.4711925502278118, 1.1266850870633725e-05), 4480: (0.47110025971888425, 0.0004998007623235601), 4608: (0.4719549481203076, 0.00029157675185662035), 4736: (0.4718897358795111, 0.00019928855429238168), 4864: (0.4736948077529995, 0.0006801598088457816), 4992: (0.47438888286334535, 0.0009264922618247851), 5120: (0.47485916942164014, 0.0005552532032016555), 5248: (0.4762872706516684, 0.0006947807178579735), 5376: (0.47594570187552643, 0.0008580258193789103), 5504: (0.47648680568948343, 0.0001928221141775377), 5632: (0.4770035982873949, 0.0025164768295452833), 5760: (0.4771107539253266, 0.0028758343681153375), 5888: (0.47670242664892515, 0.0004671058973486908), 6016: (0.47748192011692486, 0.0007622780525889383), 6144: (0.4780226683765799, 0.003114921012147609), 6272: (0.4807786349023395, 0.0018334674243093152), 6400: (0.4851708013214765, 0.0023074291825970483), 6528: (0.48676954349360024, 0.0027062439176978315), 6656: (0.4876619488916999, 0.0028639903296809044), 6784: (0.4888683641189163, 0.0011754742238735127), 6912: (0.4911127670699073, 0.0017549172471556518), 7040: (0.491382910834512, 0.0016269985661599429), 7168: (0.4934488973309896, 0.0006230659208145695), 7296: (0.49404549445130075, 0.000426904587441157), 7424: (0.4944784778436361, 0.0006743690578238648), 7552: (0.49511736866347145, 0.0007521241154874181), 7680: (0.4928864851393253, 0.0033905508838486254), 7808: (0.4960107101966724, 0.0008901956841131653), 7936: (0.49667380500611114, 0.0010816240536770383), 8064: (0.49651277195325905, 0.0007193458628451666)}}}


def run():
    data_dir = '../input'
    submission, y, x, test_x, nn_x, nn_test_x, len_dic = get_data()
    nn_y = y
    del submission, y, x, test_x, nn_test_x
    gc.collect()

    rnn_model = RNNTrainer(lr=7e-3, lr_fin=1.9e-3, batch_size=512, len_dic=len_dic, seed=0)

    init_param = [('lr', 7e-3), ('lr_fin', 2e-3), ('batch_size', 4096)]
    param_dic = {'lr': [2e-3, 4e-3, 8e-3, 1e-2],
                 'lr_fin': [1e-3, 2e-3, 4e-3],
                 'batch_size': ['grid'] + list(128 * np.arange(32, 64)),
                 'seed': ['grid', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}

    tune(rnn_model, nn_x, nn_y, init_param, param_dic, measure_func=measure_handler, detail=True, kc=(2, 1),
         random_state=5000, data_dir=data_dir, max_optimization=False, score_min_gain=1e-3)


if __name__ == '__main__':
    run()