# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn import ensemble
from sklearn.externals import joblib
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance
from matplotlib import pyplot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

data_path = '../input'
# data_path = 'data'
out_path = 'out/pre_3'
submission_path = '../input'
ver = 6


# Any results you write to the current directory are saved as output.

def unreasonable_data(data):
    print("----------Reasonable of Data----------")
    print("Min Value:", data.min())
    print("Max Value:", data.max())
    print("Average Value:", data.mean())
    print("Center Point of Data:", data.median())
    print(data.describe())


def drop_duplicate(data, sub_set):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(sub_set, keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before - after)


def pre_process_data_3():
    sales = pd.read_csv('%s/sales_train.csv' % data_path, parse_dates=['date'], infer_datetime_format=True,
                        dayfirst=True)
    val = pd.read_csv('%s/test.csv' % data_path)
    subset = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
    drop_duplicate(sales, sub_set=subset)
    drop_duplicate(val, sub_set=['shop_id', 'item_id'])

    # unreasonable_data(sales['item_cnt_day'])
    # unreasonable_data(sales['item_price'])

    median = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) & (
            sales.item_price > 0)].item_price.median()
    sales.loc[sales.item_price < 0, 'item_price'] = median
    sales['item_cnt_day'] = sales['item_cnt_day'].clip(0, 1000)
    # sales['item_cnt_day'] = sales['item_cnt_day'].clip(0, 20)
    sales['item_price'] = sales['item_price'].clip(0, 300000)

    # =======================
    # From https://www.kaggle.com/dlarionov/feature-engineering-xgb/notebook
    # Якутск Орджоникидзе, 56
    sales.loc[sales.shop_id == 0, 'shop_id'] = 57
    val.loc[val.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    sales.loc[sales.shop_id == 1, 'shop_id'] = 58
    val.loc[val.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    sales.loc[sales.shop_id == 10, 'shop_id'] = 11
    val.loc[val.shop_id == 10, 'shop_id'] = 11
    # =======================

    # Rearrange the raw data to be monthly sales by item-shop
    df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().clip(0,
                                                                                                           20).reset_index()
    df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
    df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
    data = pd.merge(val, df, on=['item_id', 'shop_id'], how='left').fillna(0)
    data['item_id'] = np.log1p(data['item_id'])
    return data


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def train_model(x_train, y_train):
    reg = ensemble.ExtraTreesRegressor(n_estimators=512, max_depth=20,
                                       random_state=50)
    reg.fit(x_train, y_train)
    y_pre = reg.predict(x_train)
    score = np.sqrt(mean_squared_error(y_train, y_pre))
    print('RMSE cliped:', np.sqrt(mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    return reg


def linear_model(x_train, y_train):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pre = lr.predict(x_train)
    print('RMSE cliped:', np.sqrt(mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    return lr


def xgb_model(x_train, y_train, x_train_val, y_train_val):
    model = XGBRegressor(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.9,
        subsample=0.9,
        eta=0.15,
        seed=42)
    model.fit(
        x_train,
        y_train,
        eval_metric="rmse",
        eval_set=[(x_train, y_train), (x_train_val, y_train_val)],
        verbose=True,
        early_stopping_rounds=10)
    y_pre = model.predict(x_train)
    print('RMSE cliped:', np.sqrt(mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    plot_importance(model)
    pyplot.show()
    return model


def light_gbm_model(x_train, y_train):
    lgb_params = {
        'feature_fraction': 1,
        'metric': 'rmse',
        'min_data_in_leaf': 16,
        'bagging_fraction': 0.85,
        'learning_rate': 0.03,
        'objective': 'mse',
        'bagging_seed': 2 ** 7,
        'num_leaves': 32,
        'bagging_freq': 3,
        'verbose': 0
    }
    estimator = lgb.train(lgb_params, lgb.Dataset(x_train, label=y_train), 300)
    y_pre = estimator.predict(x_train)
    print('RMSE cliped:', np.sqrt(mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    return estimator


def pre_data(data_type, reg, x_test):
    if reg is None:
        reg = joblib.load('%s/%s_model_weight.model' % (out_path, data_type))
    y_pre = reg.predict(x_test)
    return y_pre


test = pre_process_data_3()
test_date_info = test.drop(labels=['ID'], axis=1)

y_train_normal = test_date_info['2015-10']
x_train_normal = test_date_info.drop(labels=['2015-10'], axis=1)
x_train_normal.columns = np.append(['shop_id', 'item_id'],
                                   np.arange(0, 33, 1))
x_train_val = x_train_normal[-100:]
y_train_val = y_train_normal[-100:]
xgb_model = xgb_model(x_train_normal[:-100], y_train_normal[:-100], x_train_val, y_train_val)
# linear_model = linear_model(x_train_normal, y_train_normal)
# light_gbm_model = light_gbm_model(x_train_normal, y_train_normal)
# normal_model = train_model(x_train_normal, y_train_normal)

test_x = test_date_info.drop(labels=['2013-01'], axis=1)
test_x.columns = np.append(['shop_id', 'item_id'],
                           np.arange(0, 33, 1))
# test_y_1 = pre_data('normal', normal_model, test_x)
# test_y_2 = pre_data('light_gbm', light_gbm_model, test_x)
# test_y_3 = pre_data('linear', linear_model, test_x)
test_y_4 = pre_data('xgb', xgb_model, test_x)
test_y = test_y_4

test['item_cnt_month'] = test_y
test['item_cnt_month'] = test['item_cnt_month'].clip(0, 20)
test[['ID', 'item_cnt_month']].to_csv('ver_%d.csv' % ver, index=False)
