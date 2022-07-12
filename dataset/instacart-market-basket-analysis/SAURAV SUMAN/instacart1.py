# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime


DATA_DIR = "../input/"
PRIOR_FILE = "order_products__prior"
ORDERS_FILE = "orders"


def load_input_data():
    PATH = "{}{}{}".format(DATA_DIR, PRIOR_FILE, ".csv")
    prior = pd.read_csv(PATH, dtype={'order_id': np.int32,
                                     'product_id': np.uint16,
                                     'add_to_cart_order': np.int16,
                                     'reordered': np.int8})

    PATH = "{}{}{}".format(DATA_DIR, ORDERS_FILE, ".csv")
    orders = pd.read_csv(PATH, dtype={'order_id': np.int32,
                                      'user_id': np.int64,
                                      'order_number': np.int16,
                                      'order_dow': np.int8,
                                      'order_hour_of_day': np.int8,
                                      'days_since_prior_order': np.float32})
    return prior, orders


def apply_parallel(df_groups, _func):
    nthreads = multiprocessing.cpu_count()
    print("nthreads: {}".format(nthreads))

    res = Parallel(n_jobs=nthreads)(delayed(_func)(grp.copy()) for _, grp in df_groups)
    return pd.concat(res)


def add_order_streak(df):
    tmp = df.copy()
    tmp.user_id = 1

    UP = tmp.pivot(index="product_id", columns='order_number').fillna(-1)
    UP.columns = UP.columns.droplevel(0)

    x = np.abs(UP.diff(axis=1).fillna(0)).values[:, ::-1]
    df.set_index("product_id", inplace=True)
    df['order_streak'] = np.multiply(np.argmax(x, axis=1) + 1, UP.iloc[:, -1])
    df.reset_index(drop=False, inplace=True)
    return df


if __name__ == '__main__':
    prior, orders = load_input_data()

    print("orders: {}".format(orders.shape))
    print("take only recent 5 orders per user:")
    orders = orders.groupby(['user_id']).tail(5 + 1)
    print("orders: {}".format(orders.shape))

    prior = orders.merge(prior, how='inner', on="order_id")
    prior = prior[['user_id', 'product_id', 'order_number']]
    print("prior: {}".format(prior.shape))

    user_groups = prior.groupby('user_id')
    s = datetime.now()
    df = apply_parallel(user_groups, add_order_streak)
    e = datetime.now()
    print("time elapsed: {}".format(e - s))

    df.drop("order_number", axis=1).reset_index(drop=True)
    df = df[['user_id', 'product_id', 'order_streak']].drop_duplicates()
    print(df.head(n=10))
    df.to_csv("order_streaks.csv", index=False)
    print("order_streaks.csv has been written")