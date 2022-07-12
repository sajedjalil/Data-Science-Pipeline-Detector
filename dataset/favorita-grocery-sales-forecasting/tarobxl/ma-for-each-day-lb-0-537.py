# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# MA result from https://www.kaggle.com/paulorzp/log-means-and-medians-to-predict-new-itens-0-546/code
def read_ma():
    ma = pd.read_csv("../output/comb8.csv")
    ma.columns = ['id','unit_sales_ma']
    ma['unit_sales_ma'] = pd.np.log1p(ma['unit_sales_ma']) # logarithm conversion
    print("ma", ma.columns, ma.shape)
    print(ma.head(3))
    return ma


dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}

def prepare_train_data():
    print("----- Reading the data -----")

    train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                            skiprows=range(1, 101688779) #Skip dates before 2017-01-01
                            )

    print("----- Scale the data -----")
    train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
    train['unit_sales'] = pd.np.log1p(train['unit_sales']) #logarithm conversion

    print("----- Reindex the data -----")
    # creating records for all items, in all markets on all dates
    # for correct calculation of daily unit sales averages.
    u_dates = train.date.unique()
    u_stores = train.store_nbr.unique()
    u_items = train.item_nbr.unique()
    train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
    train = train.reindex(
        pd.MultiIndex.from_product(
            (u_dates, u_stores, u_items),
            names=["date", "store_nbr", "item_nbr"]
        )
    )

    del u_dates, u_stores, u_items

    print("----- Fill in missing values -----")
    # Fill NaNs
    train.loc[:, "unit_sales"].fillna(0, inplace=True)
    train.reset_index(inplace=True) # reset index and restoring unique columns
    lastdate = train.iloc[train.shape[0]-1].date

    print("train", train.columns, train.shape)
    print(train.head(3))

    return train, lastdate

def average_dow(train):
    # Average by dow
    train["dow"] = train["date"].dt.dayofweek
    train_mean_dow = train[['store_nbr', 'item_nbr', 'dow', 'unit_sales']].groupby(['store_nbr', 'item_nbr', 'dow']).mean()
    train_mean_dow.reset_index(inplace=True)
    train_mean_dow.columns = ['store_nbr', 'item_nbr', 'dow', 'unit_sales_dow']
    print("train_mean_dow", train_mean_dow.columns, train_mean_dow.shape)
    print(train_mean_dow.head(3))

    train_mean_week = train_mean_dow[['store_nbr', 'item_nbr', 'unit_sales_dow']].groupby(['store_nbr', 'item_nbr']).mean()
    train_mean_week.reset_index(inplace=True)
    train_mean_week.columns = ['store_nbr', 'item_nbr', 'unit_sales_week']
    print("train_mean_week", train_mean_week.columns, train_mean_week.shape)
    print(train_mean_week.head(3))

    train_mean = pd.merge(train_mean_dow, train_mean_week, on=['store_nbr', 'item_nbr'])
    print("train_mean", train_mean.columns, train_mean.shape)
    print(train_mean.head(3))

    # store_nbr  item_nbr  dow  unit_sales_dow  unit_sales_week
    return train_mean


def prepare_test_data(ma):
    # dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}

    # Load test
    test = pd.read_csv("../input/test.csv", usecols=[0, 1, 2, 3], dtype=dtypes, parse_dates=['date'])

    print("test", test.columns, test.shape)
    print(test.head(3))

    rs_ma = pd.merge(ma, test, on=['id'])
    rs_ma["dow"] = rs_ma["date"].dt.dayofweek

    print("rs_ma", rs_ma.columns, rs_ma.shape)
    print(rs_ma.head(3))

    # id  unit_sales_ma       date  store_nbr  item_nbr  dow
    return rs_ma


def adjust_sub(rs_ma, train_mean):
    final = pd.merge(rs_ma, train_mean, on=['store_nbr', 'item_nbr', 'dow'], how='left')
    print("final - v1", final.columns, final.shape)
    print(final.head(3))
    #(['id', 'unit_sales_ma', 'date', 'store_nbr', 'item_nbr', 'dow', 'unit_sales_dow', 'unit_sales_week'])

    final['unit_sales'] = final['unit_sales_ma']
    pos_idx = final['unit_sales_week'] > 0
    final_pos = final.loc[pos_idx]

    final.loc[pos_idx, 'unit_sales'] = final_pos['unit_sales_ma'] * final_pos['unit_sales_dow'] / final_pos['unit_sales_week']
    final['unit_sales'] = pd.np.expm1(final['unit_sales']) # restoring unit values

    final[['id','unit_sales']].to_csv('../output/ma_scale_dow.csv.gz', index=False, float_format='%.4f', compression='gzip')

ma = read_ma()
train, lastdate = prepare_train_data()
train_mean = average_dow(train)
rs_ma = prepare_test_data(ma)
adjust_sub(rs_ma, train_mean)
