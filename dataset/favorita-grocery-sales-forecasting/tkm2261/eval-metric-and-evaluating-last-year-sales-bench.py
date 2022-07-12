# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

WEIGHTS = None

def NWRMSLE(y, pred):
    y = y.clip(0, y.max())
    pred = pred.clip(0, pred.max())
    score = np.nansum(WEIGHTS * ((np.log1p(pred) - np.log1p(y)) ** 2)) / WEIGHTS.sum()
    return np.sqrt(score)

if __name__ == '__main__':
    df = pd.read_csv('../input/train.csv',
             usecols=['date', 'store_nbr', 'item_nbr', 'unit_sales'],
             dtype={'store_nbr': np.int32, 'item_nbr': np.int32, 'unit_sales':np.float32},
             skiprows=(1, 66458908),
             parse_dates=['date'])
    print('loading end')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    print('extracting end')
    df_item = pd.read_csv('../input/items.csv',
                          dtype={'item_nbr':np.int32, 'perishable':np.bool},
                          usecols=['item_nbr', 'perishable'])
    df = df.merge(df_item, how='left', on='item_nbr', copy=False)
    
    print('merging end')
    
    del df_item
    gc.collect()
    
    df_2016 = df[df['year'] == 2016]
    df_2017 = df[df['year'] == 2017]

    del df
    gc.collect()

    print('splitting end')

    df_2016.rename(columns={'unit_sales': 'prev_unit_sales'}, inplace=True)

    w = df_2017.perishable.values
    WEIGHTS = np.where(w == 1, 1.25, 1.)

    df_2017 = df_2017.merge(df_2016, how='left', on=['store_nbr', 'item_nbr', 'month', 'day'])

    print('merging end')

    y_train = df_2017.unit_sales.values
    y_train = np.where(y_train < 0, 0, y_train)

    preds = df_2017.prev_unit_sales.fillna(0).values
    preds = np.where(preds < 0, 0, preds)

    print('data prep. end')

    print('Zero-filling NWRMSLE:', NWRMSLE(y_train, preds))

    preds = df_2017.prev_unit_sales.fillna(df_2017.prev_unit_sales.mean()).values
    preds = np.where(preds < 0, 0, preds)

    print('Mean-filling NWRMSLE:', NWRMSLE(y_train, preds))
    print('These scores seem like the Last Year Sales Bench score')
