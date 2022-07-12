## Import Libraries -----------------------------------------------
import numpy as np 
import pandas as pd 
from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from fastai import *
from fastai.tabular import *

import os

## set path
PATH = "../input/"
# print(os.listdir("../input"))

## define functions --------------------------------------------------------------------
# vectorized haversine function
def haversine_vect(lat1, lon1, lat2, lon2, to_radians = True, earth_radius = 6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = pd.np.radians([lat1, lon1, lat2, lon2])

    a = pd.np.sin((lat2-lat1)/2.0)**2 + \
        pd.np.cos(lat1) * pd.np.cos(lat2) * pd.np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * pd.np.arcsin(np.sqrt(a))
    
from pandas.api.types import is_string_dtype, is_numeric_dtype

## from fastai
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

## from fastai
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

## from fastai
def numericalize(df, col, name, max_n_cat):
  if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
      df[name] = col.cat.codes+1
        
## from fastai
def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict    
    
## Read data---------------------------------------------------------------------------    
traintypes = {'key': 'object',
              'fare_amount': 'float32',
              # 'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'int8'}

## read part of the training data
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory = False, nrows = 4000000, verbose = True,
                     parse_dates = ['pickup_datetime'],
                     dtype = traintypes)
## read test data
df_test_raw = pd.read_csv(f'{PATH}test.csv', low_memory=False, parse_dates = ['pickup_datetime'])

df_raw.drop(['key'], axis=1, inplace = True)
df_test_raw.drop(['key'], axis=1, inplace = True)

## Clean Data (quick cleaning) ---------------------------------------------------------

## remove entries with fare < 0
df_raw = df_raw[df_raw['fare_amount'] > 0]


## Add features ---------------------------------------------------------------------------
## add date part
add_datepart(df_raw, 'pickup_datetime')

## add haversine distance
df_raw['h_distance'] = haversine_vect(df_raw["pickup_latitude"], df_raw["pickup_longitude"],\
                                      df_raw["dropoff_latitude"], df_raw["dropoff_longitude"])
## add data part
add_datepart(df_test_raw, 'pickup_datetime')

## add haversine distance
df_test_raw['h_distance'] = haversine_vect(df_test_raw["pickup_latitude"], df_test_raw["pickup_longitude"],\
                                           df_test_raw["dropoff_latitude"], df_test_raw["dropoff_longitude"])   
## process data and return NA dict                                           
df, y, nas = proc_df(df_raw, 'fare_amount')
df_t, _, _= proc_df(df_test_raw, na_dict = nas) 

## Fit model -------------------------------------------------------------------------------------------
m = RandomForestRegressor(n_estimators = 100, 
                          n_jobs = -1,
                          min_samples_leaf = 3,
                          max_features = 0.5,
                          oob_score = True)
m.fit(df, y)

## Predict on test set -----------------------------------------------------------------------------------
res = m.predict(df_t)

## Create submission -------------------------------------------------------------------------------------
subm = pd.read_csv(f'{PATH}sample_submission.csv')
subm['fare_amount'] = res

subm.to_csv('submission.csv', index=False)