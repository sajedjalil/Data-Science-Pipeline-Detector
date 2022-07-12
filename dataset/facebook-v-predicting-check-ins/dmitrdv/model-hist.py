# coding: utf-8
__author__ = 'Dmitry Dobryak: https://www.linkedin.com/in/dmitry-dobryak-43822a60?trk=hp-identity-name'

import numpy as np
import pandas as pd
import math
import time
import concurrent.futures

from multiprocessing import cpu_count
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def coord_to_region_id(x, y, size, max):
    x_id = int(1000 * x / size)
    y_id = int(1000 * y / size)

    return y_id * int(max / size) + x_id  

def calc_region(df, size, max):
    df_region = pd.DataFrame([{
                               'place_id': k,
                               'region': coord_to_region_id(gp['x'].median(), gp['y'].median(), size, max),
                              } for k, gp in df.groupby('place_id')])
    
    df_res = pd.merge(df, df_region, on='place_id', how='inner')

    return df_res

# calculate features for each events
def calc_data_feature(df):
    initial_date = np.datetime64('2014-01-01T01:01',   #Arbitrary decision
                                     dtype='datetime64[m]') 
    
    initial_date = np.datetime64('2014-01-01T01:01',   #Arbitrary decision
                                 dtype='datetime64[m]') 
    #working on df_train  
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    
    df_tmp = pd.DataFrame()
    
    df_tmp['row_id'] = df.row_id
    df_tmp['hour'] = d_times.hour
    df_tmp['weekday'] = d_times.weekday
    df_tmp['day'] = d_times.day
    df_tmp['dayofyear'] = d_times.dayofyear
    df_tmp['month'] = d_times.month
    df_tmp['year'] = d_times.year

    df_res = pd.merge(df, df_tmp, on='row_id', how='inner')
    
    return df_res

def split_df(df):
    min_time = int( df.time.min() )
    max_time = int( df.time.max() )

    threshold = int(0.75 * (max_time - min_time))

    df_train = df[ df.time <= min_time + threshold ]
    df_test = df[ df.time > min_time + threshold ]

    return df_train, df_test

# calculate histograms for each places
def calc_place_feature(df):
    df_place_hist_x = pd.DataFrame()
    df_place_hist_y = pd.DataFrame()
    df_place_hist_hour = pd.DataFrame()
    df_place_hist_weekday = pd.DataFrame()
    df_place_hist_month = pd.DataFrame()
    df_place_feature = pd.DataFrame()

    place_feature = []
    range_coord = range(0, 10200, 100)

    for k, gp in df.groupby('place_id'):
        arr_x = []
        arr_y = []
        for item in gp.itertuples():
            arr_x.append(int(1000 * round(item.x, 2)))
            arr_y.append(int(1000 * round(item.y, 2)))

        hist_x, bin_edges = np.histogram(arr_x, bins=range_coord, density=True)
        hist_y, bin_edges = np.histogram(arr_y, bins=range_coord, density=True)

        hist_hour, bin_edges = np.histogram(gp.hour, bins=range(0, 25), density = True)
        hist_weekday, bin_edges = np.histogram(gp.weekday, bins=range(0, 8), density = True)
        hist_month, bin_edges = np.histogram(gp.month, bins=range(1, 14), density = True)

        df_place_hist_x[k] = hist_x
        df_place_hist_y[k] = hist_y
        df_place_hist_hour[k] = hist_hour
        df_place_hist_weekday[k] = hist_weekday
        df_place_hist_month[k] = hist_month

        place_feature.append({ 'place_id': k,
                               'place_x': gp['x'].median(), 
                               'place_y': gp['y'].median(), 
                               'place_count': gp.shape[0] })

    df_place_feature = pd.DataFrame(place_feature).set_index(['place_id'])

    return df_place_feature, df_place_hist_x, df_place_hist_y, df_place_hist_hour, df_place_hist_weekday, df_place_hist_month

# join each event with all places histograms
def generate_train_ds(df_data, df_place_feature, df_place_hist_x, df_place_hist_y, df_place_hist_hour, df_place_hist_weekday, df_place_hist_month):
    arr = []

    for item in df_data.itertuples():
        df = pd.DataFrame()

        df['x_prob'] = df_place_hist_x.iloc[int(10 * round(item.x, 2))]
        df['y_prob'] = df_place_hist_y.iloc[int(10 * round(item.y, 2))]
        df['hour_prob'] = df_place_hist_hour.iloc[int(item.hour)]
        df['weekday_prob'] = df_place_hist_weekday.iloc[int(item.weekday)]
        df['month_prob'] = df_place_hist_month.iloc[int(item.month) - 1]

        df['row_id'] = item.row_id
        df['accuracy'] = item.accuracy
        df['place_id'] = df_place_feature.index

        df['gravity'] = [ place_feature.place_count / math.sqrt( (place_feature.place_x - item.x)**2 + (place_feature.place_y - item.y)**2 + 0.0001) 
                          for place_feature in df_place_feature.itertuples() ]

        df['result'] = [ 1 if place_id == item.place_id else 0 for place_id in df_place_feature.index ]

        arr.append(df)

    return pd.concat(arr)

def train(df_train_input, df_train_output, df_test_input, df_test_output, estimators):
    model_rfc = RandomForestClassifier(n_estimators = estimators).fit(df_train_input, df_train_output)
    score = model_rfc.score(df_test_input, df_test_output)

    return model_rfc, score

def build(df_train, df_test):
    print('calc place hists')
    df_place_feature, df_place_hist_x, df_place_hist_y, df_place_hist_hour, df_place_hist_weekday, df_place_hist_month = calc_place_feature(df_train)
    
    print('generate train input data')
    df_train_ds = generate_train_ds(df_train, df_place_feature, df_place_hist_x, df_place_hist_y, df_place_hist_hour, df_place_hist_weekday, df_place_hist_month)
    df_test_ds = generate_train_ds(df_test, df_place_feature, df_place_hist_x, df_place_hist_y, df_place_hist_hour, df_place_hist_weekday, df_place_hist_month)
    
    print('train model')
    df_train_positive = df_train_ds[df_train_ds.result == 1]
    df_train_negative = df_train_ds[df_train_ds.result == 0].sample(frac = 0.05)
    df_train_negative = df_train_negative.sample(n = min([ 2 * df_train_positive.shape[0], df_train_negative.shape[0] ]))
    df_train_short = pd.concat( [df_train_positive, df_train_negative] )

    df_train_output = df_train_short.result
    df_train_input = df_train_short[ ['x_prob', 'y_prob', 'hour_prob', 'weekday_prob', 'month_prob', 'accuracy', 'gravity'] ]

    df_test_output = df_test_ds.result
    df_test_input = df_test_ds[ ['x_prob', 'y_prob', 'hour_prob', 'weekday_prob', 'month_prob', 'accuracy', 'gravity'] ]

    model, score = train(df_train_input, df_train_output, df_test_input, df_test_output, 70)
    print('model score %s' % score)
    
    return model, score

if __name__ == '__main__':
    # load df
    print('load data')
    df = pd.read_csv('../input/train.csv')
    
    # keep places in 1 region for test purposes
    print('filter data by region')
    df = calc_region(df, 250, 1000)
    df = df[ df.region == 0 ]
    df = df.drop('region', axis = 1)
    #places = pd.Series(df.place_id.unique()).sample(n = 100);
    #df = df[ df['place_id'].isin(places) ]


    # calc features
    print('calc data features')
    df = calc_data_feature(df)

    # split df to train and test by time
    df_train, df_test = split_df(df)

    # build model
    build(df_train, df_test)

    print('finish')