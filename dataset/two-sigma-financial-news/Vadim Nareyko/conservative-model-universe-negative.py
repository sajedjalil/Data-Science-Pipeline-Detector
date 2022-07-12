import numpy as np
import pandas as pd
import numpy.ma as ma

import random
from datetime import datetime, date
import time
import gc
from resource import getrusage, RUSAGE_SELF

import warnings
warnings.filterwarnings('ignore')

import multiprocessing

N_THREADS=multiprocessing.cpu_count()
N_LAG = [10]
RETURN_FEATURES = ['returnsOpenPrevMktres10']
EPS = 1e-17

import logging as logging
def ini_log(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handlers = [logging.StreamHandler(None), logging.FileHandler(filename, 'a')]

    fmt=logging.Formatter('%(asctime)-15s: %(levelname)s  %(message)s')
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

log = ini_log('out.log')
log.info(f'N_THREADS: {N_THREADS}')

old_max = 0
def using(point=""):
    global old_max
    max_rss = getrusage(RUSAGE_SELF).ru_maxrss
    if max_rss > old_max:
        old_max = max_rss
    log.info(f'{point} max RSS {old_max/1024/1024:.1f}Gib')
    gc.collect();
    
from multiprocessing import Pool

def create_lag(df_code):
    global N_LAG
    n_lag=N_LAG
    prevlag = 1    
    for window in np.sort(n_lag):
        rolled = df_code[RETURN_FEATURES].shift(prevlag).rolling(window=window)
        df_code = df_code.join(rolled.mean().add_suffix(f'_w_{window}_mean'))
        df_code = df_code.join(rolled.std().add_suffix(f'_w_{window}_std'))
    return df_code.fillna(0)

def generate_lag_features(df,n_lag = N_LAG):

    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+RETURN_FEATURES] for df_code in df_codes]
    
    pool = Pool(N_THREADS)
    all_df = pool.map(create_lag, df_codes)
    
    new_df = pd.concat(all_df)
    del all_df; gc.collect()
    using(f'Generated LAGS')
    new_df.drop(RETURN_FEATURES,axis=1,inplace=True)
    pool.close();
    del pool; gc.collect()
    
    return new_df
    
def preparedf(df):
    if df.time.dtype.kind != 'O':
        df['time'] = df['time'].dt.date
    return df
    
def get_metric(df, c='returns', eps=EPS):
    df[c] = df['predict'] * df.returnsOpenNextMktres10
    day_returns = df.groupby('time')[c].sum().reset_index()
    return day_returns[c].mean() / (day_returns[c].std()+eps)

def initialize_values(items=5000, features=4, history=15):
    return np.ones((items, features, history))*np.nan

def add_values(a, items=100):
    return np.concatenate([a, initialize_values(items, a.shape[1], a.shape[2])])

def get_code(a):
    global codes, history
    try: 
        return codes[a]
    except KeyError:
        codes[a] = len(codes)
        if len(codes) > history.shape[0]:
            history = add_values(history, 100)
        return codes[a]

def list2codes(l):
    return np.array([get_code(a) for a in l])
    
    
from kaggle.competitions import twosigmanews

log.info('Reading datasets')
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
del news_train_df
using('Done')

log.info('Train Dataset Preparation')
market_train_df = market_train_df[market_train_df.universe==1]
market_train_df = preparedf(market_train_df)
lags = generate_lag_features(market_train_df)
market_train_df = pd.merge(market_train_df,lags,how='left',on=['time','assetCode'])
log.info('Done')

log.info('Train Dataset prediction')
market_train_df['predict']=0
coeff = 1.4
mean = market_train_df.returnsOpenPrevMktres10_w_10_mean.values
std = market_train_df.returnsOpenPrevMktres10_w_10_std.values
sign = np.sign(mean)
limit = mean - coeff*sign*std
positive = (market_train_df.returnsOpenPrevMktres10 > 0) & (mean > 0) & (limit > 0)
market_train_df.loc[positive, 'predict']=1

v = np.sort(market_train_df.returnsOpenPrevMktres10.dropna().values)
vm = v[len(v)//10000:len(v)-len(v)//10000]
v_mean = vm.mean()
v_std = vm.std()

market_train_df.loc[market_train_df.returnsOpenPrevMktres10<vm.mean()-coeff*vm.std(), 'predict'] = 0
market_train_df.loc[market_train_df.returnsOpenPrevMktres10>vm.mean()+coeff*vm.std(), 'predict'] = 0

using('Done')

log.info('Preparation for prediction')
codes = dict(zip(market_train_df.assetCode.unique(), np.arange(market_train_df.assetCode.nunique())))
history = initialize_values(len(codes), len(RETURN_FEATURES), np.max(N_LAG)+1)

latest_events = market_train_df.groupby('assetCode').tail(np.max(N_LAG)+1)
latest_events_size = latest_events.groupby('assetCode').size()

for s in latest_events_size.unique():
    for i in range(len(RETURN_FEATURES)):
        l = latest_events[
            latest_events.assetCode.isin(latest_events_size[latest_events_size==s].index.values)
        ].groupby('assetCode')[RETURN_FEATURES[i]].apply(list)
        v = np.array([k for k in l.values])
        r = list2codes(l.index.values)
        history[r, i, -s:] = v
        del l, v, r
        
del latest_events, latest_events_size
        
del market_train_df; 
using('Done')


log.info('Prediction')
days = env.get_prediction_days()
n_days = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    del news_obs_df
    
    n_days +=1
    if (n_days % 50 == 0):
        using(f'{n_days}')
        
    market_obs_df = preparedf(market_obs_df)
    
    r = list2codes(market_obs_df.assetCode.values)
    history[r, :, :-1] = history[r, :, 1:] 
    history[r, :, -1] = market_obs_df[RETURN_FEATURES].values
    
    prevlag = 1    
    for lag in np.sort(N_LAG):
        lag_mean = np.mean(history[r, : , -lag:-prevlag], axis=2)
        lag_std = np.std(history[r, : , -lag:-prevlag], axis=2)

        for ix in range(len(RETURN_FEATURES)):
            market_obs_df[f'{RETURN_FEATURES[ix]}_w_{lag}_mean'] = lag_mean[:, ix]
            market_obs_df[f'{RETURN_FEATURES[ix]}_w_{lag}_std'] = lag_std[:, ix]
    
    market_obs_df['predict'] = 0
    mean = market_obs_df.returnsOpenPrevMktres10_w_10_mean.values
    std = market_obs_df.returnsOpenPrevMktres10_w_10_std.values
    sign = np.sign(mean)
    limit = mean - coeff*sign*std
    positive = (market_obs_df.returnsOpenPrevMktres10 > 0) & (mean > 0) & (limit > 0)
    negative = (market_obs_df.returnsOpenPrevMktres10 < 0) & (mean < 0) & (limit < 0)
    market_obs_df.loc[positive, 'predict']=1
    market_obs_df.loc[negative, 'predict']=-1

    market_obs_df.loc[market_obs_df.returnsOpenPrevMktres10<vm.mean()-coeff*vm.std(), 'predict'] = 0
    market_obs_df.loc[market_obs_df.returnsOpenPrevMktres10>vm.mean()+coeff*vm.std(), 'predict'] = 0

    market_obs_df['sharpe'] = market_obs_df.returnsOpenPrevMktres10_w_10_mean / (market_obs_df.returnsOpenPrevMktres10_w_10_std + EPS)
    s_mean = market_obs_df.sharpe.mean()
    s_std = market_obs_df.sharpe.std()
    market_obs_df.loc[market_obs_df.sharpe>s_mean+2*s_std, 'predict'] = 0
    market_obs_df.loc[market_obs_df.sharpe>s_mean-2*s_std, 'predict'] = 0


    market_obs_df['confidence'] = market_obs_df['predict'].values
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_obs_df['confidence']})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})

    env.predict(predictions_template_df)    

using('Prediction done')

env.write_submission_file()
using('Submission done')    
