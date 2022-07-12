import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
import itertools
import pickle, gzip
import glob
from sklearn.preprocessing import StandardScaler

import sys, os
import argparse
import time
from datetime import datetime as dt
import gc; gc.enable()
from functools import partial, wraps


np.warnings.filterwarnings('ignore')

from tsfresh.feature_extraction import extract_features

from numba import jit



def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) from 
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    
    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine, 
        #'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)), 
   }


def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq, 
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq,}, 
        index=df.index)
    
    return pd.concat([df, df_flux], axis=1)


def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_q1'].values
    
    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        #'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,       
        'flux_diff3': flux_diff /flux_w_mean,
        }, index=df.index)
    
    return pd.concat([df, df_flux_agg], axis=1)


def featurize(df, df_meta, aggs, fcp, n_jobs=4):
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """
    df['mjd_inc'] = df[['object_id', 'mjd']].groupby(['object_id'])['mjd'].shift(-1)
    df['mjd_inc']=df['mjd']-df['mjd_inc']
    df['bloque']=0
    df.loc[df.mjd_inc>100,'bloque']=1
    df['bloque'] = df[['object_id', 'bloque']].groupby(['object_id'])['bloque'].cumsum()
    df['mjd_inc'] = df[['object_id', 'mjd','passband','bloque']].groupby(['object_id','passband','bloque'])['mjd'].shift(-1)
    df['flux_inc'] = df[['object_id', 'flux','passband','bloque']].groupby(['object_id','passband','bloque'])['flux'].shift(-1)
    df['mjd_inc']=df['mjd']-df['mjd_inc']
    df['flux_inc']=df['flux']-df['flux_inc']
    df['pen']=df['flux_inc']/df['mjd_inc']
    df['area']=df['flux_inc']*df['mjd_inc']
    
    def q1(x):
        return x.quantile(0.10)
    
    aggs = {
        'flux': [ 'max', 'mean', 'median', 'skew',q1], #'std''min',
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum'],
        'flux_by_flux_ratio_sq':['sum','skew'],
        'area': [ 'sum','max', 'mean', 'median', 'skew',q1],
    }
    
    df = process_flux(df)

    agg_df = df.groupby('object_id').agg(aggs)
    
    aggs = {
        'flux': [ 'max', 'mean', 'median', 'skew','q1'], #'std''min',
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum'],
        'flux_by_flux_ratio_sq':['sum','skew'],
        'area': ['sum', 'max', 'mean', 'median', 'skew','q1']
    }
    
    agg_df.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df = process_flux_agg(agg_df) # new feature to play with tsfresh
    
    
    # Add more features with
    agg_df_ts_flux_passband = extract_features(df, 
                                               column_id='object_id', 
                                               column_sort='mjd', 
                                               column_kind='passband', 
                                               column_value='flux', 
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=n_jobs)
    
    agg_df_ts_flux_passband['0__range'] = agg_df_ts_flux_passband['0__maximum'].values - agg_df_ts_flux_passband['0__minimum'].values
    agg_df_ts_flux_passband['1__range'] = agg_df_ts_flux_passband['1__maximum'].values - agg_df_ts_flux_passband['1__minimum'].values
    agg_df_ts_flux_passband['2__range'] = agg_df_ts_flux_passband['2__maximum'].values - agg_df_ts_flux_passband['2__minimum'].values
    agg_df_ts_flux_passband['3__range'] = agg_df_ts_flux_passband['3__maximum'].values - agg_df_ts_flux_passband['3__minimum'].values
    agg_df_ts_flux_passband['4__range'] = agg_df_ts_flux_passband['4__maximum'].values - agg_df_ts_flux_passband['4__minimum'].values
    agg_df_ts_flux_passband['5__range'] = agg_df_ts_flux_passband['5__maximum'].values - agg_df_ts_flux_passband['5__minimum'].values
    
    agg_df_ts_flux_passband['0__1range'] = agg_df_ts_flux_passband['0__maximum'].values - agg_df_ts_flux_passband['1__maximum'].values
    agg_df_ts_flux_passband['0__2range'] = agg_df_ts_flux_passband['0__maximum'].values - agg_df_ts_flux_passband['2__maximum'].values
    agg_df_ts_flux_passband['0__3range'] = agg_df_ts_flux_passband['0__maximum'].values - agg_df_ts_flux_passband['3__maximum'].values
    agg_df_ts_flux_passband['0__4range'] = agg_df_ts_flux_passband['0__maximum'].values - agg_df_ts_flux_passband['4__maximum'].values
    agg_df_ts_flux_passband['0__5range'] = agg_df_ts_flux_passband['0__maximum'].values - agg_df_ts_flux_passband['5__maximum'].values
    
    agg_df_ts_flux_passband['1__2range'] = agg_df_ts_flux_passband['1__maximum'].values - agg_df_ts_flux_passband['2__maximum'].values
    agg_df_ts_flux_passband['1__3range'] = agg_df_ts_flux_passband['1__maximum'].values - agg_df_ts_flux_passband['3__maximum'].values
    agg_df_ts_flux_passband['1__4range'] = agg_df_ts_flux_passband['1__maximum'].values - agg_df_ts_flux_passband['4__maximum'].values
    agg_df_ts_flux_passband['1__5range'] = agg_df_ts_flux_passband['1__maximum'].values - agg_df_ts_flux_passband['5__maximum'].values
    
    agg_df_ts_flux_passband['2__3range'] = agg_df_ts_flux_passband['2__maximum'].values - agg_df_ts_flux_passband['3__maximum'].values
    agg_df_ts_flux_passband['2__4range'] = agg_df_ts_flux_passband['2__maximum'].values - agg_df_ts_flux_passband['4__maximum'].values
    agg_df_ts_flux_passband['2__5range'] = agg_df_ts_flux_passband['2__maximum'].values - agg_df_ts_flux_passband['5__maximum'].values
    
    agg_df_ts_flux_passband['3__4range'] = agg_df_ts_flux_passband['3__maximum'].values - agg_df_ts_flux_passband['4__maximum'].values
    agg_df_ts_flux_passband['3__5range'] = agg_df_ts_flux_passband['3__maximum'].values - agg_df_ts_flux_passband['5__maximum'].values
    
    agg_df_ts_flux_passband['4__5range'] = agg_df_ts_flux_passband['4__maximum'].values - agg_df_ts_flux_passband['5__maximum'].values
    
    agg_df_ts_flux_passband['0__1range_rr'] = agg_df_ts_flux_passband['0__range'].values - agg_df_ts_flux_passband['1__range'].values
    agg_df_ts_flux_passband['0__2range_rr'] = agg_df_ts_flux_passband['0__range'].values - agg_df_ts_flux_passband['2__range'].values
    agg_df_ts_flux_passband['0__3range_rr'] = agg_df_ts_flux_passband['0__range'].values - agg_df_ts_flux_passband['3__range'].values
    agg_df_ts_flux_passband['0__4range_rr'] = agg_df_ts_flux_passband['0__range'].values - agg_df_ts_flux_passband['4__range'].values
    agg_df_ts_flux_passband['0__5range_rr'] = agg_df_ts_flux_passband['0__range'].values - agg_df_ts_flux_passband['5__range'].values
    
    agg_df_ts_flux_passband['1__2range_rr'] = agg_df_ts_flux_passband['1__range'].values - agg_df_ts_flux_passband['2__range'].values
    agg_df_ts_flux_passband['1__3range_rr'] = agg_df_ts_flux_passband['1__range'].values - agg_df_ts_flux_passband['3__range'].values
    agg_df_ts_flux_passband['1__4range_rr'] = agg_df_ts_flux_passband['1__range'].values - agg_df_ts_flux_passband['4__range'].values
    agg_df_ts_flux_passband['1__5range_rr'] = agg_df_ts_flux_passband['1__range'].values - agg_df_ts_flux_passband['5__range'].values
    
    agg_df_ts_flux_passband['2__3range_rr'] = agg_df_ts_flux_passband['2__range'].values - agg_df_ts_flux_passband['3__range'].values
    agg_df_ts_flux_passband['2__4range_rr'] = agg_df_ts_flux_passband['2__range'].values - agg_df_ts_flux_passband['4__range'].values
    agg_df_ts_flux_passband['2__5range_rr'] = agg_df_ts_flux_passband['2__range'].values - agg_df_ts_flux_passband['5__range'].values
    
    agg_df_ts_flux_passband['3__4range_rr'] = agg_df_ts_flux_passband['3__range'].values - agg_df_ts_flux_passband['4__range'].values
    agg_df_ts_flux_passband['3__5range_rr'] = agg_df_ts_flux_passband['3__range'].values - agg_df_ts_flux_passband['5__range'].values
    
    agg_df_ts_flux_passband['4__5range_rr'] = agg_df_ts_flux_passband['4__range'].values - agg_df_ts_flux_passband['5__range'].values
    
    agg_df_ts_flux_passband['0__1mean_rr'] = agg_df_ts_flux_passband['0__mean'].values - agg_df_ts_flux_passband['1__mean'].values
    agg_df_ts_flux_passband['0__2mean_rr'] = agg_df_ts_flux_passband['0__mean'].values - agg_df_ts_flux_passband['2__mean'].values
    agg_df_ts_flux_passband['0__3mean_rr'] = agg_df_ts_flux_passband['0__mean'].values - agg_df_ts_flux_passband['3__mean'].values
    agg_df_ts_flux_passband['0__4mean_rr'] = agg_df_ts_flux_passband['0__mean'].values - agg_df_ts_flux_passband['4__mean'].values
    agg_df_ts_flux_passband['0__5mean_rr'] = agg_df_ts_flux_passband['0__mean'].values - agg_df_ts_flux_passband['5__mean'].values
    
    agg_df_ts_flux_passband['1__2mean_rr'] = agg_df_ts_flux_passband['1__mean'].values - agg_df_ts_flux_passband['2__mean'].values
    agg_df_ts_flux_passband['1__3mean_rr'] = agg_df_ts_flux_passband['1__mean'].values - agg_df_ts_flux_passband['3__mean'].values
    agg_df_ts_flux_passband['1__4mean_rr'] = agg_df_ts_flux_passband['1__mean'].values - agg_df_ts_flux_passband['4__mean'].values
    agg_df_ts_flux_passband['1__5mean_rr'] = agg_df_ts_flux_passband['1__mean'].values - agg_df_ts_flux_passband['5__mean'].values
    
    agg_df_ts_flux_passband['2__3mean_rr'] = agg_df_ts_flux_passband['2__mean'].values - agg_df_ts_flux_passband['3__mean'].values
    agg_df_ts_flux_passband['2__4mean_rr'] = agg_df_ts_flux_passband['2__mean'].values - agg_df_ts_flux_passband['4__mean'].values
    agg_df_ts_flux_passband['2__5mean_rr'] = agg_df_ts_flux_passband['2__mean'].values - agg_df_ts_flux_passband['5__mean'].values
    
    agg_df_ts_flux_passband['3__4mean_rr'] = agg_df_ts_flux_passband['3__mean'].values - agg_df_ts_flux_passband['4__mean'].values
    #agg_df_ts_flux_passband['3__5mean_rr'] = agg_df_ts_flux_passband['3__mean'].values - agg_df_ts_flux_passband['5__mean'].values
    
    #agg_df_ts_flux_passband['4__5mean_rr'] = agg_df_ts_flux_passband['4__mean'].values - agg_df_ts_flux_passband['5__mean'].values
    
    
    del agg_df_ts_flux_passband['1__range']
    
    
    agg_df_ts_flux = extract_features(df, 
                                      column_id='object_id', 
                                      column_value='flux', 
                                      default_fc_parameters=fcp['flux'], n_jobs=n_jobs)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(df, 
                                      column_id='object_id', 
                                      column_value='flux_by_flux_ratio_sq', 
                                      default_fc_parameters=fcp['flux_by_flux_ratio_sq'], n_jobs=n_jobs)

    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, 
                                  column_id='object_id', 
                                  column_value='mjd', 
                                  default_fc_parameters=fcp['mjd'], n_jobs=n_jobs)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    
    agg_df_mjd_pas = extract_features(df_det, 
                                  column_id='object_id', 
                                  column_value='mjd', 
                                  column_kind='passband',
                                  default_fc_parameters=fcp['mjd_pas'], n_jobs=n_jobs)
    
    agg_df_mjd_pas.columns = [str(col) + '_mjd_pas' for col in agg_df_mjd_pas.columns]
    
    agg_df_mjd_pas['0_diff_det_mjd_pas'] = agg_df_mjd_pas['0__maximum_mjd_pas'].values - agg_df_mjd_pas['0__minimum_mjd_pas'].values
    agg_df_mjd_pas['1_diff_det_mjd_pas'] = agg_df_mjd_pas['1__maximum_mjd_pas'].values - agg_df_mjd_pas['1__minimum_mjd_pas'].values
    agg_df_mjd_pas['2_diff_det_mjd_pas'] = agg_df_mjd_pas['2__maximum_mjd_pas'].values - agg_df_mjd_pas['2__minimum_mjd_pas'].values
    agg_df_mjd_pas['3_diff_det_mjd_pas'] = agg_df_mjd_pas['3__maximum_mjd_pas'].values - agg_df_mjd_pas['3__minimum_mjd_pas'].values
    agg_df_mjd_pas['4_diff_det_mjd_pas'] = agg_df_mjd_pas['4__maximum_mjd_pas'].values - agg_df_mjd_pas['4__minimum_mjd_pas'].values
    agg_df_mjd_pas['5_diff_det_mjd_pas'] = agg_df_mjd_pas['5__maximum_mjd_pas'].values - agg_df_mjd_pas['5__minimum_mjd_pas'].values
    
    del agg_df_mjd_pas['0__maximum_mjd_pas'], agg_df_mjd_pas['0__minimum_mjd_pas']
    del agg_df_mjd_pas['1__maximum_mjd_pas'], agg_df_mjd_pas['1__minimum_mjd_pas']
    del agg_df_mjd_pas['2__maximum_mjd_pas'], agg_df_mjd_pas['2__minimum_mjd_pas']
    del agg_df_mjd_pas['3__maximum_mjd_pas'], agg_df_mjd_pas['3__minimum_mjd_pas']
    del agg_df_mjd_pas['4__maximum_mjd_pas'], agg_df_mjd_pas['4__minimum_mjd_pas']
    del agg_df_mjd_pas['5__maximum_mjd_pas'], agg_df_mjd_pas['5__minimum_mjd_pas']
    
    agg_df_mjd_pas['0_1_range_diff_mjd_pas'] = agg_df_mjd_pas['0_diff_det_mjd_pas'].values - agg_df_mjd_pas['1_diff_det_mjd_pas'].values
    agg_df_mjd_pas['0_2_range_diff_mjd_pas'] = agg_df_mjd_pas['0_diff_det_mjd_pas'].values - agg_df_mjd_pas['2_diff_det_mjd_pas'].values
    agg_df_mjd_pas['0_3_range_diff_mjd_pas'] = agg_df_mjd_pas['0_diff_det_mjd_pas'].values - agg_df_mjd_pas['3_diff_det_mjd_pas'].values
    agg_df_mjd_pas['0_4_range_diff_mjd_pas'] = agg_df_mjd_pas['0_diff_det_mjd_pas'].values - agg_df_mjd_pas['4_diff_det_mjd_pas'].values
    agg_df_mjd_pas['0_5_range_diff_mjd_pas'] = agg_df_mjd_pas['0_diff_det_mjd_pas'].values - agg_df_mjd_pas['5_diff_det_mjd_pas'].values

    agg_df_mjd_pas['1_2_range_diff_mjd_pas'] = agg_df_mjd_pas['1_diff_det_mjd_pas'].values - agg_df_mjd_pas['2_diff_det_mjd_pas'].values
    agg_df_mjd_pas['1_3_range_diff_mjd_pas'] = agg_df_mjd_pas['1_diff_det_mjd_pas'].values - agg_df_mjd_pas['3_diff_det_mjd_pas'].values
    agg_df_mjd_pas['1_4_range_diff_mjd_pas'] = agg_df_mjd_pas['1_diff_det_mjd_pas'].values - agg_df_mjd_pas['4_diff_det_mjd_pas'].values
    agg_df_mjd_pas['1_5_range_diff_mjd_pas'] = agg_df_mjd_pas['1_diff_det_mjd_pas'].values - agg_df_mjd_pas['5_diff_det_mjd_pas'].values

    agg_df_mjd_pas['2_3_range_diff_mjd_pas'] = agg_df_mjd_pas['2_diff_det_mjd_pas'].values - agg_df_mjd_pas['3_diff_det_mjd_pas'].values
    agg_df_mjd_pas['2_4_range_diff_mjd_pas'] = agg_df_mjd_pas['2_diff_det_mjd_pas'].values - agg_df_mjd_pas['4_diff_det_mjd_pas'].values
    agg_df_mjd_pas['2_5_range_diff_mjd_pas'] = agg_df_mjd_pas['2_diff_det_mjd_pas'].values - agg_df_mjd_pas['5_diff_det_mjd_pas'].values

    agg_df_mjd_pas['3_4_range_diff_mjd_pas'] = agg_df_mjd_pas['3_diff_det_mjd_pas'].values - agg_df_mjd_pas['4_diff_det_mjd_pas'].values
    agg_df_mjd_pas['3_5_range_diff_mjd_pas'] = agg_df_mjd_pas['3_diff_det_mjd_pas'].values - agg_df_mjd_pas['5_diff_det_mjd_pas'].values

    agg_df_mjd_pas['4_5_range_diff_mjd_pas'] = agg_df_mjd_pas['4_diff_det_mjd_pas'].values - agg_df_mjd_pas['5_diff_det_mjd_pas'].values

    
    agg_df_ts_flux_passband.index.rename('object_id', inplace=True) 
    agg_df_ts_flux.index.rename('object_id', inplace=True) 
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True) 
    agg_df_mjd.index.rename('object_id', inplace=True)      
    agg_df_mjd_pas.index.rename('object_id', inplace=True)  
    agg_df_ts = pd.concat([agg_df, 
                           agg_df_ts_flux_passband, 
                           agg_df_ts_flux, 
                           agg_df_ts_flux_by_flux_ratio_sq, 
                           agg_df_mjd,
                           agg_df_mjd_pas], axis=1).reset_index()
    
    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    
    result['hpxmjd_det']=result['hostgal_photoz']*result['mjd_diff_det']
    return result


def process_meta(filename):
    meta_df = pd.read_csv(filename)
    
    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values, 
                   meta_df['gal_l'].values, meta_df['gal_b'].values))
    #
    #meta_dict['hostgal_photoz_certain'] = np.multiply(
    #         meta_df['hostgal_photoz'].values, 
    #         np.exp(meta_df['hostgal_photoz_err'].values))
    
    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    
    meta_df['hp_ratio_sq']= np.power(meta_df['hostgal_photoz']/ meta_df['hostgal_photoz_err'], 2.0)
    #meta_df['hp_by_hp_ratio_sq']= meta_df['hostgal_photoz'] * meta_df['hp_ratio_sq'] 
    
    return meta_df


####

aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum', 'skew'],
        'flux_by_flux_ratio_sq':['sum','skew']}
fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            #'longest_strike_below_mean': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
                
        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,       
        },
                
        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'}, 
                    {'coeff': 1, 'attr': 'abs'}
                ],
            'kurtosis' : None, 
            'skewness' : None,
            'mean': None,
            'maximum': None,
            'minimum': None
        },
                
        'mjd': {
            'maximum': None, 
            'minimum': None,
            'mean_change': None,
            #'mean_abs_change': None,
            'mean': None
        },
        
        'mjd_pas': {
            'maximum': None, 
            'minimum': None,
            #'mean_change': None,
            #'mean_abs_change': None,
           # 'mean': None
        },}
meta_train = process_meta('../input/training_set_metadata.csv')    
train = pd.read_csv('../input/training_set.csv')
full_train = featurize(train, meta_train, aggs, fcp)

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
classes = sorted(y.unique())

class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)


    
if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'] 
    #del full_train['distmod'] 
    del full_train['hostgal_specz']
    del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
    del full_train['ddf']   
    del full_train['4__kurtosis']
    del full_train['5__skewness']
    del full_train['mwebv']
    del full_train['5__minimum']
    del full_train['4__minimum']
    del full_train['flux_ratio_sq_sum']
    del full_train['2__maximum']
    del full_train['0__fft_coefficient__coeff_0__attr_"abs"']
    del full_train['1__minimum']
    
    
    # Create correlation matrix
    corr_matrix = full_train.corr().abs()

# Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(to_drop)
    full_train=full_train.drop(to_drop, axis=1)

train_mean = full_train.mean(axis=0)
full_train.fillna(train_mean, inplace=True)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

full_train_new = full_train.copy()
ss = StandardScaler()
full_train_ss = ss.fit_transform(full_train_new)

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter
from sklearn.metrics import confusion_matrix

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred):  
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss
    
def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

K.clear_session()
def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 512
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=full_train_ss.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//2,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//4,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//8,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
    
    model.add(Dense(len(classes), activation='softmax'))
    return model

unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
        
y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map)

y_count = Counter(y_map)
wtable = np.zeros((len(unique_y),))
for i in range(len(unique_y)):
    wtable[i] = y_count[i]/y_map.shape[0]


clfs = []
oof_preds = np.zeros((len(full_train_ss), len(classes)))
epochs = 600
batch_size = 100
for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    print(fold_)
    checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)
    x_train, y_train = full_train_ss[trn_], y_categorical[trn_]
    x_valid, y_valid = full_train_ss[val_], y_categorical[val_]
    
    model = build_model(dropout_rate=0.5,activation='tanh')    
    model.compile(loss=mywloss, optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,shuffle=True,verbose=0,callbacks=[checkPoint])       
    
    print('Loading Best Model')
    model.load_weights('./keras.model')
    # # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict_proba(x_valid,batch_size=batch_size)
    print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid,batch_size=batch_size)))
    clfs.append(model)
    
print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_categorical,oof_preds))

sample_sub = pd.read_csv('../input/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()

meta_test = pd.read_csv('../input/test_set_metadata.csv')

import time

start = time.time()
chunks = 5000000

meta_test = process_meta('../input/test_set_metadata.csv')

remain_df = None
for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):

    print(i_c)

    if i_c<90:
        unique_ids = np.unique(df['object_id'])
        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df
    else:
        unique_ids = np.unique(df['object_id'])
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids)]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids)]], axis=0)

    full_test = featurize(df, meta_test, aggs,fcp)

    full_test[full_train.columns] = full_test[full_train.columns].fillna(train_mean)
    full_test_ss = ss.transform(full_test[full_train.columns])

    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(full_test_ss) / folds.n_splits
        else:
            preds += clf.predict_proba(full_test_ss) / folds.n_splits
    
   # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])
    
    # Store predictions
    preds_df = pd.DataFrame(preds, columns=class_names)
    preds_df['object_id'] = full_test['object_id']
    preds_df['class_99'] = 0.18 * preds_99 / np.mean(preds_99) 
    
    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False)
    else: 
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)
        
    del full_test, preds_df, preds
#     print('done')
    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

z = pd.read_csv('predictions.csv')

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('single_predictions.csv', index=True)
