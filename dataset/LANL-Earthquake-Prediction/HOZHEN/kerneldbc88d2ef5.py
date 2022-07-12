# -*- coding: utf-8 -*-
"""
Created on Thu May 16 07:55:58 2019

@author: HO Zhen Wai Olivier
"""

import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

import os
import random
import pywt
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
#import lightgbm as lgb
#import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

import seaborn as sns

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings("ignore")
train.head()

pd.options.display.precision = 15

#def plot_acc_ttf_data(train_ad_sample100, train_ttf_sample100, title = "Acoustic data vs time to failure: 1% sample"):
    #fig,ax1 = plt.subplots(figsize=(12,8))
    #plt.title(title)
    #plt.plot(train_ad_sample100, color = 'r')
    #ax1.set_ylabel('acoustic data', color = 'r')
    #plt.legend(['acoustic data'], loc=(0.01, 0.95))
    #ax2 = ax1.twinx()
    #plt.plot(train_ttf_sample100, color='b')
    #ax2.set_ylabel('time to failure', color='b')
    #plt.legend(['time to failure'], loc=(0.01, 0.9))
    #plt.grid(True)

#train_ad_sample100 = train['acoustic_data'].values[::100]
#train_ttf_sample100 = train['time_to_failure'].values[::100]

#plot_acc_ttf_data(train_ad_sample100, train_ttf_sample100)
#del train_ad_sample100
#del train_ttf_sample100
#plot_acc_ttf_data(train['acoustic_data'].values[:rows],train['time_to_failure'].values[:rows],"Acoustic data vs time to failure : 1st segment")


#feature generation


rows = 150000
segments = int(np.floor(train.shape[0] / rows))
bonussample = 500

def autocor(s,t=1):
    result = np.corrcoef(s[t:],s[:-t])[0,1]
    return result

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]
    
def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta

def calc_change_rate(x):
    change = (np.diff(x) / x[:-1])
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


#name = []
#for i in ['sig','abssig','haar_A','haar_D','db_A','db_D','bior_A','bior_D']:
    #for j in ['ave','std','max','min','kurt','quan99','quan50','quan75','meanlast_5000','meanfirst_5000','stdlast_5000','stdfirst_5000','countbig','classic_stlta1','classic_stlta2','classic_stlta3']:
        #name.append(i+'_'+j)
#name += ['autocor_'+str(i+1) for i in range(autor)] + ['trend','abstrend','moving_average_3000_mean','maxtomindiff']
autor = 200
X_train = pd.DataFrame(index=range(segments+bonussample), dtype=np.float64)
y_train = pd.DataFrame(index=range(segments+bonussample), dtype=np.float64,
                       columns=['time_to_failure'])

total_mean = train['acoustic_data'].mean()
total_std = train['acoustic_data'].std()
total_max = train['acoustic_data'].max()
total_min = train['acoustic_data'].min()
total_sum = train['acoustic_data'].sum()
total_abs_sum = np.abs(train['acoustic_data']).sum()

class Engine(object):
    def __init__(self,data_frame_init,data_frame_result, bonus = False, training = False):
        self.data_init = data_frame_init
        self.data_result = data_frame_result
        self.bonus = bonus
        self.training = training
    def __call__(self,segment):
        if self.training == True:
            if self.bonus == False:
                seg = self.data_init.iloc[segment*rows:segment*rows+rows]
                x = pd.Series(seg['acoustic_data'].values)
            else:
                loca = random.randint(0,int(self.data_init.shape[0])-rows)
                seg = self.data_init.iloc[loca:loca+rows]
                #noise = np.random.normal(0,1,rows)
                x = pd.Series(seg['acoustic_data'].values)# + noise)
            y = seg['time_to_failure'].values[-1]
            y_train.loc[segment, 'time_to_failure'] = y
        else:
            seg = pd.read_csv('../input/test/' +segment + '.csv')
            x = pd.Series(seg['acoustic_data'].values)
        
        cA, cD = pywt.dwt(x,'haar')
        LFHT = [0 if np.abs(a) >10 else a for a in cA]
        LFLT = [0 if np.abs(a) <10 else a for a in cA]
        HFHT = [0 if np.abs(b) >7 else b for b in cD]
        HFLT = [0 if np.abs(b) <7 else b for b in cD]
        
        cAA,cDD = pywt.dwt(x,'db1')
        cAAA, cDDD = pywt.dwt(x,'bior1.1')
        def update(name,s):
            self.data_result.loc[segment, name+'_ave'] = s.mean()
            self.data_result.loc[segment, name+'_std'] = s.std()
            self.data_result.loc[segment, name+'_max'] = s.max()
            self.data_result.loc[segment, name+'_min'] = s.min()
            self.data_result.loc[segment, name+'_quan95'] = np.quantile(s,0.95)
            self.data_result.loc[segment, name+'_quan99'] = np.quantile(s,0.99)
            self.data_result.loc[segment, name+'_quan05'] = np.quantile(s,0.05)
            self.data_result.loc[segment, name+'_quan01'] = np.quantile(s,0.01)
            self.data_result.loc[segment, name+'_med'] = np.quantile(s,0.5)
        def update2(name, x):
            self.data_result.loc[segment,name+ 'mean_change_abs'] = np.mean(np.diff(x))
            self.data_result.loc[segment,name+ 'mean_change_rate'] = calc_change_rate(x)
        
            self.data_result.loc[segment,name+ 'std_first_50000'] = x[:50000].std()
            self.data_result.loc[segment,name+ 'std_last_50000'] = x[-50000:].std()
            self.data_result.loc[segment,name+ 'std_first_10000'] = x[:10000].std()
            self.data_result.loc[segment,name+ 'std_last_10000'] = x[-10000:].std()
        
            self.data_result.loc[segment,name+ 'avg_first_50000'] = x[:50000].mean()
            self.data_result.loc[segment,name+ 'avg_last_50000'] = x[-50000:].mean()
            self.data_result.loc[segment,name+ 'avg_first_10000'] = x[:10000].mean()
            self.data_result.loc[segment,name+ 'avg_last_10000'] = x[-10000:].mean()
        
            self.data_result.loc[segment,name+ 'min_first_50000'] = x[:50000].min()
            self.data_result.loc[segment,name+ 'min_last_50000'] = x[-50000:].min()
            self.data_result.loc[segment,name+ 'min_first_10000'] = x[:10000].min()
            self.data_result.loc[segment,name+ 'min_last_10000'] = x[-10000:].min()
        
            self.data_result.loc[segment,name+ 'max_to_min'] = x.max() / np.abs(x.min())
            self.data_result.loc[segment,name+ 'max_to_min_diff'] = x.max() - np.abs(x.min())
            self.data_result.loc[segment,name+ 'count_big'] = len(x[np.abs(x) > 500])
            self.data_result.loc[segment,name+ 'sum'] = x.sum()
        
            self.data_result.loc[segment,name+ 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000].values)
            self.data_result.loc[segment,name+ 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:].values)
            self.data_result.loc[segment,name+ 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000].values)
            self.data_result.loc[segment,name+ 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:].values)
        
            self.data_result.loc[segment,name+ 'trend'] = add_trend_feature(x)
            self.data_result.loc[segment,name+ 'abs_trend'] = add_trend_feature(x, abs_values=True)
            self.data_result.loc[segment,name+ 'kurt'] = x.kurtosis()
            self.data_result.loc[segment,name+ 'mad'] = x.mad()
            self.data_result.loc[segment,name+ 'skew'] = x.skew()
        
            self.data_result.loc[segment,name+ 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
            self.data_result.loc[segment,name+ 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
            self.data_result.loc[segment,name+ 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
            self.data_result.loc[segment,name+ 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
            self.data_result.loc[segment,'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
            ewma = pd.Series.ewm
            self.data_result.loc[segment,name+ 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
            self.data_result.loc[segment,name+ 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
            self.data_result.loc[segment,name+ 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
            no_of_std = 3
            self.data_result.loc[segment,name+'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
            self.data_result.loc[segment,name+'MA_700MA_BB_high_mean'] = (self.data_result.loc[segment, 'Moving_average_700_mean'] + no_of_std * self.data_result.loc[segment, name+'MA_700MA_std_mean']).mean()
            self.data_result.loc[segment,name+'MA_700MA_BB_low_mean'] = (self.data_result.loc[segment, 'Moving_average_700_mean'] - no_of_std * self.data_result.loc[segment, name+'MA_700MA_std_mean']).mean()
            self.data_result.loc[segment,name+'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
            self.data_result.loc[segment,name+'MA_400MA_BB_high_mean'] = (self.data_result.loc[segment, 'Moving_average_700_mean'] + no_of_std * self.data_result.loc[segment, name+'MA_400MA_std_mean']).mean()
            self.data_result.loc[segment,name+'MA_400MA_BB_low_mean'] = (self.data_result.loc[segment, 'Moving_average_700_mean'] - no_of_std * self.data_result.loc[segment, name+'MA_400MA_std_mean']).mean()
            self.data_result.loc[segment,name+'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
            self.data_result.drop('Moving_average_700_mean', axis=1, inplace=True)
    
            self.data_result.loc[segment,name+'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
            self.data_result.loc[segment,name+'q999'] = np.quantile(x,0.999)
            self.data_result.loc[segment,name+'q001'] = np.quantile(x,0.001)
            self.data_result.loc[segment,name+'ave10'] = stats.trim_mean(x, 0.1)
            for windows in [10, 100, 1000]:
                x_roll_std = x.rolling(windows).std().dropna().values
                x_roll_mean = x.rolling(windows).mean().dropna().values
        
                self.data_result.loc[segment,name+ 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
                self.data_result.loc[segment,name+ 'std_roll_std_' + str(windows)] = x_roll_std.std()
                self.data_result.loc[segment,name+ 'max_roll_std_' + str(windows)] = x_roll_std.max()
                self.data_result.loc[segment,name+ 'min_roll_std_' + str(windows)] = x_roll_std.min()
                self.data_result.loc[segment,name+ 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
                self.data_result.loc[segment,name+ 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
                self.data_result.loc[segment,name+ 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
                self.data_result.loc[segment,name+ 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
                self.data_result.loc[segment,name+ 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                self.data_result.loc[segment,name+ 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                self.data_result.loc[segment,name+ 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
                self.data_result.loc[segment,name+ 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                self.data_result.loc[segment,name+ 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
                self.data_result.loc[segment,name+ 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
                self.data_result.loc[segment,name+ 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
                self.data_result.loc[segment,name+ 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
                self.data_result.loc[segment,name+ 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
                self.data_result.loc[segment,name+ 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
                self.data_result.loc[segment,name+ 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
                self.data_result.loc[segment,name+ 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                self.data_result.loc[segment,name+ 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                self.data_result.loc[segment,name+ 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
        
        update('sig',x)
        update('haar_A',cA)
        update('haar_D',cD)
        update('db_A',cAA)
        update('db_D',cDD)
        update('bior_A',cAAA)
        update('bior_D',cDDD)
        update('abs', np.abs(x))
        update2('sig',x)
        update2('HT',pd.Series(pywt.idwt(LFHT,HFHT,'haar')))
        update2('LT',pd.Series(pywt.idwt(LFLT,HFLT,'haar')))
        update2('HTLT',pd.Series(pywt.idwt(HFHT,HFLT,'haar')))
        update2('LTHT',pd.Series(pywt.idwt(HFLT,HFHT,'haar')))
        for i in range(autor):
            self.data_result.loc[segment,'autocor_'+str(i+1)] = autocor(x,i+1)

# Apply helper functions to data
engine = Engine(train,X_train,bonus = False, training = True)
for i in tqdm(range(segments)):
    engine(i)
engine = Engine(train, X_train, bonus = True, training = True)
for i in tqdm(range(segments, segments + bonussample)):
    engine(i)

print(f'{X_train.shape[0]} samples in new train data and {X_train.shape[1]} potential features.')
#Features visualisation (the more correlation to the objective)
#plt.figure(figsize=(44,24))
#cols = list(np.abs(X_train.corrwith(y_train['time_to_failure'])).sort_values(ascending = False).head(24).index)
#for i, col in enumerate(cols):
#    plt.subplot(6,4,i+1)
#    plt.plot(X_train[col], color = 'blue')
#    plt.title(col)
#    ax.set_ylabel(col, color='b')
    
#    ax2 = ax1.twinx()
#    plt.plot(y_train, color = 'r')
#    ax.set_ylabel('time_to_failure', color='g')
#    plt.legend([col, 'time_to_failure'], loc = (0.1,0.9))
#   plt.grid(False)
#Can we create more features based on the correlated features?

#Data cleaning
means_dict = {}
for col in X_train.columns:
    if X_train[col].isnull().any():
        print(col)
        mean_value = X_train.loc[X_train[col] != -np.inf, col].mean()
        X_train.loc[X_train[col] == -np.inf, col] = mean_value
        X_train[col] = X_train[col].fillna(mean_value)
        means_dict[col] = mean_value
#Read test data and compute feature
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
engine = Engine(None, X_test)
for seg_id in tqdm(X_test.index):
    engine(seg_id)

#Data cleaning
print(f'{X_test.shape[0]} new data for testing.' )
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
        X_test[col] = X_test[col].fillna(means_dict[col])


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)


#Models 



rfc = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, verbose = 1)
rfc.fit(X_train_scaled,y_train.values.flatten())
y_pred = rfc.predict(X_train_scaled)

#Disp feature importance


#Visual evaluation
#plt.figure(figsize=(6, 6))
#plt.scatter(y_train.values.flatten(), y_pred)
#plt.xlim(0, 20)
#plt.ylim(0, 20)
#plt.xlabel('actual', fontsize=12)
#plt.ylabel('predicted', fontsize=12)
#plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
#plt.show()

#Result score
score = mean_absolute_error(y_train.values.flatten(), np.ravel(y_pred))
print(f'Score: {score:0.3f}')




submission['time_to_failure'] = rfc.predict(X_test_scaled)
submission.to_csv('submission.csv')