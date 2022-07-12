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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
#from catboost import CatBoostRegressor
from sklearn.model_selection import RepeatedKFold
from statsmodels.tsa.arima_model import ARMA
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
import seaborn as sns

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy.signal import stft
from scipy import stats
from scipy.signal import welch
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
autor = 50
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
        
        #extreme
        mm_1000 = []
        for l in range(int(np.floor(len(x)/1000))):
            mm_1000.append(x[l*1000:l*1000+1000].max())
        mm_1000 = pd.Series(mm_1000)
        c, loc, scale = gev.fit(mm_1000)
        
        self.data_result.loc[segment, 'mm_1000c'] = c
        self.data_result.loc[segment, 'mm_1000loc'] = loc
        self.data_result.loc[segment, 'mm_1000scale'] = scale
        
        mm_10000 = []
        for l in range(int(np.floor(len(x)/10000))):
            mm_10000.append(x[l*10000:l*10000+10000].max())
        mm_10000 = pd.Series(mm_10000)
        c, loc, scale = gev.fit(mm_10000)
        
        self.data_result.loc[segment, 'mm_10000c'] = c
        self.data_result.loc[segment, 'mm_10000loc'] = loc
        self.data_result.loc[segment, 'mm_10000scale'] = scale
        
        pot_0995 = pd.Series(x[x > np.quantile(x,0.995)])
        c, loc, scale = gpd.fit(pot_0995)
        self.data_result.loc[segment,'pot_995c'] =c
        self.data_result.loc[segment,'pot_995loc'] =loc
        self.data_result.loc[segment,'pot_995scale'] =scale
        
        pot_0990 = pd.Series(x[x > np.quantile(x,0.990)])
        c, loc, scale = gpd.fit(pot_0990)
        self.data_result.loc[segment,'pot_990c'] =c
        self.data_result.loc[segment,'pot_990loc'] =loc
        self.data_result.loc[segment,'pot_990scale'] =scale
        
        pot_09990 = pd.Series(x[x > np.quantile(x,0.9990)])
        c, loc, scale = gpd.fit(pot_09990)
        self.data_result.loc[segment,'pot_9990c'] =c
        self.data_result.loc[segment,'pot_9990loc'] =loc
        self.data_result.loc[segment,'pot_9990scale'] =scale
        
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
                fx, tx, zxx = stft(x,nperseg = windows)
                update('stft_f_r'+str(windows),np.real(fx))
                update('stft_t_r'+str(windows),np.real(tx))
                update('stft_z_r'+str(windows),np.real(zxx))
                update('stft_f_c'+str(windows),np.imag(fx))
                update('stft_t_c'+str(windows),np.imag(tx))
                update('stft_z_c'+str(windows),np.imag(zxx))
                
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
            for threshold in [5,10,25,50,100]:
                self.data_result.loc[segment, 'num_peak_over_'+str(threshold)] = np.sum([content > threshold for content in np.abs(x)])
        
        fw, Pww = welch(x)
        update('welch_spectrum',pd.Series(fw))
        update('welch_dens', pd.Series(Pww))
        update('sig',x)
        update2('sig',x)
        update('haar_A',cA)
        update('haar_D',cD)
        update('abs', np.abs(x))
        
        for i in range(autor):
            self.data_result.loc[segment,'autocor_'+str(i+1)] = autocor(x,i+1)
        #ARMA modelling coeff as features
        #p = 5
        #q = 5
        

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

# Set up validation strategy
rkf = RepeatedKFold(n_splits = 2, n_repeats = 1, random_state = 2019)

def train_model(X = X_train_scaled, X_test = X_test_scaled, y = y_train, par = None, model_lib = 'lgb', require_fold = False, folds = rkf,plot_feature_importance = False, model = None):
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    if require_fold == False:
        def temp_(X):
            yield X.index, X.index
        LF = temp_(X)
    else:
        LF = folds.split(X)
    for fold_, (train_index, valid_index) in enumerate(LF):
        print("Fold:", fold_)
        X_tr, X_val = X.iloc[train_index], X.iloc[valid_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[valid_index]
        if model_lib == 'lgb':
            model = lgb.LGBMRegressor(** par, n_estimators=50000)
            model.fit(X_tr, y_tr, eval_set = [(X_tr,y_tr), (X_val, y_val)], eval_metric='mae', verbose = 10000, early_stopping_rounds = 200)
            y_pred = model.predict(X_val)
            score = mean_absolute_error(y_val.values.flatten(),y_pred)
            print(f'Fold{fold_} : MAE  = {score:.4f}.')
            pred_temp = model.predict(X_test, num_iteration = model.best_iteration_)
        if model_lib == 'xgb':
            train_data = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_val, label=y_val, feature_names=X.columns)
            model = xgb.train(dtrain = train_data, num_boost_round = 20000, evals = [(train_data,'train'),(valid_data,'valid_data')], early_stopping_rounds=200, verbose_eval=500, params=par)
            y_pred = model.predict(xgb.DMatrix(X_val, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            score = mean_absolute_error(y_val.values.flatten(),y_pred)
            print(f'Fold{fold_} : MAE  = {score:.4f}.')
            pred_temp = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        if model_lib == 'sklearn':
            model = model
            model.fit(X_tr, np.ravel(y_tr))
            y_pred = np.ravel(model.predict(X_val))
            score = mean_absolute_error(np.ravel(y_val.values),y_pred)
            print(f'Fold{fold_} : MAE  = {score:.4f}.')
            pred_temp = model.predict(X_test)
        
        if model_lib == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_ + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            
        prediction = (prediction * fold_ + pred_temp) / (fold_ + 1)
        scores.append(score)
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    if model_lib == 'lgb':
        feature_importance["importance"] /= (fold_ + 1)
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        return prediction, feature_importance
    else:
        return prediction
    
    
    
lgb_param = {'num_leaves': 64,
    'objective':'huber',
    'boosting':'gbdt',
    'num_iterations': 100,
    'learning_rate' : 0.1,
    'num_threads' : -1,
    'max_depth':-1,
    'seed' : 2019,
    'min_data_in_leaf': 60,
    'bagging_fraction': 0.8,
    'bagging_freq':5,
    'bagging_seed':2019,
    'feature_fraction':0.8,
    'feature_fraction_seed':2019,
    'early_stopping_round':200,
    'metric':'mae',
    'verbosity':-1}
pred_lgb, feature_importance = train_model(par=lgb_param, model_lib='lgb', plot_feature_importance=True, require_fold = True)

xgb_params = {'eta':0.05,
    'max_depth': 8,
    'subsample':0.8,
    'objective':'reg:linear',
    'eval_metric':'mae',
    'silent': True,
    'nthread' : 4}
pred_xgb = train_model(par = xgb_params, model_lib = 'xgb',require_fold = True)

rfc = RandomForestRegressor(n_estimators = 2000, n_jobs = -1, verbose = 1)
pred_rfc = train_model(model_lib = 'sklearn',require_fold = False, model = rfc)

nSVR = NuSVR(nu = 0.7, C = 3.0, tol=0.01)
pred_svrnu = train_model(model_lib = 'sklearn',require_fold = True, model = nSVR)

SV = SVR(C = 1.0, epsilon = 0.1)
pred_svr = train_model(model_lib = 'sklearn',require_fold = True, model = SV)

neigh = KNeighborsRegressor(n_neighbors = 10)
pred_neigh = train_model(model_lib = 'sklearn',require_fold = True, model = neigh)


#Visual evaluation
#plt.figure(figsize=(6, 6))
#plt.scatter(y_train.values.flatten(), y_pred)
#plt.xlim(0, 20)
#plt.ylim(0, 20)
#plt.xlabel('actual', fontsize=12)
#plt.ylabel('predicted', fontsize=12)
#plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
#plt.show()




X_train.to_csv('train_set_features', index=False)
X_test.to_csv('test_set_features', index=False)

#submission['time_to_failure'] = pred_lgb
#submission.to_csv('submission_lgb.csv')
#submission['time_to_failure'] = pred_xgb
#submission.to_csv('submission_xgb.csv')
#submission['time_to_failure'] = pred_rfc
#submission.to_csv('submission_rfc.csv')
#submission['time_to_failure'] = pred_svrnu
#submission.to_csv('submission_nuSVR.csv')
#submission['time_to_failure'] = pred_svr
#submission.to_csv('submission_svr.csv')
#submission['time_to_failure'] = pred_neigh
#submission.to_csv('submission_neigh.csv')

submission['time_to_failure'] = (pred_neigh + pred_lgb + pred_xgb + pred_rfc + pred_svrnu + pred_svr)/6
#submission['time_to_failure'] = pred_rfc
submission.to_csv('submission_v5.csv')