# LANL MODULE: my user defined functions

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, gmean, hmean, mode

# FEATURE GENERATING FUNCTIONS

def auto_corr(data, order=1):
    """Computes the autocorrelation of a given order"""
    return np.corrcoef(data[order:],data[:-order])[0,1]

def auto_cov(data, order=1):
    """Computes the autocovariance of a given order"""
    return np.cov(data[order:],data[:-order])[0,1]

def pct_in_range(data, n_obs, low, high):
    """Proportion of observations that fall between the low and high tresholds"""
    return np.sum((data >= low) & (data < high))/n_obs
   
def pct_abv(data, n_obs, bound):
    """Proportion of observations above a given bound"""
    return np.sum(data >= bound)/n_obs

def pct_val_topn(data, n):
    """Proportion of total value/mass in n highest observations"""
    data_sort = np.sort(np.abs(data))
    return np.sum(data_sort[-n:])/np.sum(data_sort)

def fft_freq(data, n_obs):
    """Frequecy measure based on the method for fitting a sine curve from the data"""
    t = np.arange(0, n_obs)
    Y = np.array(data)
    ff_t = np.fft.fftfreq(len(t), (t[1]-t[0]))
    ff_Y = abs(np.fft.fft(Y))
    return abs(ff_t[np.argmax(ff_Y[1:])+1])

# FEATURE VECTOR GENERATOR

def gen_feature_vector(df, n_obs):
    """Generates a vector of 14 features"""
    df_d1 = df.diff().dropna()
    df_abs_dev = np.abs(df - np.mean(df))
    return [
        np.mean(df),
        np.std(df),
        auto_corr(df),
        np.log(skew(df)**2),
        np.log(kurtosis(df)),
        auto_corr(df_d1),
        np.mean(np.abs(df_abs_dev)),
        gmean(np.abs(df_abs_dev)),
        hmean(np.abs(df_abs_dev)),
        pct_val_topn(df_abs_dev, 500),
        pct_val_topn(df_abs_dev, 25000),
        pct_abv(df_abs_dev, n_obs, 750), 
        mode(df).count[0]/n_obs,
        fft_freq(df,n_obs)               
    ]  

# TRAINING DATA GENERATOR

def gen_training_data(data_path, n_obs=150000, n_thru=5, k=14):
    """Generates training data for LANL"""
        
    skips = [int(1 + (n_obs/n_thru)*skip_i) for skip_i in range(n_thru)]
    
    X_list = []
    y_list = [] 

    for n_skip in skips:

        i = 0 # row index counter
        # 629145480 is total number of rows in dataset
        # max_i is the number of complete chunks that can be extracted
        max_i = int(np.floor((629145480-n_skip+1)/n_obs))
        # set up zero arrays for X and y training data
        X = np.zeros([max_i, k])
        y = np.zeros([max_i, ])    

        for chunk in pd.read_csv(data_path,
                                 chunksize=n_obs, 
                                 header=None, 
                                 skiprows=n_skip): 
            # any chunks with fewer than 150000 rows are ignored
            if chunk.shape[0] < n_obs:
                pass
            # any chunks where an earthquake event occurs are set as NaN
            # these are identified by a 'reset' in time_to_failure
            # (i.e. time_to_failure is higher at the end than at the start)
            elif (chunk.iloc[0,1] - chunk.iloc[-1,1]) < 0:
                y[i] = np.nan
                X[i,:] = np.nan
                i +=1
            # everything else is collected for the training dataset
            else:
                y[i] = chunk.iloc[-1,1] # last time_to_failure obs
                X[i, :] = gen_feature_vector(chunk[0], n_obs)
                i += 1

        y_list.append(y)    
        X_list.append(X)       
    
    # X training data
    X_train = np.concatenate(X_list)    
    k = X_train.shape[1]
    X_train = X_train[~np.isnan(X_train)]
    X_train = X_train.reshape(int(X_train.shape[0]/k), k)
    
    # y training data
    y_train = np.concatenate(y_list)
    y_train = y_train[~np.isnan(y_train)]
    
    return X_train, y_train
    
# TEST DATA GENERATOR

def gen_test_data(data_path):
    """Generates test data from LANL test file segments"""
    X_test_list = []
    seg_id = []
    for file in os.listdir(data_path):
        file_df = pd.read_csv(os.path.join(data_path,file))
        seg_id.append(file[:-4])
        X_test_list.append(gen_feature_vector(file_df.acoustic_data, file_df.shape[0]))
    X_test = np.concatenate(X_test_list).reshape(len(X_test_list), len(X_test_list[0]))
    return seg_id, X_test
    
    
    
    
    
    