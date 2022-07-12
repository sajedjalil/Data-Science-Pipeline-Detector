import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import skew, kurtosis, gmean, hmean, mode
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

#autocorrelation features
def auto_corr(data, order=1):
    """Computes the autocorrelations of a given order"""
    return np.corrcoef(data[order:],data[:-order])[0,1]

#autocovariance features
def auto_cov(data, order=1):
    """Computes the autocorrelations of a given order"""
    return np.cov(data[order:],data[:-order])[0,1]

#concentration features
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
    """Frequecy measure based on the method for fitting a sine curve from the data (cite StackOverflow article)"""
    t = np.arange(0, n_obs)
    Y = np.array(data)
    ff_t = np.fft.fftfreq(len(t), (t[1]-t[0]))
    ff_Y = abs(np.fft.fft(Y))
    return abs(ff_t[np.argmax(ff_Y[1:])+1])
    
def get_train_data(n_obs, n_skip):
      
    i = 0  
    k = 14
    max_i = int(np.floor((629145480-n_skip+1)/150000))
    y = np.zeros([max_i, ])
    X = np.zeros([max_i, k])
    
    for chunk in pd.read_csv('../input/train.csv', chunksize=n_obs, header=None, skiprows=n_skip):
        
        if (chunk.iloc[0,1] - chunk.iloc[-1,1]) < 0:
            y[i] = np.nan
            X[i,:] = np.nan
        else:
            y[i] = chunk.iloc[-1,1]
            acst_d1 = chunk[0].diff().dropna()
            acst_nomean = np.abs(chunk[0] - np.mean(chunk[0]))
            X[i,:] = [
                np.mean(chunk[0]),
                np.std(chunk[0]),
                auto_corr(chunk[0]),
                np.log(skew(chunk[0])**2),
                np.log(kurtosis(chunk[0])),
                #5
                auto_corr(acst_d1),
                np.mean(np.abs(acst_nomean)),
                gmean(np.abs(acst_nomean)),
                hmean(np.abs(acst_nomean)),
                pct_val_topn(acst_nomean, 500),
                #10
                pct_val_topn(acst_nomean, 25000),
                pct_abv(acst_nomean, n_obs, 750), 
                mode(chunk[0]).count[0]/n_obs,
                fft_freq(chunk[0],n_obs)               
            ]        
            
        i += 1
        if i == max_i:
            break
        
    return X, y
    
n_obs = 150000
n_thru = 5
skips = [int(1 + (n_obs/n_thru)*skip_i) for skip_i in range(n_thru)]
X_list = []
y_list = []

for skip in skips:
    X, y = get_train_data(n_obs, skip)
    X_list.append(X)
    y_list.append(y)

X = np.concatenate(X_list)    
k = X.shape[1]
X = X[~np.isnan(X)]
X = X.reshape(int(X.shape[0]/k), k)

y = np.concatenate(y_list)
y = y[~np.isnan(y)]

steps = [('scaler', StandardScaler()),
        ('reg', ElasticNet(alpha=0.01))]
pipeline = Pipeline(steps)
pipeline.fit(X, y)

X_test_list = []
seg_id = []
for file in os.listdir("../input/test"):
    file_df = pd.read_csv("../input/test/" + file).acoustic_data
    seg_id.append(file[:-4])
    acst_d1 = file_df.diff().dropna()
    acst_nomean = np.abs(file_df - np.mean(file_df))
    X_test_list.append(np.array([
            np.mean(file_df),
            np.std(file_df),
            auto_corr(file_df),
            np.log(skew(file_df)**2),
            np.log(kurtosis(file_df)),
            #5
            auto_corr(acst_d1),
            np.mean(np.abs(acst_nomean)),
            gmean(np.abs(acst_nomean)),
            hmean(np.abs(acst_nomean)),
            pct_val_topn(acst_nomean, 500),
            #10
            pct_val_topn(acst_nomean, 25000),
            pct_abv(acst_nomean, n_obs, 750), 
            mode(file_df).count[0]/n_obs,
            fft_freq(file_df,n_obs)
        ]))
X_test = np.concatenate(X_test_list).reshape(len(X_test_list), len(X_test_list[0]))
time_to_failure = pipeline.predict(X_test)
time_to_failure[time_to_failure<0] = 0

sub_dict = {'seg_id': seg_id, 'time_to_failure': time_to_failure}
submission = pd.DataFrame(sub_dict)
print(submission.head())
submission.to_csv('submission.csv', index=False)
