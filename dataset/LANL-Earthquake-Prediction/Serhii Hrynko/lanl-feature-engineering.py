import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cmath
from scipy.signal import hilbert, hann, convolve, find_peaks
from scipy import stats
from sklearn.linear_model import LinearRegression
from tsfresh.feature_extraction import feature_calculators as fc
import pywt

import warnings
warnings.filterwarnings("ignore")

R = 5
T = 17
hT = hann(T)

size = 150000
parts = 1
part_size = size//parts

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def maddest(x):
    return np.abs(x - x.mean()).mean()
    
def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def write_base_stats(X_tr, segment, prefix, x):
    q = np.quantile(x, [.01, .05, .5, .95, .99])
    X_tr.loc[segment, '{0}_mean'.format(prefix)] = x.mean()
    X_tr.loc[segment, '{0}_median'.format(prefix)] = q[2]
    X_tr.loc[segment, '{0}_std'.format(prefix)] = x.std()
    X_tr.loc[segment, '{0}_skew'.format(prefix)] = stats.skew(x)
    X_tr.loc[segment, '{0}_q05'.format(prefix)] = q[1]
    X_tr.loc[segment, '{0}_q95'.format(prefix)] = q[3]
    X_tr.loc[segment, '{0}_qdif'.format(prefix)] = np.subtract(q[4], q[0])
    X_tr.loc[segment, '{0}_qsum'.format(prefix)] = np.add(q[0], q[4])
    X_tr.loc[segment, '{0}_diff'.format(prefix)] = np.diff(x).mean()
    X_tr.loc[segment, '{0}_grad'.format(prefix)] = np.gradient(x).mean()
    X_tr.loc[segment, '{0}_grad2'.format(prefix)] = np.gradient(np.gradient(x)).mean()

def write_adv_stats(X_tr, segment, prefix, x):
    mean = x.mean()
    std = x.std()
    X_tr.loc[segment, '{0}_atrend'.format(prefix)] = add_trend_feature(x, abs_values=True)
    X_tr.loc[segment, '{0}_energy'.format(prefix)] = fc.abs_energy(x)
    X_tr.loc[segment, '{0}_sum_of_changes'.format(prefix)] = fc.absolute_sum_of_changes(x)
    X_tr.loc[segment, '{0}_mean_abs_change'.format(prefix)] = fc.mean_abs_change(x)
    X_tr.loc[segment, '{0}_mean_change'.format(prefix)] = fc.mean_change(x)
    X_tr.loc[segment, '{0}_range_noise05_count'.format(prefix)] = fc.range_count(x, mean-std/2, mean+std/2)
    X_tr.loc[segment, '{0}_range_noise_count'.format(prefix)] = fc.range_count(x, mean-std, mean+std)
    X_tr.loc[segment, '{0}_range_noise2_count'.format(prefix)] = fc.range_count(x, mean-std*2, mean+std*2)
    X_tr.loc[segment, '{0}_ratio_unique_values'.format(prefix)] = fc.ratio_value_number_to_time_series_length(x)
    X_tr.loc[segment, '{0}_time_rev_asym_stat_T'.format(prefix)] = fc.time_reversal_asymmetry_statistic(x, T)
    X_tr.loc[segment, '{0}_time_rev_asym_stat_5T'.format(prefix)] = fc.time_reversal_asymmetry_statistic(x, 5*T)
    X_tr.loc[segment, '{0}_autocorrelation'.format(prefix)] = np.correlate(x, x)[0]
    X_tr.loc[segment, '{0}_autocorrelation_R'.format(prefix)] = fc.autocorrelation(x, R)
    X_tr.loc[segment, '{0}_autocorrelation_T'.format(prefix)] = fc.autocorrelation(x, T)
    X_tr.loc[segment, '{0}_c3_R'.format(prefix)] = fc.c3(x, R)
    X_tr.loc[segment, '{0}_c3_T'.format(prefix)] = fc.c3(x, T)
    X_tr.loc[segment, '{0}_long_strk_above_mean'.format(prefix)] = fc.longest_strike_above_mean(x)
    X_tr.loc[segment, '{0}_long_strk_below_mean'.format(prefix)] = fc.longest_strike_below_mean(x)
    X_tr.loc[segment, '{0}_cid_ce_1'.format(prefix)] = fc.cid_ce(x, 1)
    X_tr.loc[segment, '{0}_binned_entropy_R'.format(prefix)] = fc.binned_entropy(x, R)
    X_tr.loc[segment, '{0}_binned_entropy_T'.format(prefix)] = fc.binned_entropy(x, T)
    X_tr.loc[segment, '{0}_num_crossing_mean'.format(prefix)] = fc.number_crossing_m(x, mean)
    X_tr.loc[segment, '{0}_num_peaks_2'.format(prefix)] = fc.number_peaks(x, 2)
    X_tr.loc[segment, '{0}_num_peaks_R'.format(prefix)] = fc.number_peaks(x, R)
    X_tr.loc[segment, '{0}_num_peaks_T'.format(prefix)] = fc.number_peaks(x, T)
    X_tr.loc[segment, '{0}_spkt_welch_density_1'.format(prefix)] = list(fc.spkt_welch_density(x, [{'coeff': 1}]))[0][1]
    X_tr.loc[segment, '{0}_spkt_welch_density_T'.format(prefix)] = list(fc.spkt_welch_density(x, [{'coeff': T}]))[0][1]
    X_tr.loc[segment, '{0}_spkt_welch_density_5T'.format(prefix)] = list(fc.spkt_welch_density(x, [{'coeff': 5*T}]))[0][1]
    X_tr.loc[segment, '{0}_time_rev_asym_stat_1'.format(prefix)] = fc.time_reversal_asymmetry_statistic(x, 1)
    X_tr.loc[segment, '{0}_time_rev_asym_stat_T'.format(prefix)] = fc.time_reversal_asymmetry_statistic(x, T)
    X_tr.loc[segment, '{0}_time_rev_asym_stat_5T'.format(prefix)] = fc.time_reversal_asymmetry_statistic(x, 5*T)
    
def write_abs(X_tr, segment, prefix, x):
    write_base_stats(X_tr, segment, prefix, x)
    xa = np.abs(x - x.mean())
    write_base_stats(X_tr, segment, '{0}_abs'.format(prefix), xa)

def write_adv_abs(X_tr, segment, prefix, x):
    write_adv_stats(X_tr, segment, prefix, x)
    xa = np.abs(x - x.mean())
    write_adv_stats(X_tr, segment, '{0}_abs'.format(prefix), xa)

def write_hilbert(X_tr, segment, prefix, x):
    mean = x.mean()
    x0 = x - mean
    write_abs(X_tr, segment, prefix, x)
    write_adv_abs(X_tr, segment, prefix, x)
    
    analytic_signal = hilbert(x0)
    env = np.abs(analytic_signal)
    write_base_stats(X_tr, segment, '{0}_env'.format(prefix), env)

    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    ifreq = np.abs((np.diff(instantaneous_phase, 2) / (2.0*np.pi)))
    write_abs(X_tr, segment, '{0}_ifreq'.format(prefix), ifreq)

def write_hann(X_tr, segment, prefix, x):
    write_hilbert(X_tr, segment, prefix, x)
    
    y2 = np.median(rolling_window(x, 3), axis=-1)
    write_hilbert(X_tr, segment, '{0}_rW'.format(prefix), y2)
    
    xd = denoise_signal(y2)
    write_hilbert(X_tr, segment, '{0}_denoise'.format(prefix), xd)
    
    zT = convolve(x, hT, mode='same') / sum(hT)
    write_hilbert(X_tr, segment, '{0}_hT'.format(prefix), zT)

def write_features(X_tr, segment, x):
    for i in range(parts):
        write_hann(X_tr, segment, i+1, x[i*part_size:(i+1)*part_size])

submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(dtype=np.float64, index=submission.index)
for seg_id in tqdm(submission.index):
    seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')

    x = seg['acoustic_data']
    write_features(X_test, seg_id, x)

X_test.to_csv('test_features.csv', index=True)

train = pd.read_csv('../input/lanl-training-acoustic-data/train.csv', dtype={'acoustic_data': np.int16}, usecols=[0])
train.head()

segments = (train.shape[0]) // size

X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)

start = 0
for segment in tqdm(range(segments)):
    seg = train.iloc[start:start+size]
    start += size

    x = seg['acoustic_data']
    write_features(X_tr, segment, x)

X_tr.to_csv('train_features.csv', index=False)
