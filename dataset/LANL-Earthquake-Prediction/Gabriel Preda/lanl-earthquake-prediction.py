#packages
import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from sklearn.svm import NuSVR, SVR
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from scipy.signal import hann, hilbert, convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
#settings
warnings.filterwarnings('ignore')
np.random.seed(2019)
#logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def read_train_data(nrows=None):
    #load data
    logger.info('Start read data')
    PATH="../input/"
    train_df = pd.read_csv(PATH+'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},nrows=nrows)

    return train_df

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


def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    
    '''
    zc = np.fft.fft(xc)
    #X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    
    #FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    #X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()

    #X.loc[seg_id, 'Rmean_last_2500'] = realFFT[-2500:].mean()
    X.loc[seg_id, 'Rstd__last_2500'] = realFFT[-2500:].std()
    X.loc[seg_id, 'Rmax_last_2500'] = realFFT[-2500:].max()
    X.loc[seg_id, 'Rmin_last_2500'] = realFFT[-2500:].min()    
    #X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()
    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()
    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()
    #X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()
    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()
    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()
       
    #X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))
    #X.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])
    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()
    X.loc[seg_id, 'abs_min'] = np.abs(xc).min()
    
    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()
    
    #X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()
    #X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()
    #X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()
    #X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()
    
    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()
    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()
    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()
    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()
    
    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()
    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()
    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()
    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()
    
    X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())
    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])
    X.loc[seg_id, 'sum'] = xc.sum()
    
    #X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])
    #X.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])
    #X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])
    #X.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])
    
    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)
    
    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)
    
    X.loc[seg_id, 'trend'] = add_trend_feature(xc)
    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)
    #X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()
    
    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()
    
    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()
    #X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()
    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()
    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()
    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()
    #X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
    #X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
    #X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
    #X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    #X.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)
    #X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    #X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
    no_of_std = 2
    #X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
    #X.loc[seg_id,'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    #X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    #X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    #X.loc[seg_id,'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    #X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    #X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()
    
    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
    X.loc[seg_id, 'q999'] = np.quantile(xc,0.999)
    X.loc[seg_id, 'q001'] = np.quantile(xc,0.001)
    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)
    '''
    for windows in [10, 32, 100, 316, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values
        
        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    
def feature_engineering(train_df):
     #features engineering
    logger.info('Features engineering')
    rows = 150000
    segments = int(np.floor(train_df.shape[0] / rows))
    print("Number of segments: ", segments)
    train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
    train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
    total_mean = train_df['acoustic_data'].mean()
    total_std = train_df['acoustic_data'].std()
    total_max = train_df['acoustic_data'].max()
    total_min = train_df['acoustic_data'].min()
    total_sum = train_df['acoustic_data'].sum()
    total_abs_sum = np.abs(train_df['acoustic_data']).sum()
    # process train data
    logger.info('Process train data')
    for seg_id in range(segments):
        seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
        create_features(seg_id, seg, train_X)
        train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    # process test data
    logger.info('Process test data')
    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
    for seg_id in test_X.index:
        seg = pd.read_csv('../input/test/' + seg_id + '.csv')
        create_features(seg_id, seg, test_X)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
    del train_df
    gc.collect()
    return scaled_train_X, scaled_test_X, train_y, submission

# process test data
def run_model(scaled_train_X, scaled_test_X, train_y):
    logger.info('Run model')
    n_fold = 9
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    train_columns = scaled_train_X.columns.values
    params = {'num_leaves': 65, 'min_data_in_leaf': 12, 
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.001,
             "boosting": "gbdt",
             "feature_fraction": 0.81,"bagging_freq": 1, "bagging_fraction": 0.81, "bagging_seed": 42,
             "metric": 'mae',
             "lambda_l1": 0.1,
             "verbose": -1, "nthread": -1,  "random_state": 44000}
    oof = np.zeros(len(scaled_train_X))
    predictions = np.zeros(len(scaled_test_X))
    
    #run model
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X,train_y.values)):
        strLog = "fold {}".format(fold_)
        print(strLog)
        logger.info(strLog)
        
        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
    
        model = lgb.LGBMRegressor(**params, n_estimators = 20000)
        model.fit(X_tr, y_tr, 
                        eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                        verbose=1000, 
                        early_stopping_rounds=1000)
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
        #predictions
        predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits
    strLog = "CV score: {}".format(np.sqrt(mean_squared_error(train_y.values, oof)))
    print(strLog)
    logger.info(strLog)
    return predictions

def submit(submission, predictions):
    # submission
    logger.info('Submisison')
    submission.time_to_failure = predictions
    submission.to_csv('submission.csv',index=True)

def main(nrows=None):
    train_df = read_train_data(nrows)
    scaled_train_X, scaled_test_X, train_y, submission = feature_engineering(train_df)
    predictions = run_model(scaled_train_X, scaled_test_X, train_y)
    submit(submission, predictions)
    
if __name__ == "__main__":
    main()