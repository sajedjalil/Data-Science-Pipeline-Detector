# Detailed introduction available here https://www.kaggle.com/jagofc/detailed-local-api-introduction

import numpy as np
import pandas as pd


dtypes = {'timestamp': np.int64, 'Asset_ID': np.int8,
          'Count': np.int32,     'Open': np.float64,
          'High': np.float64,    'Low': np.float64,
          'Close': np.float64,   'Volume': np.float64,
          'VWAP': np.float64,    'Target': np.float64}

id_2_weight = {0:  4.30406509320417,
               1:  6.779921907472252,
               2:  2.3978952727983707, 
               3:  4.406719247264253,
               4:  3.555348061489413,
               5:  1.3862943611198906,
               6:  5.8944028342648505,
               7:  2.079441541679836,
               8:  1.0986122886681098,
               9:  2.3978952727983707,
               10: 1.0986122886681098,
               11: 1.6094379124341005,
               12: 2.079441541679836,
               13: 1.791759469228055}

name_2_code = {
    'Bitcoin Cash': 'BCH', 
    'Binance Coin': 'BNB',
    'Bitcoin': 'BTC',
    'EOS.IO':'EOS',
    'Ethereum Classic':'ETC',
    'Ethereum':'ETH',
    'Litecoin':'LTC',
    'Monero':'XMR',
    'TRON':'TRX',
    'Stellar':'XLM',
    'Cardano':'ADA',
    'IOTA':'MIOTA',
    'Maker':'MKR',
    'Dogecoin':'DOGE'
}


def datestring_to_timestamp(ts):
    return int(pd.Timestamp(ts).timestamp())


def read_csv_slice(file_path, dtypes=dtypes, use_window=None):
    df = pd.read_csv(file_path, dtype=dtypes)
    if use_window is not None: 
        df = df[(df.timestamp >= use_window[0]) & (df.timestamp < use_window[1])]
    return df


def weighted_correlation(a, b, weights):
    w = np.ravel(weights)
    a = np.ravel(a)
    b = np.ravel(b)
    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w
    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)
    return corr


LB_WINDOW = (datestring_to_timestamp('2021-06-13T00:00'),
             datestring_to_timestamp('2021-09-22T00:00'))


class API:
    def __init__(self, df, use_window=None):
        df = df.astype(dtypes)
        if use_window is not None: 
            df = df[(df.timestamp >= use_window[0]) & (df.timestamp <= use_window[1])]
        df['row_id'] = df.index
        dfg = df.groupby('timestamp')
        
        self.data_iter = dfg.__iter__()
        self.init_num_times = len(dfg)
        self.next_calls = 0
        self.pred_calls = 0
        self.predictions = []
        self.targets = []
        
        print("This version of the API is not optimized and should not be used to",
              "estimate the runtime of your code on the hidden test set. ;)")

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.init_num_times - self.next_calls
        
    def __next__(self):
        assert self.pred_calls == self.next_calls, \
            "You must call `predict()` successfully before you can get the next batch of data."
        timestamp, df = next(self.data_iter)
        self.next_calls += 1
        data_df = df.drop(columns=['Target'])
        true_df = df.drop(columns=['timestamp','Count','Open','High','Low','Close','Volume','VWAP'])
        true_df = true_df[['row_id', 'Target', 'Asset_ID']]
        self.targets.append(true_df)
        pred_df = true_df.drop(columns=['Asset_ID'])
        pred_df['Target'] = 0.
        return data_df, pred_df
    
    def predict(self, pred_df):
        assert self.pred_calls == self.next_calls - 1, \
            "You must get the next batch of data from the API before making a new prediction."
        assert pred_df.columns.to_list() == ['row_id', 'Target'], \
            "Prediction dataframe should have columns `row_id` and `Target`."
        pred_df = pred_df.astype({'row_id': np.int64, 'Target': np.float64})
        self.predictions.append(pred_df)
        self.pred_calls += 1
        
    def score(self, id_2_weight=id_2_weight):
        pred_df = pd.concat(self.predictions).rename(columns={'Target':'Prediction'})
        true_df = pd.concat(self.targets)
        scoring_df = pd.merge(true_df, pred_df, on='row_id', how='left')
        scoring_df['Weight'] = scoring_df.Asset_ID.map(id_2_weight)
        scoring_df = scoring_df[scoring_df.Target.isna()==False]
        if scoring_df.Prediction.var(ddof=0) < 1e-10:
            score = -1
        else:
            score = weighted_correlation(scoring_df.Prediction, scoring_df.Target, scoring_df.Weight)
        return scoring_df, score
