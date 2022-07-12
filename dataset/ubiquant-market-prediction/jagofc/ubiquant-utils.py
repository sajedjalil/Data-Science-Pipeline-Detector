import numpy as np
import pandas as pd


def pearson(df, col1, col2):
    return df.corr()[col1][col2]


def metric(df, col1='prediction', col2='target'):
    assert 'time_id' in df.columns.to_list(), \
        "df must contain column named 'time_id'."
    return np.mean(df.groupby(['time_id']).apply(pearson, col1=col1, col2=col2))


class API:
    def __init__(self, df):
        dfg = df.groupby('time_id')
        self.df_iter = dfg.__iter__()
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
        time_id, df = next(self.df_iter)
        self.next_calls += 1
        data_df = df.drop(columns=['target'])
        # keep time_id in self.targets to make scoring easier:
        true_df = df.loc[:,['row_id', 'time_id', 'target']] 
        self.targets.append(true_df)
        pred_df = true_df.drop(columns=['time_id'])
        pred_df['target'] = 0.
        return data_df, pred_df
    
    def predict(self, pred_df):
        assert self.pred_calls == self.next_calls - 1, \
            "You must get the next batch of data from the API before making a new prediction."
        assert pred_df.columns.to_list() == ['row_id', 'target'], \
            "Prediction dataframe should have columns `row_id` and `target`."        
        pred_df = pred_df.astype({'row_id': 'str', 'target': np.float64})
        self.predictions.append(pred_df)
        self.pred_calls += 1
        
    def score(self):    
        pred_df = pd.concat(self.predictions).rename(columns={'target':'prediction'})
        true_df = pd.concat(self.targets)
        score_df = pd.merge(true_df, pred_df, on='row_id', how='left')
        score_df = score_df[score_df.target.isna()==False]
        if score_df.prediction.var(ddof=0) < 1e-10:
            score = -1
        else:
            score = metric(score_df)
        score_df = score_df.drop(columns=['time_id'])
        return score_df, score