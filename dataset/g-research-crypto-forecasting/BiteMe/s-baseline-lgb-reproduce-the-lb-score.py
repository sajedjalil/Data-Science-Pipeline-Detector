# %% [markdown]
# # Import and load dfs
# 
# References: [Tutorial to the G-Research Crypto Competition](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition)

# %% [markdown]
# ### I think I managed to reproduce the LB socre using the code given by the organizer.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:51:06.35286Z","iopub.execute_input":"2021-12-23T05:51:06.35356Z","iopub.status.idle":"2021-12-23T05:51:08.636529Z","shell.execute_reply.started":"2021-12-23T05:51:06.353431Z","shell.execute_reply":"2021-12-23T05:51:08.635671Z"}}
import pandas as pd
import numpy as np
import time
from lightgbm import LGBMRegressor
import gresearch_crypto


TRAIN_CSV = '../input/allfeatherdata/transformed_train.feather'
ASSET_DETAILS_CSV = '../input/g-research-crypto-forecasting/asset_details.csv'

def read_csv_strict(file_name):
    df = pd.read_feather(file_name)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
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

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:54:15.322762Z","iopub.execute_input":"2021-12-23T05:54:15.323447Z","iopub.status.idle":"2021-12-23T05:54:33.419079Z","shell.execute_reply.started":"2021-12-23T05:54:15.323401Z","shell.execute_reply":"2021-12-23T05:54:33.418141Z"}}
!pip install -U scikit-learn

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:54:38.613162Z","iopub.execute_input":"2021-12-23T05:54:38.613602Z","iopub.status.idle":"2021-12-23T05:54:38.639507Z","shell.execute_reply.started":"2021-12-23T05:54:38.613557Z","shell.execute_reply":"2021-12-23T05:54:38.638201Z"}}
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:08:13.123733Z","iopub.execute_input":"2021-12-23T05:08:13.12459Z","iopub.status.idle":"2021-12-23T05:08:44.174454Z","shell.execute_reply.started":"2021-12-23T05:08:13.124532Z","shell.execute_reply":"2021-12-23T05:08:44.173337Z"}}
df_train = read_csv_strict(TRAIN_CSV)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:09:12.975693Z","iopub.execute_input":"2021-12-23T05:09:12.975989Z","iopub.status.idle":"2021-12-23T05:09:13.014006Z","shell.execute_reply.started":"2021-12-23T05:09:12.975959Z","shell.execute_reply":"2021-12-23T05:09:13.012892Z"}}
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
df_asset_details

# %% [markdown]
# # Training

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T01:40:34.314332Z","iopub.execute_input":"2021-12-23T01:40:34.314662Z","iopub.status.idle":"2021-12-23T01:40:34.331672Z","shell.execute_reply.started":"2021-12-23T01:40:34.31463Z","shell.execute_reply":"2021-12-23T01:40:34.330672Z"}}
# Two new features from the competition tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df, row=False):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    
    
    df_feat["Close/Open"] = df_feat["Close"] / df_feat["Open"] 
    df_feat["Close-Open"] = df_feat["Close"] - df_feat["Open"] 
    df_feat["High-Low"] = df_feat["High"] - df_feat["Low"] 
    df_feat["High/Low"] = df_feat["High"] / df_feat["Low"]
    if row:
        df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean()
    else:
        df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    
    df_feat['High/Mean'] = df_feat['High'] / df_feat['Mean']
    df_feat['Low/Mean'] = df_feat['Low'] / df_feat['Mean']
    df_feat['Volume/Count'] = df_feat['Volume'] / (df_feat['Count'] + 1)

    ## possible seasonality, datetime  features (unlikely to me meaningful, given very short time-frames)
    ### to do: add cyclical features for seasonality
    times = pd.to_datetime(df["timestamp"],unit="s",infer_datetime_format=True)
    if row:
        df_feat["hour"] = times.hour  # .dt
        df_feat["dayofweek"] = times.dayofweek 
        df_feat["day"] = times.day 
    else:
        df_feat["hour"] = times.dt.hour  # .dt
        df_feat["dayofweek"] = times.dt.dayofweek 
        df_feat["day"] = times.dt.day 
    #df_feat.drop(columns=["time"],errors="ignore",inplace=True)  # keep original epoch time, drop string

    return df_feat


def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]

    # TODO: Try different models here!
    model = LGBMRegressor(n_estimators=10)
    model.fit(X, y)
    return X, y, model

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:54:48.409441Z","iopub.execute_input":"2021-12-23T05:54:48.40981Z","iopub.status.idle":"2021-12-23T05:54:48.560431Z","shell.execute_reply.started":"2021-12-23T05:54:48.409771Z","shell.execute_reply":"2021-12-23T05:54:48.559337Z"}}
from sklearn.model_selection import TimeSeriesSplit
test_day = 60
gap_day = 1
cv = TimeSeriesSplit(n_splits=5, test_size=test_day * 24 * 60, gap=gap_day * 24 * 60)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:55:09.85925Z","iopub.execute_input":"2021-12-23T05:55:09.859875Z","iopub.status.idle":"2021-12-23T05:55:09.866616Z","shell.execute_reply.started":"2021-12-23T05:55:09.859825Z","shell.execute_reply":"2021-12-23T05:55:09.865889Z"}}
import sklearn
sklearn.__version__

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:48:59.844585Z","iopub.execute_input":"2021-12-23T05:48:59.844886Z","iopub.status.idle":"2021-12-23T05:49:15.56746Z","shell.execute_reply.started":"2021-12-23T05:48:59.844852Z","shell.execute_reply":"2021-12-23T05:49:15.566464Z"}}
for asset_id in range(14):
    asset_df = train_df[train_df.Test]

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:13:17.355708Z","iopub.execute_input":"2021-12-23T05:13:17.356313Z","iopub.status.idle":"2021-12-23T05:13:21.681024Z","shell.execute_reply.started":"2021-12-23T05:13:17.356268Z","shell.execute_reply":"2021-12-23T05:13:21.67993Z"}}
for i in range(14):
    print(f"datasize for id={i}: {df_train[df_train.Asset_ID == i].shape[0]}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T05:13:48.589977Z","iopub.execute_input":"2021-12-23T05:13:48.590286Z","iopub.status.idle":"2021-12-23T05:13:48.910015Z","shell.execute_reply.started":"2021-12-23T05:13:48.590256Z","shell.execute_reply":"2021-12-23T05:13:48.909165Z"}}


# %% [markdown]
# ## Loop over all assets

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T01:40:35.181741Z","iopub.execute_input":"2021-12-23T01:40:35.182047Z","iopub.status.idle":"2021-12-23T01:41:28.696377Z","shell.execute_reply.started":"2021-12-23T01:40:35.182014Z","shell.execute_reply":"2021-12-23T01:41:28.695634Z"}}
Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T01:27:50.651002Z","iopub.execute_input":"2021-12-23T01:27:50.652038Z","iopub.status.idle":"2021-12-23T01:27:50.67603Z","shell.execute_reply.started":"2021-12-23T01:27:50.651997Z","shell.execute_reply":"2021-12-23T01:27:50.674973Z"}}
# Check the model interface
x = get_features(df_train.iloc[1], row=True)
y_pred = models[0].predict([x])
y_pred[0]

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T01:43:29.008986Z","iopub.execute_input":"2021-12-23T01:43:29.009844Z","iopub.status.idle":"2021-12-23T01:43:31.30267Z","shell.execute_reply.started":"2021-12-23T01:43:29.009799Z","shell.execute_reply":"2021-12-23T01:43:31.301793Z"}}
y_true = df_test["Target"]
y_pred = pd.Series(np.nan, index=y_true.index)
for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    df_asset = df_test[df_test.Asset_ID == asset_id]
    x_asset = get_features(df_asset)
    y = models[asset_id].predict(x_asset)
    y_pred[df_test.Asset_ID == asset_id] = y

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T01:49:46.89502Z","iopub.execute_input":"2021-12-23T01:49:46.895375Z","iopub.status.idle":"2021-12-23T01:49:46.902432Z","shell.execute_reply.started":"2021-12-23T01:49:46.895339Z","shell.execute_reply":"2021-12-23T01:49:46.901273Z"}}
df_asset_details_asset = df_asset_details.set_index("Asset_ID")
weight = df_asset_details_asset["Weight"].to_dict()
asset_ids = df_test.Asset_ID
weight = asset_ids.replace(weight)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T01:53:46.284791Z","iopub.execute_input":"2021-12-23T01:53:46.285204Z","iopub.status.idle":"2021-12-23T01:53:46.471909Z","shell.execute_reply.started":"2021-12-23T01:53:46.285168Z","shell.execute_reply":"2021-12-23T01:53:46.470988Z"}}
is_nan = np.isnan(y_true)
y_true = y_true[~is_nan]
y_pred = y_pred[~is_nan]
weight = weight[~is_nan]
print(weighted_correlation(y_true, y_pred, weight))

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-11-06T00:45:51.361771Z","iopub.execute_input":"2021-11-06T00:45:51.362127Z","iopub.status.idle":"2021-11-06T00:45:51.981721Z","shell.execute_reply.started":"2021-11-06T00:45:51.362087Z","shell.execute_reply":"2021-11-06T00:45:51.981002Z"}}
env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        model = models[row['Asset_ID']]
        t = time.time()
        x_test = get_features(row, row=True)
        y_pred = model.predict([x_test])[0]
        print("t={}".format(time.time() - t))
        df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
        
        
        # Print just one sample row to get a feeling of what it looks like
        if i == 0 and j == 0:
            display(x_test)

    # Display the first prediction dataframe
    if i == 0:
        display(df_pred)

    # Send submissions
    env.predict(df_pred)

# %% [code]
