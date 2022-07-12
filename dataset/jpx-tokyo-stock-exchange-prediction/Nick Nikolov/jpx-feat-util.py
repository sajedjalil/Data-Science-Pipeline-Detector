# This is an utility script which is used by
# - a notebook [jpx_feat_generation](https://www.kaggle.com/code/notabene/jpx-feat-generation) which deals with the CV. It pre-computes the CV fold files which can then be used by simply importing the output of this notebook
# - a notebook [jpx-lgb-train](https://www.kaggle.com/code/notabene/jpx-lgb-train) which imports the fold files and trains a separate LGB model on each fold
# - a notebook which imports the models and the utility script to produce a submission file


import numpy as np
import pandas as pd
import random
import os
from datetime import timedelta
from scipy import stats
import lightgbm as lgb
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from matplotlib import pyplot as plt

filter_features =['High','Low','Open','Close','Volume']

max_lag = 70
params_lgb = {'learning_rate': 0.005,'metric':'None','objective': 'regression','boosting': 'gbdt','verbosity': 0,'n_jobs': -1,'force_col_wise':True}

def train_filter500 (train):
    print('train_filter_lgm_all original length = ' + str(len(train)))
    train = fill_nan_inf(train[train.Open.notnull() & train.Target.notnull()])
    print('train_filter_lgm_all pruned length = ' + str(len(train)))
    target_grouping = train.groupby('SecuritiesCode')['Target']
    target_spread = (target_grouping.max()- target_grouping.min()).sort_values()
    spread_l,spread_h=list(target_spread[:500].index), list(target_spread[500:].index)
    y_train, y_val = train[train['SecuritiesCode'].isin(spread_h)]['Target'].copy(), train[train['SecuritiesCode'].isin(spread_l)]['Target'].copy()
    train,val = train[train['SecuritiesCode'].isin(spread_h)], train[train['SecuritiesCode'].isin(spread_l)]
    train,val = train[filter_features],val[filter_features]
    train_ds, val_ds = lgb.Dataset(train, label = y_train), lgb.Dataset(val, label = y_val)

    model = lgb.train(params = params_lgb, 
                train_set = train_ds, 
                valid_sets = [train_ds, val_ds], 
                num_boost_round = 3000, 
                feval=feval_pearsonr,
                callbacks=[ lgb.early_stopping(stopping_rounds=300, verbose=True), lgb.log_evaluation(period=100)])  
    return model

def train_filter_lgm_all (train):
    print('train_filter_lgm_all original length = ' + str(len(train)))
    train = fill_nan_inf(train[train.Open.notnull() & train.Target.notnull()])
    print('train_filter_lgm_all pruned length = ' + str(len(train)))
    target_grouping = train.groupby('SecuritiesCode')['Target']
    target_spread = (target_grouping.max()- target_grouping.min()).sort_values()
    spread_h,spread_l=list(target_spread[:1000].index), list(target_spread[1000:].index)
    y_train, y_val = train[train['SecuritiesCode'].isin(spread_h)]['Target'].copy(), train[train['SecuritiesCode'].isin(spread_l)]['Target'].copy()
    val = train[train['SecuritiesCode'].isin(spread_l)].copy()
    train = train[train['SecuritiesCode'].isin(spread_h)]
    train,val = train[filter_features],val[filter_features]
    train_ds, val_ds = lgb.Dataset(train, label = y_train), lgb.Dataset(val, label = y_val)

    model = lgb.train(params = params_lgb, 
                train_set = train_ds, 
                valid_sets = [train_ds, val_ds], 
                num_boost_round = 3000, 
                feval=feval_pearsonr,
                callbacks=[ lgb.early_stopping(stopping_rounds=300, verbose=True), lgb.log_evaluation(period=100)])  
    return model

def train_filter_lgm (train, test, fold):
    result = test[["SecuritiesCode", 'Date']].copy()
    y_test = test['Target'].copy()
    print('y_test len = ' + str(len(y_test)))
    test = test[filter_features]
    print('train_filter_lgm: y_test has '+ str(len(y_test) - len(y_test.dropna())) + ' rows with NaN values')
    print('train_filter_lgm: train.shape is ' + str(train.shape[0])+', ' + str(train.shape[1]))
    print('train_filter_lgm: test.shape is ' + str(test.shape[0])+', ' + str(test.shape[1]))
    print('train_filter_lgm: test has '+ str(len(test) - len(test.dropna())) + ' rows with NaN values')
    print('train_filter_lgm: train has '+ str(len(train) - len(train.dropna())) + ' rows with NaN values')
    
    model = train_filter_lgm_all (train)
    pickle.dump(model, open(f'lgb_model{fold}', 'wb'))
    
    result.loc[:, "predict"] = model.predict(test)
    result.loc[:, "Target"] = y_test.values

    feature_imp = lgb_feat_importances(model, train.columns)[:50] #50 most important features
    feature_imp.to_pickle(f"feat_imp{fold}.pkl")

    plt.figure(figsize=(20, 10))
    sns.barplot(x="importance", y="feature", data=feature_imp)
    plt.title(f'LightGBM Features Fold {fold}')
    plt.tight_layout()
    plt.show()
    #plt.savefig('lgbm_importances-01.png')
    
    df, res_dates = [], result.Date.unique()
    for date in res_dates:
        curr = result[result.Date == date].sort_values("predict", ascending=False)
        curr.loc[:, "Rank"] = np.arange(len(curr["predict"]))
        df.append(curr)
    ranked = pd.concat(df, ignore_index=True)
    sharpe=calc_spread_return_sharpe(ranked)
    #sharpe_ratio.append(sharpe)
    print("Test Sharpe: {}".format(sharpe))
    print("Test MSE: {}".format(mean_squared_error(y_test,result["predict"])))
    return sharpe

def inference_filter (prices, model, sample_prediction):
    prices,lp = fill_nan_inf(prices[filter_features]), len(prices)
    sample_prediction["Prediction"] = model.predict(prices)
    sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
    sample_prediction['Rank'] = np.arange(0,lp)
    sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    return sample_prediction[["Date","SecuritiesCode","Rank"]]

def inference (prices,supp,sample_prediction, models, drop_cols):
    lp = len(prices)
    prices.Date,supp.Date = pd.to_datetime(prices.Date),pd.to_datetime(supp.Date)
    tdate = max(prices.Date) # should be just one Date
    print(prices.Date.unique())
    print('inference: len(prices) = '+str(lp))
    supp.drop(supp[(supp.Date < tdate - timedelta(days=max_lag)) | (supp.Date >= tdate)].index, inplace=True)
    print('inference: len(supp) = '+str(len(supp)))
    prices = pd.concat([supp, fill_nan_inf(prices)])
    print('inference: prices.shape is:')
    print(prices.shape)
    feat = create_features(prices, None, tdate)
    print('inference: feat num of unique stocks = ' + str(len(feat.SecuritiesCode.unique())))
    print('inference: feat unique dates are:')
    print(feat.Date.unique())
    print('inference: feat.shape is:')
    print(feat.shape)
    X_test = feat[feat.Date == tdate]
    print('inference: X_test.shape is:')
    print(X_test.shape)
    if drop_cols:
        X_test = X_test.drop(drop_cols, axis=1)
    else:
        X_test = X_test[filter_features]
    lgbm_preds = [] 
    for model in models:
        lgbm_preds.append( model.predict(X_test) )
    lgbm_preds = np.mean(lgbm_preds, axis=0)
    sample_prediction["Prediction"] = lgbm_preds
    sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
    sample_prediction['Rank'] = np.arange(0,lp)
    sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    return sample_prediction[["Date","SecuritiesCode","Rank"]]

def fill_nan_inf(df):
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    #tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def lgb_feat_importances(model, train_columns):
    #based on: https://stackoverflow.com/a/60095798/206253
    
    #assumption: feature importance values are ordered just like the model matrix columns were ordered
    #during training (incl. one-hot dummy cols), see LightGBM #209.
    
    if "basic.Booster" in str(model.__class__):
        # lightgbm.basic.Booster was trained directly, so using feature_importance() function 
        cv_varimp_df = pd.DataFrame([train_columns, model.feature_importance()]).T
    else:
        # Scikit-learn API LGBMClassifier or LGBMRegressor was fitted, 
        # so using feature_importances_ property
        cv_varimp_df = pd.DataFrame([train_columns, model.feature_importances_]).T

    cv_varimp_df.columns = ['feature', 'importance']
    cv_varimp_df.sort_values(by='importance', ascending=False, inplace=True)
    return cv_varimp_df

def feval_rmse(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'rmse', mean_squared_error(y_true, y_pred), False

def feval_pearsonr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'pearsonr', stats.pearsonr(y_true, y_pred)[0], True

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

drop_cols = ['RowId', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Date', 'ExpectedDividend', 'SupervisionFlag', 'section', '33sector', '17sector', 'SizeCode', 'MarketCapitalization', 'cash_avg_range', 'cash_avg_diff', 'std_day_range', 'std_day_diff']
drop_cols2 = ['RowId', 'SecuritiesCode', 'Target', 'Date', 'ExpectedDividend', 'SupervisionFlag', 'section', '33sector', '17sector', 'SizeCode']
def set_rank(df):
    """
    Args:
        df (pd.DataFrame): including predict column
    Returns:
        df (pd.DataFrame): df with Rank
    """
    # sort records to set Rank
    df = df.sort_values("predict", ascending=False)
    # set Rank starting from 0
    df.loc[:, "Rank"] = np.arange(len(df["predict"]))
    return df

def get_stock_features(price, code, tdate = None):
    #to do: add more features which normalise the longitudinal features on market metadata
    ##print('code = ' + str(code))
    feats = price.loc[price["SecuritiesCode"] == code].copy()
    feats['Date'] = pd.to_datetime(feats['Date'])
    feats = feats.sort_values(by=['Date'])
    ##print('feats.shape [0] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['out_of_hours_diff'] = feats['Open'] / feats.shift(periods=-1, fill_value=1)['Close']
    feats['5d_return'] = feats["Close"].pct_change(5) * 0.01
    feats['10d_return'] = feats["Close"].pct_change(10) * 0.01
    feats['22d_return'] = feats["Close"].pct_change(22) * 0.01
    feats['66d_return'] = feats["Close"].pct_change(66) * 0.01
    ##print('feats.shape [1] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats["5d_volatility"] = np.log(feats["Close"]).diff().rolling(5).std()
    feats["10d_volatility"] = np.log(feats["Close"]).diff().rolling(10).std()
    feats["22d_volatility"] = np.log(feats["Close"]).diff().rolling(22).std()
    feats["66d_volatility"] = np.log(feats["Close"]).diff().rolling(66).std()
    #
    ##print('feats.shape [2] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['5d_normalized_day_diff'] = feats['day_diff'] / feats.rolling(window=5)['day_diff'].median()
    feats['10d_normalized_day_diff'] = feats['day_diff'] / feats.rolling(window=10)['day_diff'].median()
    feats['22d_normalized_day_diff'] = feats['day_diff'] / feats.rolling(window=22)['day_diff'].median()
    feats['66d_normalized_day_diff'] = feats['day_diff'] / feats.rolling(window=66)['day_diff'].median()
    #
    ##print('feats.shape [3] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['5d_normalized_day_range'] = feats['day_range'] / feats.rolling(window=5)['day_range'].median()
    feats['10d_normalized_day_range'] = feats['day_range'] / feats.rolling(window=10)['day_range'].median()
    feats['22d_normalized_day_range'] = feats['day_range'] / feats.rolling(window=22)['day_range'].median()
    feats['66d_normalized_day_range'] = feats['day_range'] / feats.rolling(window=66)['day_range'].median()
    #
    ##print('feats.shape [4] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['5d_norm_out_of_hours_diff'] = feats['out_of_hours_diff'] / feats.rolling(window=5)['out_of_hours_diff'].median()
    feats['10d_norm_out_of_hours_diff'] = feats['out_of_hours_diff'] / feats.rolling(window=10)['out_of_hours_diff'].median()
    feats['22d_norm_out_of_hours_diff'] = feats['out_of_hours_diff'] / feats.rolling(window=22)['out_of_hours_diff'].median()
    feats['66d_norm_out_of_hours_diff'] = feats['out_of_hours_diff'] / feats.rolling(window=66)['out_of_hours_diff'].median()
    #
    ##print('feats.shape [5] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['5d_norm_volume'] = feats['Volume'] / feats.rolling(window=5)['Volume'].median()
    feats['10d_norm_volume'] = feats['Volume'] / feats.rolling(window=10)['Volume'].median()
    feats['22d_norm_volume'] = feats['Volume'] / feats.rolling(window=22)['Volume'].median()
    feats['66d_norm_volume'] = feats['Volume'] / feats.rolling(window=66)['Volume'].median()
    #
    feats['1d_delta_close'] = feats['Close'] / feats['Close'].shift(periods=-1, fill_value=1) # need to remove the first record
    feats['1d_delta_day_range'] = feats['day_range'] / feats['day_range'].shift(periods=-1, fill_value=1) # need to remove the first record
    feats['1d_delta_day_diff'] = feats['day_diff'] / feats['day_diff'].shift(periods=-1, fill_value=1) # need to remove the first record
    feats['1d_delta_out_of_hours_diff'] = feats['out_of_hours_diff'] / feats['out_of_hours_diff'].shift(periods=-1, fill_value=1) # need to remove the first record
    feats['1d_delta_vol'] = feats['Volume'] / feats['Volume'].shift(periods=-1, fill_value=1)
    #
    ##print('feats.shape [9] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['prev_1d_delta_close'] = feats['1d_delta_close'].shift(periods=-1, fill_value=1)
    feats['prev_1d_delta_day_range'] = feats['1d_delta_day_range'].shift(periods=-1, fill_value=1)
    feats['prev_1d_delta_day_diff'] = feats['1d_delta_day_diff'].shift(periods=-1, fill_value=1)
    feats['prev_1d_delta_out_of_hours'] = feats['1d_delta_out_of_hours_diff'].shift(periods=-1, fill_value=1)
    feats['prev_1d_delta_vol'] = feats['1d_delta_vol'].shift(periods=-1, fill_value=1)
    feats['prev_med_day_range'] = feats['med_day_range'].shift(periods=-1, fill_value=1)
    feats['prev_med_day_diff'] = feats['med_day_diff'].shift(periods=-1, fill_value=1)
    #
    ##print('feats.shape [10] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    feats['5d_norm_close'] = feats['Close'] / feats.rolling(window=5)['Close'].median()
    feats['10d_norm_close'] = feats['Close'] / feats.rolling(window=10)['Close'].median()
    feats['22d_norm_close'] = feats['Close'] / feats.rolling(window=22)['Close'].median()
    feats['66d_norm_close'] = feats['Close'] / feats.rolling(window=66)['Close'].median()
    ##print('feats.shape [11] = '+str(feats.shape[0]) + '/' + str(feats.shape[1]))
    ##print(str(len(feats) - len(feats.dropna())) + ' rows with NaN values')
    #feats = feats.dropna()
    if tdate:
        return feats[feats.Date == tdate]
    return feats[max_lag:]

def create_features (nprices, codes = None, tdate = None): #this function is supposed to replace all pre-processing & feature generation. It calls get_stock_features()
    
    meta = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')
    meta = meta.rename(columns={"Section/Products": "section", "17SectorCode": "17sector", "NewIndexSeriesSizeCode": "SizeCode", "33SectorCode": "33sector"})
    meta = meta.drop(['EffectiveDate', 'Name', 'NewMarketSegment', '33SectorName', '17SectorName', 'TradeDate', 'IssuedShares', 'Universe0', 'NewIndexSeriesSize', 'Close'], axis=1)
    stocks = frozenset(nprices['SecuritiesCode'])
    meta = meta[meta['SecuritiesCode'].isin(stocks)]
    meta.loc[meta['section'] == 'First Section (Domestic)','section'] = 0
    meta.loc[meta['section'] == 'JASDAQ(Standard / Domestic)','section'] = 1
    meta.loc[meta['section'] == 'Mothers (Domestic)','section'] = 2
    meta.loc[meta['section'] == 'Second Section(Domestic)','section'] = 3
    meta.loc[meta['section'] == 'JASDAQ(Growth/Domestic)','section'] = 4
    meta.loc[meta['SizeCode'] == '-','SizeCode'] = 0
    meta.loc[meta['33sector'] == '-','33sector'] = 0
    meta.loc[meta['17sector'] == '-','17sector'] = 0
    
    nprices = nprices.merge(meta)
    
    #adjusting the prices & volumes
    print('create_features: '+ str(len(nprices) - len(nprices.dropna())) + ' rows with NaN values at point 0')
    f = lambda df: df['AdjustmentFactor'][::-1].cumprod()[::-1]
    nprices['CumAdjustmentFactor'] = nprices.groupby('SecuritiesCode').apply(f).values
    for price in ['Open','High','Low','Close']:
        nprices[price] *= nprices['CumAdjustmentFactor']
    nprices['Volume'] /= nprices['CumAdjustmentFactor']
    nprices = nprices.drop(['CumAdjustmentFactor', 'AdjustmentFactor'], axis=1)
    #derivative features
    nprices['day_range'] = nprices['High'] / nprices['Low']
    nprices['day_diff'] = nprices['Close'] / nprices['Open']
    print('create_features: nprices.shape [0] = '+str(nprices.shape[0]) + '/' + str(nprices.shape[1]))
    #add cash_vol and develop normalised and lagged derivative features
    ##nprices['cash_avg_range'] = ((nprices['High'] + nprices['Low']) / 2) *nprices['Volume']
    ##nprices['cash_avg_diff'] = ((nprices['Close'] + nprices['Open']) / 2) *nprices['Volume']
    #print(str(len(nprices) - len(nprices.dropna())) + ' rows with NaN values at point 3')
    print('create_features: nprices.shape [1] = '+str(nprices.shape[0]) + '/' + str(nprices.shape[1]))
    #day aggregations
    day_groups = nprices.groupby("Date")
    #whole market aggregations (need to do lagged versions)
    nprices['med_day_range'] = day_groups['day_range'].transform('median')
    nprices['med_day_diff'] = day_groups['day_diff'].transform('median')
    nprices['avg_day_range'] = day_groups['day_range'].transform('mean')
    nprices['avg_day_diff'] = day_groups['day_diff'].transform('mean')
    ##nprices['std_day_range'] = day_groups['day_range'].transform('std')
    ##nprices['std_day_diff'] = day_groups['day_diff'].transform('std')
    print('create_features: nprices.shape [2] = '+str(nprices.shape[0]) + '/' + str(nprices.shape[1]))
    #sector/segment aggregations (add lagged versions, too)
    #section
    day_section_groups = nprices.groupby(["Date", 'section'])
    nprices['med_day_section_range'] = day_section_groups['day_range'].transform('median')
    nprices['med_day_section_diff'] = day_section_groups['day_diff'].transform('median')
    nprices['avg_day_section_range'] = day_section_groups['day_range'].transform('mean')
    nprices['avg_day_section_diff'] = day_section_groups['day_diff'].transform('mean')
    #SizeCode
    day_SizeCode_groups = nprices.groupby(["Date", 'SizeCode'])
    nprices['med_day_SizeCode_range'] = day_SizeCode_groups['day_range'].transform('median')
    nprices['med_day_SizeCode_diff'] = day_SizeCode_groups['day_diff'].transform('median')
    nprices['avg_day_SizeCode_range'] = day_SizeCode_groups['day_range'].transform('mean')
    nprices['avg_day_SizeCode_diff'] = day_SizeCode_groups['day_diff'].transform('mean')
    #print(str(len(nprices) - len(nprices.dropna())) + ' rows with NaN values at point 7')
    print('create_features: nprices.shape [5] = '+str(nprices.shape[0]) + '/' + str(nprices.shape[1]))
    #non-zero defaults
    nonzero_diff = np.where(nprices['day_diff'] != 0, nprices['day_diff'], 1)
    print(nonzero_diff)
    nonzero_range = np.where(nprices['day_range'] != 0, nprices['day_range'], 1)
    #add stock normalisation by total market:
    print(np.where(nprices['med_day_diff'] != 0, nprices['med_day_diff'], nonzero_diff))
    ##nprices['n_day_diff_med'] = nprices['day_diff'] / np.where(nprices['med_day_diff'] != 0, nprices['med_day_diff'], nonzero_diff)
    nprices['n_day_range_med'] = nprices['day_range'] / np.where(nprices['med_day_range'] != 0, nprices['med_day_range'], nonzero_range)
    ##nprices['n_day_diff_avg'] = nprices['day_diff'] / np.where(nprices['avg_day_diff'] != 0, nprices['avg_day_diff'], nonzero_diff)
    ##nprices['n_day_range_avg'] = nprices['day_range'] / np.where(nprices['avg_day_range'] != 0, nprices['avg_day_range'], nonzero_range)
    #print(str(len(nprices) - len(nprices.dropna())) + ' rows with NaN values at point 8')
    #add stock normalisation by section:
    #longitudinal features
    ##nprices = get_stock_features(nprices, 1301)
    if not codes: codes = nprices["SecuritiesCode"].unique()    
    buff = []
    for code in codes:
        buff.append(get_stock_features(nprices, code, tdate))
        if tdate and len(buff[-1]) > 1:
            print(' create_features: code = ' + str(code) + ' has featLen = ' + str(len(buff[-1])))
    nprices = pd.concat(buff)
    print('create_features: nprices.shape [7] = '+str(nprices.shape[0]) + '/' + str(nprices.shape[1]))
    print('create_features: feature generation finished')
    del buff
    return nprices