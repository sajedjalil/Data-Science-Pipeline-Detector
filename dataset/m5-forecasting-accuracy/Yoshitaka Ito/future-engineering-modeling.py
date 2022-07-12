# -----------------------
# thanks to:
#  https://www.kaggle.com/anshuls235/m5-forecasting-eda-fe-modelling
# -----------------------

import os
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import joblib

from sklearn.metrics import mean_squared_error

# read data
sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales.name = 'sales'
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
calendar.name = 'calendar'
prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
prices.name = 'prices'

#Add zero sales for the remaining days 1942-1969
# 予測対象の日数を追加
for d in range(1942,1970):
    col = 'd_' + str(d)
    sales[col] = 0
    sales[col] = sales[col].astype(np.int16)

# --- decrease volume of data ---
# メモリ節約のため、データ量削減
sales_bd = np.round(sales.memory_usage().sum()/(1024*1024),1)
calendar_bd = np.round(calendar.memory_usage().sum()/(1024*1024),1)
prices_bd = np.round(prices.memory_usage().sum()/(1024*1024),1)

#Downcast in order to save memory
def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df  

sales = downcast(sales)
prices = downcast(prices)
calendar = downcast(calendar)

# convert from wide to long format
# 横持ちから縦持ちに変換
df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()

# combine data
# calendar, pricesデータとjoin
df = pd.merge(df, calendar, on='d', how='left')
df = pd.merge(df, prices, on=['store_id','item_id','wm_yr_wk'], how='left') 


# Label Encoding
# change categories to numerics by serides.cat.codes() method
# カテゴリを数値化
d_id = dict(zip(df.id.cat.codes, df.id))
d_item_id = dict(zip(df.item_id.cat.codes, df.item_id))
d_dept_id = dict(zip(df.dept_id.cat.codes, df.dept_id))
d_cat_id = dict(zip(df.cat_id.cat.codes, df.cat_id))
d_store_id = dict(zip(df.store_id.cat.codes, df.store_id))
d_state_id = dict(zip(df.state_id.cat.codes, df.state_id))

# delete 'd_' in 'd' column
df.d = df['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)

# take list of columns
cols = df.dtypes.index.tolist()

# take list of data type
types = df.dtypes.values.tolist()

# enumerate() get index and contents of list
for i, type in enumerate(types):
    if type.name == 'category':
        # convert categorical feature to numerical feature
        df[cols[i]] = df[cols[i]].cat.codes


        
# --- feture engineering ---

# - mean encoding -
# 基本的なグループごとの販売数の平均値
df['item_sold_avg'] = df.groupby('item_id')['sold'].transform('mean').astype(np.float16)
# df['state_sold_avg'] = df.groupby('state_id')['sold'].transform('mean').astype(np.float16)
# df['store_sold_avg'] = df.groupby('store_id')['sold'].transform('mean').astype(np.float16)
# df['cat_sold_avg'] = df.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
# df['dept_sold_avg'] = df.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
# df['cat_dept_sold_avg'] = df.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
df['store_item_sold_avg'] = df.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
# df['cat_item_sold_avg'] = df.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
# df['dept_item_sold_avg'] = df.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
# df['state_store_sold_avg'] = df.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
# df['state_store_cat_sold_avg'] = df.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
# df['store_cat_dept_sold_avg'] = df.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)


# - trend by date -
# 月日別平均販売数、標準偏差
df['weekly_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','weekday'])['sold'].transform('mean').astype(np.float16)
df['weekly_sold_std'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','weekday'])['sold'].transform('std').astype(np.float16)
df['monthly_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','month'])['sold'].transform('mean').astype(np.float16)
df['monthly_sold_std'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','month'])['sold'].transform('std').astype(np.float16)


# - snap trend -
# snap日における販売数の傾向
# df['weekly_snapCA_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday', 'snap_CA'])['sold'].transform('mean').astype(np.float16)
# df['weekly_snapTX_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday', 'snap_TX'])['sold'].transform('mean').astype(np.float16)
# df['weekly_snapWI_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday', 'snap_WI'])['sold'].transform('mean').astype(np.float16)
df['weekly_snapCA_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday', 'snap_CA'])['sold'].transform('mean').astype(np.float16)
df['weekly_snapTX_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday', 'snap_TX'])['sold'].transform('mean').astype(np.float16)
df['weekly_snapWI_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday', 'snap_WI'])['sold'].transform('mean').astype(np.float16)
# df['weekly_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','weekday'])['sold'].transform('mean').astype(np.float16)
df['weekly_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday'])['sold'].transform('mean').astype(np.float16)
df['snap_CA_trend'] = (df['weekly_snapCA_avg_sold'] - df['weekly_avg_sold']).astype(np.float16)
df['snap_TX_trend'] = (df['weekly_snapTX_avg_sold'] - df['weekly_avg_sold']).astype(np.float16)
df['snap_WI_trend'] = (df['weekly_snapWI_avg_sold'] - df['weekly_avg_sold']).astype(np.float16)
df.drop(['weekly_avg_sold', 'weekly_snapCA_avg_sold', 'weekly_snapTX_avg_sold', 'weekly_snapWI_avg_sold'],axis=1,inplace=True)
gc.collect()


# - event trend -
# event日における販売数の傾向
# df['weekly_event_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday', 'event_type_1'])['sold'].transform('mean').astype(np.float16)
df['weekly_event_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday', 'event_type_1'])['sold'].transform('mean').astype(np.float16)
# df['weekly_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','weekday'])['sold'].transform('mean').astype(np.float16)
df['weekly_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday'])['sold'].transform('mean').astype(np.float16)
df['event_trend'] = (df['weekly_event_avg_sold'] - df['weekly_avg_sold']).astype(np.float16)
df.drop(['weekly_avg_sold'],axis=1,inplace=True)
gc.collect()


# - lag -
# 時間遅れ
def create_lags(dt):
    lags = [1,2,3,6,7,30]
    lag_cols = [f"lag_{lag}" for lag in lags ]   
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sold"]].groupby("id")["sold"].shift(lag)
        
create_lags(df)


# - moving average -
# 移動平均
df['rolling_sold_mean_7'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)

df['rolling_sold_mean_28'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=28).mean()).astype(np.float16)

df['rolling_sold_mean_180'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=180).mean()).astype(np.float16)

# 7日移動標準偏差
df['rolling_sold_std'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=7).std()).astype(np.float16)

# expanding window
# 移動平均を徐々に増やして、それぞれのレコードごとに過去全ての日数の販売数を算出
df['expanding_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.expanding(2).mean()).astype(np.float16)


# - moving average per SNAP -
# SNAP毎の移動平均の傾向
# df['rolling_sold_mean_snapCA_7'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'snap_CA'])['sold'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)
# df['rolling_sold_mean_snapTX_7'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'snap_TX'])['sold'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)
# df['rolling_sold_mean_snapWI_7'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'snap_WI'])['sold'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)

# df['rolling_sold_snap_CA_7'] = df['rolling_sold_mean_snapCA_7'] - df['rolling_sold_mean_7']
# df['rolling_sold_snap_TX_7'] = df['rolling_sold_mean_snapTX_7'] - df['rolling_sold_mean_7']
# df['rolling_sold_snap_WI_7'] = df['rolling_sold_mean_snapWI_7'] - df['rolling_sold_mean_7']

# df.drop(['mean7_7_avg_sold', 'mean7_7_snapCA_avg_sold', 'mean7_7_snapTX_avg_sold', 'mean7_7_snapWI_avg_sold'],axis=1,inplace=True)
# df.drop(['rolling_sold_mean_snapCA_7', 'rolling_sold_mean_snapTX_7', 'rolling_sold_mean_snapWI_7'], axis=1, inplace=True) 
# gc.collect()


# - selling trend -
# 販売数の差分傾向
df['daily_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sold'].transform('mean').astype(np.float16)
df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
df['selling_trend'] = (df['daily_avg_sold'] - df['avg_sold']).astype(np.float16)
df.drop(['daily_avg_sold','avg_sold'],axis=1,inplace=True)


# - SNAP_selling_trend -
# 各SNAP日の販売数の差分傾向
# SNAP_CA
df['daily_avg_sold_SNAP_CA'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','snap_CA','d'])['sold'].transform('mean').astype(np.float16)
df['avg_sold_SNAP_CA'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'snap_CA'])['sold'].transform('mean').astype(np.float16)
df['selling_trend_SNAP_CA'] = (df['daily_avg_sold_SNAP_CA'] - df['avg_sold_SNAP_CA']).astype(np.float16)
df.drop(['daily_avg_sold_SNAP_CA','avg_sold_SNAP_CA'],axis=1,inplace=True)

# SNAP_TX
df['daily_avg_sold_SNAP_TX'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','snap_TX','d'])['sold'].transform('mean').astype(np.float16)
df['avg_sold_SNAP_TX'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'snap_TX'])['sold'].transform('mean').astype(np.float16)
df['selling_trend_SNAP_TX'] = (df['daily_avg_sold_SNAP_TX'] - df['avg_sold_SNAP_TX']).astype(np.float16)
df.drop(['daily_avg_sold_SNAP_TX','avg_sold_SNAP_TX'],axis=1,inplace=True)

# SNAP_WI
df['daily_avg_sold_SNAP_WI'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','snap_WI','d'])['sold'].transform('mean').astype(np.float16)
df['avg_sold_SNAP_WI'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'snap_WI'])['sold'].transform('mean').astype(np.float16)
df['selling_trend_SNAP_WI'] = (df['daily_avg_sold_SNAP_WI'] - df['avg_sold_SNAP_WI']).astype(np.float16)
df.drop(['daily_avg_sold_SNAP_WI','avg_sold_SNAP_WI'],axis=1,inplace=True)

gc.collect()


# - 1lag's sales volume per SNAP -
# SNAP毎の 1lagの販売数
# df['lag_7_snapCA_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'lag_7', 'snap_CA'])['sold'].transform('mean').astype(np.float16)
# df['lag_7_snapTX_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'lag_7', 'snap_TX'])['sold'].transform('mean').astype(np.float16)
# df['lag_7_snapWI_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'lag_7', 'snap_WI'])['sold'].transform('mean').astype(np.float16)
df['lag_1_snapCA_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'lag_1', 'snap_CA'])['sold'].transform('mean').astype(np.float16)
df['lag_1_snapTX_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'lag_1', 'snap_TX'])['sold'].transform('mean').astype(np.float16)
df['lag_1_snapWI_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'lag_1', 'snap_WI'])['sold'].transform('mean').astype(np.float16)
# df['lag_7_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','lag_7'])['sold'].transform('mean').astype(np.float16)
# df['lag_7_avg_sold'] = df.groupby(['item_id', 'store_id', 'weekday', 'lag_7'])['sold'].transform('mean').astype(np.float16)
df['lag_1_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','lag_1'])['sold'].transform('mean').astype(np.float16)
# df['lag_7_snap_CA'] = df['lag_7_snapCA_avg_sold'] - df['lag_7_avg_sold']
# df['lag_7_snap_TX'] = df['lag_7_snapTX_avg_sold'] - df['lag_7_avg_sold']
# df['lag_7_snap_WI'] = df['lag_7_snapWI_avg_sold'] - df['lag_7_avg_sold']
df['lag_1_snap_CA'] = df['lag_1_snapCA_avg_sold'] - df['lag_1_avg_sold']
df['lag_1_snap_TX'] = df['lag_1_snapTX_avg_sold'] - df['lag_1_avg_sold']
df['lag_1_snap_WI'] = df['lag_1_snapWI_avg_sold'] - df['lag_1_avg_sold']

# df.drop(['lag_7_avg_sold', 'lag_7_snap_CA', 'lag_7_snap_TX','lag_7_snap_WI'], axis=1, inplace=True)
df.drop(['lag_1_avg_sold', 'lag_1_snapCA_avg_sold', 'lag_1_snapTX_avg_sold','lag_1_snapWI_avg_sold'], axis=1, inplace=True)
gc.collect()


# - prices -
# 価格関連変数

# price max
# df['price_max'] = df.groupby(['store_id', 'item_id','wm_yr_wk'])['sell_price'].transform('max')

# price min
# df['price_min'] = df.groupby(['store_id', 'item_id','wm_yr_wk'])['sell_price'].transform('min')

# moving average price std
# df['rolling_price_std'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform(lambda x: x.rolling(window=7).std()).astype(np.float16)

# moving average price
# df['price_mean'] = df.groupby(['store_id', 'item_id','wm_yr_wk'])['sell_price'].transform('mean')
# df['rolling_price_mean_7'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)

# price trend 
df['daily_avg_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sell_price'].transform('mean').astype(np.float16)
df['avg_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform('mean').astype(np.float16)
df['price_trend'] = (df['daily_avg_price'] - df['avg_price']).astype(np.float16)
df.drop(['daily_avg_price','avg_price'],axis=1,inplace=True)

# monthly prices
df['price_momentun_m'] = df['sell_price'] / df.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform(lambda x: np.mean([i for i in x if not np.isnan(i)]))


# - revenue -
# 売上関連変数
# revenue trend
df['daily_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sold'].transform('mean').astype(np.float16)
df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
df['daily_avg_sell_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sell_price'].transform('mean').astype(np.float16)
df['avg_sell_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform('mean').astype(np.float16)
df['revenue_trend'] = (df['daily_avg_sold'] * df['daily_avg_sell_price'] - df['avg_sold'] * df['avg_sell_price']).astype(np.float16)
df.drop(['daily_avg_sold','avg_sold', 'daily_avg_sell_price', 'avg_sell_price'],axis=1,inplace=True)

# moving average revenue
# 7日間移動平均売上
df['rolling_sall_price_mean_7'] = df.groupby(['id', 'item_id', 'dept_id','cat_id', 'store_id', 'state_id'])['sell_price'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)
df['rolling_revenue_mean_7'] = df['rolling_sold_mean_7'] * df['rolling_sall_price_mean_7']
df.drop(['rolling_sall_price_mean_7'], axis=1, inplace=True)

# weekly revenue trend
# 週間売上傾向
df['weekly_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','weekday'])['sold'].transform('mean').astype(np.float16)
df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
df['weekly_avg_sell_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','weekday'])['sell_price'].transform('mean').astype(np.float16)
df['avg_sell_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform('mean').astype(np.float16)
df['weekly_revenue_trend'] = (df['weekly_avg_sold'] * df['weekly_avg_sell_price'] - df['avg_sold'] * df['avg_sell_price']).astype(np.float16)
df.drop(['weekly_avg_sold','avg_sold', 'weekly_avg_sell_price', 'avg_sell_price'],axis=1,inplace=True)

# monthly revenue trend
# 月間売上傾向
df['monthly_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','month'])['sold'].transform('mean').astype(np.float16)
df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
df['monthly_avg_sell_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','month'])['sell_price'].transform('mean').astype(np.float16)
df['avg_sell_price'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform('mean').astype(np.float16)
df['monthly_revenue_trend'] = (df['monthly_avg_sold'] * df['monthly_avg_sell_price'] - df['avg_sold'] * df['avg_sell_price']).astype(np.float16)
df.drop(['monthly_avg_sold','avg_sold', 'monthly_avg_sell_price', 'avg_sell_price'],axis=1,inplace=True)

gc.collect()


# - Those are just ideas -
# その他、思いつき変数
df['lag1_lag2'] = (df['lag_1'] * df['lag_2']).astype(np.float16)
df['lag_1_lag6'] = (df['lag_1'] * df['lag_6']).astype(np.float16)
df['lag_1_lag_2_lag_6'] = df['lag_1'] * df['lag_2'] * df['lag_6'].astype(np.float16)
df['log_lag1_lag2'] = np.log1p(df['lag1_lag2']).astype(np.float16)
df['log_lag_1_lag6'] = np.log1p(df['lag_1_lag6']).astype(np.float16)
df['log_lag_1_lag_2_lag_6'] = np.log1p(df['lag_1_lag_2_lag_6']).astype(np.float16)
df['revenue_selling_trend'] = df['revenue_trend'] * df['selling_trend'].astype(np.float16)
df['log_revenue_selling_trend'] = np.log1p(df['revenue_selling_trend']).astype(np.float16)



# --- Modeling ---

# drop 'date'
# 'date'列削除
data.drop('date',axis=1,inplace=True)

# remove data having lots of null values
# 前半レコードnull値多いので削除
data = data[data['d']>=36]

# split data
# train用とtest用に分割
valid = data[(data['d']>=1914) & (data['d']<1942)][['id','d','sold']]
test = data[data['d']>=1942][['id','d','sold']]
eval_preds = test['sold']
valid_preds = valid['sold']


#Get the store ids
stores = sales.store_id.cat.codes.unique().tolist()

# train & predict by models
# 店舗ごとにモデル作成し予測
for store in stores:
    df = data[data['store_id']==store]
    
    #Split the data
    X_train, y_train = df[df['d']<1914].drop('sold',axis=1), df[df['d']<1914]['sold']
    X_valid, y_valid = df[(df['d']>=1914) & (df['d']<1942)].drop('sold',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['sold']
    X_test = df[df['d']>=1942].drop('sold',axis=1)
    
    #Train and validate
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=8,
        num_leaves=50,
        min_child_weight=300
    )
    print('*****Prediction for Store: {}*****'.format(d_store_id[store]))
    model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_valid,y_valid)],
             eval_metric='rmse', verbose=20, early_stopping_rounds=20)
    valid_preds[X_valid.index] = model.predict(X_valid)
    eval_preds[X_test.index] = model.predict(X_test)
    filename = '/kaggle/working/model'+str(d_store_id[store])+'.pkl'
    # save model
    joblib.dump(model, filename)
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()

    
# get future importance
# 特徴量重要度出力
cols = feature_importance_df_[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False).index
best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
best_features.groupby('feature').mean().sort_values(by='importance', ascending=False).to_csv('/kaggle/working/features.csv')


# get validation data
# validation用予測値抽出
valid['sold'] = valid_preds
validation = valid[['id', 'd', 'sold']]
validation = pd.pivot(validation, index='id', columns='d', values='sold').reset_index()
validation.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
validation.id = validation.id.map(d_id).str.replace('evaluation', 'validation')

# float to integer conversion
# 予測値を整数化
cols = validation.columns.to_list()
cols.remove('id')
validation[cols] = validation[cols].applymap(round)


# RMSE of validation data
# validation予測値のRMSEを確認
def rmse_f(y, pred_y):
    rmse = np.sqrt(mean_squared_error(y, pred_y))
    print("test's rmse score: {:.3f}".format(rmse))
val_data = sales.loc[:, 'd_1914':'d_1941']
eval_array = val_data.values.reshape(-1,)
predicted_array = validation.drop('id', axis=1).values.reshape(-1,)
# calculate rmse
rmse_f(eval_array, predicted_array)


# get evaluation
# evaluation用予測値
test['sold'] = eval_preds
evaluation = test[['id', 'd', 'sold']]
evaluation = pd.pivot(evaluation, index='id', columns='d', values='sold').reset_index()
evaluation.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
evaluation.id = evaluation.id.map(d_id)

# float to integer conversion
# 整数化
cols = evaluation.columns.to_list()
cols.remove('id')
evaluation[cols] = evaluation[cols].applymap(round)


# submit
# 提出用ファイル出力
submit = pd.concat([validation, evaluation]).reset_index(drop=True)
submit.to_csv('/kaggle/working/submission.csv', index=False)

