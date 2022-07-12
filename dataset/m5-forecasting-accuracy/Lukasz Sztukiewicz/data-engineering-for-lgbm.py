#cat = pd.CategoricalDtype(categories=['a', 'b', 'c'], ordered=True)
#cates = pd.DataFrame({'A':['a','b','c','a','b','b'],'B':['a','a','a','a','b','a']})
#cates['A'] = cates['A'].astype(cat)
#cates['A'].cat.codes.astype(np.int8)

#dtype = {numcol:np.float32 for numcol in numcols}
#df.sort_values(by=['d', 'store_id', 'item_id']) orignal 
#df.sort_values(by=['d'])

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta, datetime
import lightgbm as lgb

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

pd.options.display.max_columns = 5

#some globals
IS_TRAIN = False
STARTDAY = 1914
SPAN_TO_PREDICT = 28 #how many LSTM sequences
WINDOWS = [3,7,14,21,28,56,90,140,365]

#some globals
H = SPAN_TO_PREDICT
MAX_LAG = max(WINDOWS)
TRAIN_LAST_DAY = STARTDAY-1
FIRST_DAY = datetime(2016,4, 25)

#prices
prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv",dtype = PRICE_DTYPES)
for col, dtype in PRICE_DTYPES.items():
    if dtype == "category":
        prices[col] = prices[col].cat.codes.astype(np.int16)
        prices[col] -= prices[col].min() #removing -1

#calendar
cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
cal['date'] = pd.to_datetime(cal['date'])
for col, dtype in CAL_DTYPES.items():
    if dtype == "category":
        cal[col] = cal[col].cat.codes.astype(np.int16)
        cal[col] -= cal[col].min() #removing -1

#train

#nrows = rows to get
#skiprows = number of rows skipped

startday = 1 if IS_TRAIN else TRAIN_LAST_DAY - MAX_LAG
numcols = ['d_{0}'.format(x) for x in range(startday,TRAIN_LAST_DAY+1)]
catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
dtype = {numcol:np.float32 for numcol in numcols}

dtype.update({col: "category" for col in catcols if col != "id"})

usecols = catcols + numcols
train = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", usecols=usecols, dtype=dtype)

for col in catcols:
    if col != "id":
        train[col] = train[col].cat.codes.astype(np.int16)
        train[col] -= train[col].min()
    
if not IS_TRAIN:
    for x in range(STARTDAY, STARTDAY+SPAN_TO_PREDICT):
        train['d_{0}'.format(x)] = np.nan

train = pd.melt(train,
                  id_vars = catcols,
                  value_vars = [col for col in train.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
train = train.merge(cal, on= "d", copy = False)
train = train.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
store_id = 2
x_train = train.iloc[3049*(store_id-1):3049*store_id]

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)
    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
