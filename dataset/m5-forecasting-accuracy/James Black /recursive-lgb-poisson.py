import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import datetime
from datetime import timedelta, datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
ptypes = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
ctypes = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
def make_matrix(first_date = 1000, last_date = 1913,train = True):
    
    prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv', dtype = ptypes )
    calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv', dtype = ctypes)

    for col,dtype in ptypes.items():
        if dtype == 'category':
            prices[col] = prices[col].cat.codes.astype("int16") 
            prices[col] = prices[col]- min(prices[col])
    for col,dtype in ctypes.items():
        if dtype == 'category':
            calendar[col] = calendar[col].cat.codes.astype("int16")
            calendar[col] = calendar[col]- min(calendar[col])
   
    
    calendar.d = [int(i[2:]) for i in calendar.d]
    calendar["date"] = pd.to_datetime(calendar["date"])
    
    start_day = max(1, first_date)
    numcols = [f"d_{day}" for day in range(start_day,last_date+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv',usecols = catcols+numcols, dtype = dtype)

    if not train:
        for i in range(1914,1913 + 29):
            sales[f'd_{i}'] = np.nan
        
    
    matrix = pd.melt(sales, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                     value_vars =  [i for i in sales.columns if i.startswith('d_')])
    
    matrix.rename(columns = {'value':'sales', 'variable':'d'}, inplace = True)
    for col in catcols:
        if col != "id":
            matrix[col] = matrix[col].cat.codes.astype("int16")
            matrix[col] -= matrix[col].min()
    
    matrix.d = [int(i[2:]) for i in matrix.d]
    
    matrix = matrix[matrix.d > first_date]

    
    
    matrix = matrix.merge(calendar, how = 'left', on = 'd', copy = False)
    
    matrix = matrix.merge(prices, how = 'left', on = ['store_id', 'item_id', 'wm_yr_wk'], copy = False )
  

            
    
    return matrix

    
ts = time.time()
matrix = make_matrix(365)
time.time() - ts 

matrix.info()

date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
for date_feat_name, date_feat_func in date_features.items():
    if date_feat_name in matrix.columns:
        matrix[date_feat_name] = matrix[date_feat_name].astype("int16")
    else:
        matrix[date_feat_name] = getattr(matrix["date"].dt, date_feat_func).astype("int16")


def make_lags(df, cols, lags):
    #tmp = df[['d','item_id','store_id','sales']]
    for col in cols:
        for n in lags:
            copy = df.copy()
            copy.rename(columns = {col:col+'_lag_'+str(n)}, inplace = True)
            copy.d = copy.d + n
            copy = copy[['d','item_id','store_id',col+'_lag_'+str(n)]]
            
            df = pd.merge(df, copy, on = ['d','store_id','item_id'], how = 'left')
            df.dropna(inplace = True)
    return df

ts = time.time()
matrix = make_lags(matrix,['sales'], [7,28])
time.time() - ts 

def make_mavg_features(df, features, by, windows): 
    by1 = ''
    for i in by:
        by1 += f'{i}_'
    for feature in features: 
        for window in windows:
                df[f'mavg_{feature}_{by1}_{window}'] = pd.concat([df[by],df[feature]], axis = 1).groupby(by).transform(lambda x : x.rolling(window).mean())
                df[f'mavg_{feature}_{by1}_{window}'] = df[f'mavg_{feature}_{by1}_{window}'].astype(np.float32)
    return df 

ts = time.time()
matrix = make_mavg_features(matrix, ['sales_lag_7','sales_lag_28' ],['id'], [7,28] )
matrix = make_mavg_features(matrix, ['sales_lag_7','sales_lag_28' ],['item_id'], [28] )
matrix = make_mavg_features(matrix, ['sales_lag_7','sales_lag_28' ],['item_id','state_id'], [28] )
time.time() - ts









from sklearn.utils import shuffle
matrix = shuffle(matrix, random_state = 0)

to_drop = ['id','d','sales','date', 'wm_yr_wk']
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
train_cols = matrix.columns[~matrix.columns.isin(to_drop)]

valid_ind = np.random.choice(matrix.index.values, 1_000_000, replace = False)
train_ind = np.setdiff1d(matrix.index.values, valid_ind)
print(train_ind[:5])

import gc
import lightgbm as lgb 

matrix.info()
train_data = lgb.Dataset(matrix.drop(to_drop, axis = 1).loc[train_ind], label = matrix['sales'].loc[train_ind], 
                         categorical_feature=cat_feats, free_raw_data=False)
fake_valid_data = lgb.Dataset(matrix.drop(to_drop, axis = 1).loc[valid_ind], label = matrix['sales'].loc[valid_ind],
                              categorical_feature=cat_feats,
                 free_raw_data=False)
del matrix ; gc.collect()
del valid_ind, train_ind; gc.collect()

params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}

m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 

fig, ax = plt.subplots(figsize=(12,6))
lgb.plot_importance(m_lgb, max_num_features=30, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)




def make_test_lags(df, day, cols,lags):
    for col in cols:
        for lag in lags:
            copy = df.loc[df.date == day -timedelta(days = lag)]
            copy.rename(columns = {col: col+'_lag_'+str(lag)}, inplace = True)
            copy.d = copy.d + lag 
            copy = copy[['d','item_id','store_id',col+'_lag_'+str(lag)]]
            
            df = pd.merge(df, copy, on = ['d','item_id', 'store_id'], how = 'left')
            print('made_lags')
    return df

def make_mavg_test(df,day,features, by, windows):
    by1 = ''
    for i in by:
        by1 += f'{i}_'
    for feature in features: 
        lag = int(feature[-1:])
        for window in windows:
                
                df_window = df[(df.date <= day-timedelta(days=lag)) & (df.date > day-timedelta(days=lag+window))]
                if len(by) == 1:
                    df_window_grouped = df_window.groupby(by).agg({'sales':'mean'}).reindex(df.loc[df.date==day,by[0]])
                else:
                    df_window_grouped = df_window.groupby(by).agg({'sales':'mean'}).reindex(df.loc[df.date==day,by])
                df.loc[df.date == day,f'mavg_{feature}_{by1}_{window}'] = df_window_grouped.sales.values
    return df 
    
            
def create_date_features_for_test(df):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16")
        else:
            df[date_feat_name] = getattr(
                df["date"].dt, date_feat_func).astype("int16")
    
    
alphas = [1.028, 1.023, 1.018]
weights = [1/len(alphas)]*len(alphas)  # equal weights
fday = datetime(2016,4, 25) 
te0 = make_matrix(1800,train = False)  # create master copy of `te`
create_date_features_for_test(te0)

for icount, (alpha, weight) in enumerate(zip(alphas,weights)):
    te = te0.copy()  # just copy
    cols = [f"F{i}" for i in range(1, 29)]
    for tdelta in range(0,28):
        day = fday + timedelta(days = tdelta)
        print(tdelta,day)
        tst = te[(te.date >= day - timedelta(days=57)) & (te.date <= day)].copy()
        tst = make_test_lags(tst,day,['sales'],[7,28])
        tst = make_mavg_test(tst,day,['sales_lag_7','sales_lag_28' ],['id'], [7,28])
        tst = make_mavg_test(tst,day,['sales_lag_7','sales_lag_28' ],['item_id'], [28])
        tst = make_mavg_test(tst,day,['sales_lag_7','sales_lag_28' ],['item_id','state_id'], [28])  
        tst
        tst = tst.loc[tst.date == day, train_cols]
        te.loc[te.date == day, "sales"] = alpha * m_lgb.predict(tst)      
    
    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")[
        "id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()[
        "sales"][cols].reset_index()
    te_sub.fillna(0., inplace=True)
    te_sub.sort_values("id", inplace=True)
    te_sub.reset_index(drop=True, inplace=True)
    te_sub.to_csv(f"submission_{icount}.csv", index=False)
    if icount == 0:
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    print(icount, alpha, weight)
    
    
sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission.csv",index=False)    

        
        
        
        
    
    
    
    
    

