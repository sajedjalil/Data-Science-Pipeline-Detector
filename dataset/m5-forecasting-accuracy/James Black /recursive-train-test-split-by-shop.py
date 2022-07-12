import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import datetime
from datetime import timedelta, datetime
import gc
import lightgbm as lgb 
import random 
ptypes = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
ctypes = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
SEED = 13           
random.seed(SEED)     
np.random.seed(SEED)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

matrix0 = pd.read_pickle('/kaggle/input/matrix-0/matrix_0.pkl')
valid_data0 = matrix0.loc[(matrix0.d>1885)]
matrix0 = matrix0.loc[matrix0.d>365]
#matrix= matrix0.loc[matrix0.store_id == shop].copy()
matrix0.drop(matrix0.loc[matrix0.d>1885].index, inplace = True)
matrix0.info()
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
        for i in range(1886,1913):
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
    if date_feat_name in matrix0.columns:
        matrix0[date_feat_name] = matrix0[date_feat_name].astype("int16")
    else:
        matrix0[date_feat_name] = getattr(matrix0["date"].dt, date_feat_func).astype("int16")


def create_fea(df, bys, lags):
    ''''lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id","sales"]].groupby("id")["sales"].shift(lag)'''

    wins = [7, 28]
    by1 = ''.join(bys)
    for win in wins :
        for lag in lags:
            df[f'mavg_{lag}_{by1}_{win}'] = pd.concat([df[bys],df['sales']], axis = 1).groupby(bys).transform(lambda x : x.rolling(win).mean())
            df[f'mavg_{lag}_{by1}_{win}'] = df[f'mavg_{lag}_{by1}_{win}'].astype(np.float16)

create_fea(matrix0,['id'],[7,28])
create_fea(matrix0,['item_id'],[7,28])
#create_fea(matrix0,['item_id','state_id'],[7,28])

for shop in range(10):    
    matrix= matrix0.loc[matrix0.store_id == shop].copy()
    from sklearn.utils import shuffle
    matrix = shuffle(matrix, random_state = 0)

    to_drop = ['id','d','sales','date', 'wm_yr_wk']
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    train_cols = matrix.columns[~matrix.columns.isin(to_drop)]
    
    np.random.seed(SEED) 
    valid_ind = np.random.choice(matrix.index.values, 100_000, replace = False)
    train_ind = np.setdiff1d(matrix.index.values, valid_ind)



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
        'num_iterations' : 200,
        'num_leaves': 2*11,
        "min_data_in_leaf": 2*12,
        'seed': SEED
    }

    
    model_name = f'model_{shop}'
    globals()[model_name] = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 

models = [model_0, model_1,model_2,model_3,model_4,model_5,model_6,model_7, model_8, model_9]
   



def create_lag_features_for_test(dt, day, by, lags):
    # create lag feaures just for single day (faster)
    ''''lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt.date == day, lag_col] = \
            dt.loc[dt.date ==day-timedelta(days=lag), 'sales'].values  # !!! main'''

    windows = [7, 28]
    by1 = ''.join(by)
    for win in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+win))]
            df_window_grouped = df_window.groupby(by).agg({'sales':'mean'})#.reindex(dt.loc[dt.date==day,by[0]])
            df_window_grouped.rename(columns ={'sales':f'mavg_{lag}_{by1}_{win}'}, inplace =True)   
            newcol = dt.loc[dt.date == day].merge(df_window_grouped, how = 'left', on = by)[f'mavg_{lag}_{by1}_{win}']
            dt.loc[dt.date == day,f'mavg_{lag}_{by1}_{win}'] = newcol.values      
            
            #dt.loc[dt.date == day,f'mavg_{lag}_{by1}_{win}'] = \
            #               df_window_grouped.sales.values     

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


#alphas = [1.028, 1.023, 1.018]
#weights = [1/len(alphas)]*len(alphas)  # equal weights
fday = datetime(2016,3, 28) 
te0 = make_matrix(1800,train = False) 
#te0 = te0.loc[te0.store_id == shop]
create_date_features_for_test(te0)
max_lags = 57

#for icount, (alpha, weight) in enumerate(zip(alphas,weights)):
te = te0.copy()  # just copy

te.loc[te.d>1885,'sales'] = np.nan
#cols = [f"F{i}" for i in range(1, 29)
for tdelta in range(0, 28):
    day = fday + timedelta(days=tdelta)
    print(tdelta, day.date())
    tst = te[(te.date >= day - timedelta(days=max_lags))
             & (te.date <= day)].copy()

    create_lag_features_for_test(tst, day,['id'],[7,28])
    create_lag_features_for_test(tst, day,['item_id'],[7,28])
    #create_lag_features_for_test(tst, day,['item_id','state_id'] ,[7,28])
    for shop, model in zip(range(10), models):
        tst_shop = tst.loc[tst.store_id == shop ]

        tst_shop = tst_shop.loc[tst_shop.date == day, train_cols]
        te.loc[(te.date == day)&(te.store_id == shop), "sales"] =  model.predict(tst_shop)

preds0 = te.loc[(te.d > 1885),['sales', 'store_id']].copy()
from sklearn.metrics import mean_squared_error

scores = []
for shop in range(10):
    valid_data = valid_data0.loc[valid_data0.store_id == shop]['sales']
    preds = preds0.loc[preds0.store_id == shop]['sales']
    mse = mean_squared_error
    print(shop, np.sqrt(mse(preds, valid_data)))
    error = np.sqrt(mse(preds, valid_data))
    scores.append(error)

print(scores)
print(np.mean(scores))    


