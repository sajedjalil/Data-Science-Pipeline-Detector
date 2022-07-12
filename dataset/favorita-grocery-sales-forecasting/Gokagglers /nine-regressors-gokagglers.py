
# Starter code for multiple regressors implemented by Leandro dos Santos Coelho
# Source code based on Forecasting Favorites, 1owl
# https://www.kaggle.com/the1owl/forecasting-favorites , version 10


import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, metrics
import gc; gc.enable()
import random

# classical tree approach
from sklearn.tree import DecisionTreeRegressor

# ensemble approaches
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# linear approaches
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.linear_model import Ridge, HuberRegressor

import time

np.random.seed(888)

# store the total processing time
start_time = time.time()
tcurrent   = start_time

print('Multiple regressors\n')
print('Datasets reading')


# read datasets
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}
data = {
    'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),
    'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),
    'ite': pd.read_csv('../input/items.csv'),
    'sto': pd.read_csv('../input/stores.csv'),
    'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),
    'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),
    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),
    }


# dataset processing
print('Datasets processing')

train = data['tra'][(data['tra']['date'].dt.month == 8) & (data['tra']['date'].dt.day > 15)]
del data['tra']; gc.collect();
target = train['unit_sales'].values
target[target < 0.] = 0.
train['unit_sales'] = np.log1p(target)

def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

def df_transform(df):
    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['date'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    df = df.fillna(-1)
    return df

data['ite'] = df_lbl_enc(data['ite'])
train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])
test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])
del data['tes']; gc.collect();
del data['ite']; gc.collect();

train = pd.merge(train, data['trn'], how='left', on=['date','store_nbr'])
test = pd.merge(test, data['trn'], how='left', on=['date','store_nbr'])
del data['trn']; gc.collect();
target = train['transactions'].values
target[target < 0.] = 0.
train['transactions'] = np.log1p(target)

data['sto'] = df_lbl_enc(data['sto'])
train = pd.merge(train, data['sto'], how='left', on=['store_nbr'])
test = pd.merge(test, data['sto'], how='left', on=['store_nbr'])
del data['sto']; gc.collect();

data['hol'] = data['hol'][data['hol']['locale'] == 'National'][['date','transferred']]
data['hol']['transferred'] = data['hol']['transferred'].map({'False': 0, 'True': 1})
train = pd.merge(train, data['hol'], how='left', on=['date'])
test = pd.merge(test, data['hol'], how='left', on=['date'])
del data['hol']; gc.collect();

train = pd.merge(train, data['oil'], how='left', on=['date'])
test = pd.merge(test, data['oil'], how='left', on=['date'])
del data['oil']; gc.collect();

train = df_transform(train)
test = df_transform(test)
col = [c for c in train if c not in ['id', 'unit_sales','perishable','transactions']]

x1 = train[(train['yea'] != 2016)]
x2 = train[(train['yea'] == 2016)]
del train; gc.collect();

y1 = x1['transactions'].values
y2 = x2['transactions'].values

def NWRMSLE(y, pred, w):
    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5


#------------------- forecasting based on multiple regressors (r) models
    
print('\nRunning the basic regressors ...')    

number_regressors_to_test = 12

for method in range(1, number_regressors_to_test+1):
    
    # set the seed to generate random numbers
    ra1 = round(method + 123*method + 456*method) 
    np.random.seed(ra1)
    
    
    print('\nmethod = ', method)
    
    if (method==1):
        print('Linear model (classical)')
        str_method = 'Linear model'    
        r = linear_model.LinearRegression(n_jobs=-1)

    if (method==2):
        print('Extra trees 01')
        str_method = 'ExtraTrees01'
        r = ExtraTreesRegressor(n_estimators=120, max_depth=2, n_jobs=-1, 
                                 random_state=ra1, verbose=0, warm_start=True)

    if (method==3):
        print('Extra trees 02')
        str_method = 'ExtraTrees02'
        r = ExtraTreesRegressor(n_estimators=120, max_depth=3, n_jobs=-1, 
                                 random_state=ra1, verbose=0, warm_start=True)

    if (method==4):
        print('Random forest 01')
        str_method = 'RandomForest01'
        r = RandomForestRegressor(n_estimators=120, max_depth=2, n_jobs=-1, 
                                   random_state=ra1, verbose=0, warm_start=True)

    if (method==5):
        print('Random forest 02')
        str_method = 'RandomForest02'
        r = RandomForestRegressor(n_estimators=120, max_depth=3, n_jobs=-1, 
                                   random_state=ra1, verbose=0, warm_start=True)
        
    if (method==6):
        print('ElasticNet')
        str_method = 'Elastic Net'
        r = ElasticNetCV()
        
    if (method==7):
        print('GradientBoosting 01')
        str_method = 'GradientBoosting01'
        r = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate = 0.05, 
                                       random_state=ra1, verbose=0, warm_start=True,
                                       subsample= 0.75, max_features = 0.35)
    if (method==8):
        print('GradientBoosting 02')
        str_method = 'GradientBoosting02'
        r = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate = 0.05, 
                                       random_state=ra1, verbose=0, warm_start=True,
                                       subsample= 0.80, max_features = 0.40)        
                                       
    if (method==9):
        print('GradientBoosting 03')
        str_method = 'GradientBoosting03'
        r = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate = 0.05, 
                                       random_state=ra1, verbose=0, warm_start=True,
                                       subsample= 0.85, max_features = 0.45)   
                                       
    if (method==10):
        print('Decision Tree')
        str_method = 'DecisionTree'
        r = DecisionTreeRegressor(max_depth=4)

    if (method==11):
        print('Ridge')
        str_method = 'Ridge'
        r = Ridge()
        
    if (method==12):
        print('Huber')
        str_method = 'Huber'
        r = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=120, epsilon=epsilon)
        
        
    r.fit(x1[col], y1)


    a1 = NWRMSLE(y2, r.predict(x2[col]), x2['perishable'])
    # part of the output file name
    N1 = str(a1)
    
    test['transactions'] = r.predict(test[col])
    test['transactions'] = test['transactions'].clip(lower=0.+1e-15)

    col = [c for c in x1 if c not in ['id', 'unit_sales','perishable']]
    y1 = x1['unit_sales'].values
    y2 = x2['unit_sales'].values


    # set a new seed to generate random numbers
    ra2 = round(method + 901*method + 12*method) 
    np.random.seed(ra2)

    if (method==1):
        r = linear_model.LinearRegression(n_jobs=-1)
 
    if (method==2):
        r = ExtraTreesRegressor(n_estimators=120, max_depth=2, n_jobs=-1, 
                                 random_state=ra2, verbose=0, warm_start=True)

    if (method==3):
        r = ExtraTreesRegressor(n_estimators=120, max_depth=3, n_jobs=-1, 
                                 random_state=ra2, verbose=0, warm_start=True)

    if (method==4):
        r = RandomForestRegressor(n_estimators=120, max_depth=2, n_jobs=-1, 
                                   random_state=ra2, verbose=0, warm_start=True)

    if (method==5):
        r = RandomForestRegressor(n_estimators=120, max_depth=3, n_jobs=-1, 
                                   random_state=ra2, verbose=0, warm_start=True)

    if (method==6):
        r = ElasticNetCV()
        
    if (method==7):
        r = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate = 0.05, 
                                       random_state=ra2, verbose=0, warm_start=True,
                                       subsample= 0.75, max_features = 0.35)
    if (method==8):
        r = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate = 0.05, 
                                       random_state=ra2, verbose=0, warm_start=True,
                                       subsample= 0.80, max_features = 0.40)
                                       
    if (method==9):
        r = GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate = 0.05, 
                                       random_state=ra2, verbose=0, warm_start=True,
                                       subsample= 0.85, max_features = 0.45)    
                        
    if (method==10):
        r = DecisionTreeRegressor(max_depth=4)
        
    if (method==11):
        r = Ridge() 
        
    if (method==12):
        r = HuberRegressor(fit_intercept=True, alpha=0.001, max_iter=120, epsilon=epsilon)
        

    r.fit(x1[col], y1)
    
    a2 = NWRMSLE(y2, r.predict(x2[col]), x2['perishable'])
    # part of the output file name
    N2 = str(a2)

    print('Performance: NWRMSLE(1) = ',a1,'NWRMSLE(2) = ',a2)

    test['unit_sales'] = r.predict(test[col])
    cut = 0.+1e-12 # 0.+1e-15
    test['unit_sales'] = (np.exp(test['unit_sales']) - 1).clip(lower=cut)

    output_file = 'sub v05 ' + str(str_method) + ' method ' + str(method) + N1 + ' - ' + N2 + '.csv'
 
    test[['id','unit_sales']].to_csv(output_file, index=False, float_format='%.2f')


print( "\nFinished ...")
nm=(time.time() - start_time)/60
print ("Total time %s min" % nm)

