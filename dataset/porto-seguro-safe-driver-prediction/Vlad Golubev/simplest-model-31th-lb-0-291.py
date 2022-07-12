import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import re
 
def change_datatype(df): #minimize used memory
    for col in list(df.select_dtypes(include=['int']).columns):
        if df[col].max() < 2**7 and df[col].min() >= -2**7:
            df[col] = df[col].astype(np.int8)
        elif df[col].max() < 2**8 and df[col].min() >= 0:
            df[col] = df[col].astype(np.uint8)
        elif df[col].max() < 2**15 and df[col].min() >= -2**15:
            df[col] = df[col].astype(np.int16)
        elif df[col].max() < 2**16 and df[col].min() >= 0:
            df[col] = df[col].astype(np.uint16)
        elif df[col].max() < 2**31 and df[col].min() >= -2**31:
            df[col] = df[col].astype(np.int32)
        elif df[col].max() < 2**32 and df[col].min() >= 0:
            df[col] = df[col].astype(np.uint32)
    for col in list(df.select_dtypes(include=['float']).columns):
        df[col] = df[col].astype(np.float32)

def short_names(df): #just by self, for simple work with columns
    cols = {}
    for c in df.columns:
        if (not re.match(r'ps_.*',c)):
            continue
        t = ''
        e = ''
        if c.startswith('ps_calc'):
            t = 'F'
        elif c.startswith('ps_car'):
            t = 'C'
        elif c.startswith('ps_ind'):
            t = 'I'
        elif c.startswith('ps_reg'):
            t = 'R'
        
        if c.endswith('bin'):
            e = 'b'
        elif c.endswith('cat'):
            e = 'c'
        else:
            e = 'd'
        i = re.search('\d+',c).group(0)
        cols[c] = t+i+e
    change_datatype(df)
    return df.rename(columns=cols)

# transform features: some columns to OHE, some change NaN to mean, median or both
def transform(df, oh_columns, na_columns): 
    df = pd.get_dummies(df, columns=oh_columns, dummy_na=True, drop_first=False)
    for c in na_columns:
        if na_columns[c] == 0 or c not in df.columns:
            continue
        df[c] = df[c].replace(-1, np.NaN)
        if na_columns[c] == 1:
            df[c] = df[c].fillna(df[c].mean())
        elif na_columns[c] == 2:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c+'m'] = df[c].fillna(df[c].median())
            df[c] = df[c].fillna(df[c].mean())
    change_datatype(df)
    return df[df['_']], df[~df['_']]
    
def predictTarget(df, oh_columns, na_columns, kfold): #XGB kfold
    train, test = transform(df, oh_columns, na_columns)
    # I will explain later
    params = {'eta' : 0.025,
                'gamma' : 9,
                'max_depth' : 6,
                'reg_lambda' : 1.2,
                'colsample_bytree' : 1.0,
                'min_child_weight' : 10,
                'reg_alpha' : 8,
                'scale_pos_weight' : 1.6,
                'subsample' : 0.7,

                'eval_metric' : 'auc',
                'objective' : 'binary:logistic',
                'seed' : 2017,
                'silent' : False,
                'tree_method' : 'hist'}
    trains = np.array_split(train.sample(frac=1, random_state=200), kfold)

    # for tunning i used some columns with postfix '_', about it i will write later
    col = [c for c in train.columns if not c.endswith('_') and c not in ['id', 'target']]
    test['target'] = 0.0
    for i in range(kfold):
        valid = trains[i] 
        train = pd.concat(trains[:i]+trains[i+1:])
        dtrain = xgb.DMatrix(train[col], train['target'])
        dvalid = xgb.DMatrix(valid[col], valid['target'])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(params, dtrain, 99999, watchlist, verbose_eval=False, 
                          maximize=True, early_stopping_rounds=300)
        test['target'] += model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+35)/kfold
    return test
    
df = short_names(pd.read_csv('../input/train.csv'))
test = short_names(pd.read_csv('../input/test.csv'))
test['target'] = -1
df = pd.concat([df, test])
del test

# i want save maximum information for model, and drop only 'ps_calc*'
df = df.drop([c for c in df.columns if c.startswith('F')], axis=1)
df['_'] = df['target'] != -1
na_columns = {'R03d':2, 'C11d':2, 'C12d':1, 'C14d':1, 'I04c':3} #columns for change NaN - 1:mean, 2:median, 3:both
oh_columns = [c for c in df.columns if c.endswith('c') and not c in['C01c','I04c']] #columns for OHE(not all categorical should transform to OHE) 

kfold = 8
df = predictTarget(df, oh_columns, na_columns, kfold)
df[['id', 'target']].to_csv('../input/sub.csv', index = False)