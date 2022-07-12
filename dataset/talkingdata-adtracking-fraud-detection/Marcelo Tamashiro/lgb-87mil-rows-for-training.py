import numpy as np
import pandas as pd
import os
import gc
from contextlib import contextmanager
import psutil
import time

@contextmanager
def timer_memory(name):
    t0 = time.time()
    yield
    print(f'Memory: {(psutil.Process(os.getpid()).memory_info().rss/2**30):.02f}GB')
    print(f'{name} done in {time.time()-t0:.0f}s')

def next_click(df, cols,feat):
    name = '{}_nextclick'.format('_'.join(cols))
    df['ct'] = (df['click_time'].astype(np.int64)//10**9).astype(np.int32)
    df[name] = (df.groupby(cols).ct.shift(-1)-df.ct).astype(np.float32)
    df[name] = df[name].fillna(df[name].mean())
    df[name] = df[name].astype('uint32')
    df.drop(['ct'],axis=1,inplace=True)
    gc.collect()
    feat.append(name)
    print(f'{name} max: {df[name].max()}')
    return df,feat
    
def dcount(df ,cols, dtype,feat):
    name = '{}_count'.format('_'.join(cols))  
    gp = df[cols].groupby(cols).size().rename(name).to_frame().reset_index()
    df = pd.merge(df,gp,on=cols,how='left')
    df[name] = df[name].astype(dtype)
    del gp
    gc.collect()
    feat.append(name)
    print(f'{name} max: {df[name].max()}')
    return df,feat
    
def dcountun(df ,cols, dtype,feat):
    name = '{}_countun'.format('_'.join(cols))  
    gp = df[cols].groupby(cols[:len(cols)-1])[cols[len(cols)-1]].nunique().rename(name).to_frame().reset_index()
    df = pd.merge(df,gp,on=cols[:len(cols)-1],how='left')
    df[name] = df[name].astype(dtype)
    del gp
    gc.collect()
    feat.append(name)
    print(f'{name} max: {df[name].max()}')
    return df,feat

def dcumcount(df ,cols, dtype,feat):
    name = '{}_cumcount'.format('_'.join(cols))  
    df[name] = df[cols].groupby(cols).cumcount()+1
    df[name] = df[name].astype(dtype)
    gc.collect()
    feat.append(name)
    print(f'{name} max: {df[name].max()}')
    return df,feat   


if __name__ == '__main__':
    train_path = '../input/train.csv'
    test_path = '../input/test.csv'
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    test_cols  = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
    debug = False
    chunk = 87000000
    valsize = 5000000
    start = 184903891-chunk
    if debug == True:
        chunk = 45000
        valsize=50
        
    with timer_memory('Open files'):
        train = pd.read_csv(train_path, skiprows=range(1, start), usecols=train_cols,dtype=dtypes, nrows=chunk,parse_dates=['click_time'])
        test = pd.read_csv(test_path, usecols = test_cols, dtype=dtypes ,parse_dates=['click_time'])
        feat = ['app','device','os','channel','hour'] #list with features 
        cat = ['app','device','os','channel','hour'] #list with categorical features
        merge: pd.DataFrame = pd.concat([train, test])
        print('Merge shape:',merge.shape)
        print('Merge shape:\n ',merge.dtypes)
        del train
        del test
        gc.collect()
    
    with timer_memory('New features'):
        merge['hour'] = pd.to_datetime(merge.click_time).dt.hour.astype('uint8')
        merge['day'] = pd.to_datetime(merge.click_time).dt.day.astype('uint8')
        #merge , feat = next_click(merge,['app','channel'],feat)
        merge , feat = next_click(merge,['ip','os','device','app'],feat)
        merge.drop(['click_time'], axis = 1, inplace = True) ; gc.collect()
        merge['in_test_hh'] = (2-merge['hour'].isin([4, 5, 9, 10, 13, 14])).astype('uint8')
        merge , feat = dcount(merge , ['ip','day','in_test_hh'],'uint32',feat)
        merge.drop(['in_test_hh'], axis = 1, inplace = True) ; gc.collect()
        merge , feat = dcount(merge , ['day','hour','app','channel'],'uint16',feat)
        merge , feat = dcount(merge , ['ip','day','hour','device'],'uint16',feat)
        merge , feat = dcount(merge , ['ip','app'],'uint32',feat)
        merge , feat = dcountun(merge , ['day','hour','app','channel'],'uint8',feat)
        merge.drop(['day'], axis = 1, inplace = True) ; gc.collect()
        merge , feat = dcountun(merge , ['ip','app'],'uint8',feat)
        merge , feat = dcountun(merge , ['ip','channel'],'uint8',feat)

    with timer_memory('Preparing for training'):
        import lightgbm as lgb
        train = merge[:chunk-valsize]
        val = merge[chunk-valsize:chunk] #validation, last 5mil rows from train dataset
        test = merge[chunk:]
        del merge
        gc.collect()
        y_train = (pd.read_csv(train_path, skiprows=range(1, start), usecols=['is_attributed'],dtype='uint8',nrows=chunk-valsize))['is_attributed'].values
        y_val = (pd.read_csv(train_path, skiprows=range(1, start+chunk-valsize), usecols=['is_attributed'],dtype='uint8',nrows=valsize))['is_attributed'].values
        d_train = lgb.Dataset(train[feat].values.astype(np.float32),label=y_train, feature_name=feat,categorical_feature=cat)
        d_valid = lgb.Dataset(val[feat].values.astype(np.float32),label=y_val,feature_name=feat, categorical_feature=cat)
        del train
        del val
        del y_train
        del y_val
        gc.collect()
        
    with timer_memory('Training'):
        params = {"objective": "binary",
          'metric': {'auc'},
          "boosting_type": "gbdt",
          "verbosity": -1,
          "num_threads": 4,
          "bagging_fraction": 0.8,
          "feature_fraction": 0.8,
          "learning_rate": 0.08, 
          "num_leaves": 90,
          'max_depth': 7,
          "verbose": -1,
          "min_split_gain": .3,
          "reg_alpha": .3,
          'scale_pos_weight': 99.7, 
          'two_round':True}
        model = lgb.train(params,train_set=d_train,num_boost_round=2500,valid_sets=[d_valid],verbose_eval=True,early_stopping_rounds=25)
        del d_train
        del d_valid
        gc.collect()
    
    with timer_memory('Predict and submission file'):
        test['click_id'] = pd.read_csv(test_path, usecols = ['click_id'], dtype='uint32')['click_id'].values
        test['click_id'] = test['click_id'].astype('uint32')
        gc.collect()
        test['is_attributed'] = model.predict(test[feat].values.astype(np.float32),num_iteration=model.best_iteration)
        test[['click_id','is_attributed']].to_csv('lgb87mil.csv',index=False)