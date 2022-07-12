import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from scipy.sparse import vstack, hstack, coo_matrix

from functools import partial
from multiprocessing import cpu_count, Pool
 
cores = cpu_count() #Number of CPU cores on your system

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'
        }


# ip 364778, 126413
# app 768, 521
#device 4227, 3031
# os 956, 604
# channel 500, 498
#df = pd.read_csv('../input/train.csv', dtype=dtypes)
#for f in df.columns:
#    print(f, df[f].max())
#df = pd.read_csv('../input/test.csv', dtype=dtypes)
#for f in df.columns:
#    print(f, df[f].max())
##

hash_digit   =  (2 ** 8)  ** 2
#dummy one hot
features_cnt = {#'app': 769, 'device': 4228, 'os': 957, 'channel': 501, 
                'ip': hash_digit, 
                'app_channel': hash_digit, 'channel_os': hash_digit, 'app_channel_os': hash_digit,
                'dow': 7, 'woy': 53, 'moy': 12, 'hod': 24, 'qoh': 4, 'dteom': 31,}
enc          = OneHotEncoder(dtype=np.bool, handle_unknown='ignore')
features     = list(features_cnt.keys())
enc.fit(pd.DataFrame({f: [np.uint16(i % features_cnt.get(f)) for i in range(hash_digit * 2)] for f in features})[features])
#enc.fit(pd.DataFrame({f: [i for i in range(8000)] for f in features}))
#onehot_mapping = {'app': 5000, 'device': 0, 'os': 6000, 'channel': 7000}

def hashtrick(df, interactions=[], digit=hash_digit):
    df['_'.join(interactions)] = df[interactions].apply(lambda x: '_'.join(str(x)), axis=1).apply(lambda x: hash(x) % digit)

def transform(df, labels=None, mapping=None, model=None):
    
    cols = [f for f in features if (f in df.columns)]
    for f in cols:
        if mapping is not None:
            df[f] = df[f].apply(lambda x: x + mapping.get(f, 0)).astype(np.uint16)
        elif f == 'ip':
            df[f] = df[f].apply(lambda x: x % hash_digit).astype(np.uint16)
        else:
            df[f] = df[f].astype(np.uint16) 
        gc.collect()
        
    ##assigned interactions
    for g in [f for f in features if '_' in f]:
        hashtrick(df, interactions=g.split('_'))
    
    #time
    if 'click_time' in df.columns:
        #https://www.kaggle.com/ogrellier/ftrl-in-chunck
        df["datetime"] = pd.to_datetime(df['click_time'])
        df['dow']      = df['datetime'].dt.dayofweek
        df['woy']      = df['datetime'].dt.week
        df['moy']      = df['datetime'].dt.month
        df['dteom']    = df['datetime'].dt.daysinmonth - df['datetime'].dt.day
        df['hod']      = df['datetime'].dt.hour
        df['qoh']      = df['datetime'].dt.quarter
    
    #print(df.shape)
    if labels is None:
        print('process {} sample'.format(len(df)))
        return model.predict_proba(enc.transform(df[features]))[:, 1]
    else:
#        model.partial_fit(enc.transform(df[cols]), df[labels], classes=np.array([0, 1],))
        print('process {} sample'.format(len(df)))
        return (enc.transform(df[features]), df[labels])

def process(filename, labels=None, chunksize=100000, model=None):
    #read features for train; read and predict for test
    func = partial(transform, labels=labels, model=model)
    pool = Pool(cores)
    if labels is None: #test
        return np.concatenate(pool.map(func, pd.read_csv(filename, dtype=dtypes, iterator=True, chunksize=chunksize)))
    else: #training
        print('read in data')
        res = pool.map(func, pd.read_csv(filename, dtype=dtypes, iterator=True, chunksize=chunksize))
        print('data loaded')
        for x, y in res:
            model.partial_fit(x, y, classes=np.array([0, 1],))
        print('trained 1 pass')
        for x, y in res:
            model.partial_fit(x, y, classes=np.array([0, 1],))
        print('trained 2 pass')

clf = SGDClassifier(loss='log', penalty='elasticnet', 
                    l1_ratio=0.15, max_iter=100, 
                    n_jobs=4, random_state=42, 
                    class_weight={1: 20, 0: 1}, warm_start=True)

process('../input/train.csv', labels='is_attributed', model=clf)
preds = process('../input/test.csv', model=clf)
print('predicted')
print(preds.shape)

##
submit_df = pd.read_csv('../input/sample_submission.csv', dtype=dtypes)
submit_df['is_attributed'] = preds
print(submit_df.head())
submit_df[['click_id', 'is_attributed']].to_csv('subm_lr.csv', index=False)
