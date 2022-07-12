# First things first, Random Forest model just hardly fits this TalkingData competiton with its execution time and memory usage.
# If you are looking for high scoring models, try research on LGB or XGB models.
# I make this simply because I love Random Forest, it is straight forward and easy to understand, you can read more from following link: http://www.codeastar.com/random-random-forest-tutorial/

# Updates: 
# - add unqiue feature sets count as new feature
# - add features group by count as new feature

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

#gc to run garbage collection manually
import gc, time

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

def handleClickHour(df):
    df['click_hour']= (pd.to_datetime(df['click_time']).dt.round('H')).dt.hour
    df['click_hour'] = df['click_hour'].astype('uint16')
    df = df.drop(['click_time'], axis=1)   
    return df
    
def countUniqueFeatGroupBy(df, groupby, countby, feature_name, feature_type='uint32'):   
    gp = df[groupby+[countby]].groupby(groupby)[countby].nunique().reset_index().rename(columns={countby:feature_name})
    df = df.merge(gp, on=groupby, how='left')
    del gp
    print ("Feature:[{}] Max value:[{}]".format(feature_name, df[feature_name].max()))
    df[feature_name] = df[feature_name].astype(feature_type)
    gc.collect()
    return df     

def countGroupByFreq(df, groupby, feature_name, feature_type='uint32'):   
    gp = df[groupby][groupby].groupby(groupby).size().rename(feature_name).to_frame().reset_index()
    df = df.merge(gp, on=groupby, how='left')
    del gp
    print ("Feature:[{}] Max value:[{}]".format(feature_name, df[feature_name].max()))
    df[feature_name] = df[feature_name].astype(feature_type)
    gc.collect()
    return  df      

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

#load training df (partly)
start_time = time.time()
df_train_30m = pd.read_csv('../input/train.csv', dtype=dtypes, skiprows=range(1,133333333), nrows=35333333, usecols=train_columns)
print('Load df_train_30m with [{}] seconds'.format(time.time() - start_time))

# Load testing df
start_time = time.time()
df_test = pd.read_csv('../input/test.csv', dtype=dtypes)
print('Load df_test with [{}] seconds'.format(time.time() - start_time))

train_record_index = df_train_30m.shape[0]

#handle click hour 
df_train_30m = handleClickHour(df_train_30m)
df_test = handleClickHour(df_test)
gc.collect()

#df for submit
df_submit = pd.DataFrame()
df_submit['click_id'] = df_test['click_id']

Learning_Y = df_train_30m['is_attributed']

#drop zone
df_test = df_test.drop(['click_id'], axis=1)
df_train_30m = df_train_30m.drop(['is_attributed'], axis=1)
gc.collect()

df_merge = pd.concat([df_train_30m, df_test])
del df_train_30m, df_test
gc.collect()

#group features to new features
start_time = time.time()
df_merge = countUniqueFeatGroupBy(df_merge, ['ip'], 'app', 'u_i_a', feature_type='uint16' ) ; gc.collect()
df_merge = countUniqueFeatGroupBy(df_merge, ['ip'], 'channel', 'u_i_c', feature_type='uint16' ) ; gc.collect()
df_merge = countUniqueFeatGroupBy(df_merge, ['ip'], 'device', 'u_i_d', feature_type='uint16' ) ; gc.collect()
df_merge = countUniqueFeatGroupBy(df_merge, ['ip', 'device', 'os'], 'app', 'u_ido_a', feature_type='uint16' ) ; gc.collect()
df_merge = countUniqueFeatGroupBy(df_merge, ['ip', 'device', 'os'], 'channel', 'u_ido_c', feature_type='uint16' ) ; gc.collect()
df_merge = countUniqueFeatGroupBy(df_merge, ['app'], 'channel', 'u_a_c', feature_type='uint16' ) ; gc.collect()
df_merge = countUniqueFeatGroupBy(df_merge, ['ip', 'app'], 'os', 'u_ia_o', feature_type='uint16' ) ; gc.collect()

df_merge = countGroupByFreq(df_merge, ['ip', 'device', 'os'], 'ido') ; gc.collect()
df_merge = countGroupByFreq(df_merge, ['ip', 'app'], 'ia') ; gc.collect()
df_merge = countGroupByFreq(df_merge, ['ip', 'app', 'os'], 'iao') ; gc.collect()
df_merge = countGroupByFreq(df_merge, ['ip', 'channel'], 'ic') ; gc.collect()
print('Grouping with [{}] seconds'.format(time.time() - start_time))

# Count ip for both train and test df 
start_time = time.time()
df_ip_count = df_merge['ip'].value_counts().reset_index(name='ip_count')
df_ip_count.columns = ['ip', 'ip_count']
print('Load df_ip_count with [{}] seconds'.format(time.time() - start_time))

df_merge = df_merge.merge(df_ip_count, on='ip', how='left', sort=False)
df_merge['ip_count'] = df_merge['ip_count'].astype('uint16')

df_merge = df_merge.drop(['ip'], axis=1)
del df_ip_count
gc.collect()

df_train = df_merge[:train_record_index]
df_test = df_merge[train_record_index:]

del df_merge
gc.collect()

#Use RandomForest
from sklearn.ensemble import RandomForestClassifier

VALIDATE = False

start_time = time.time()
rf = RandomForestClassifier(n_estimators=20, max_depth=14, random_state=7,verbose=2, oob_score=VALIDATE, n_jobs=4)
rf.fit(df_train, Learning_Y)
print('Train RandomForest df_train_30m with [{}] seconds'.format(time.time() - start_time))

if VALIDATE:
    print("OOB score: [{}]".format(rf.oob_score_))

importances = rf.feature_importances_
 
print ("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(df_train)), reverse=True)
print (sorted_feature_importance)

#predict
start_time = time.time()
predictions = rf.predict_proba(df_test)
print('Predict RandomForest df_train_30m with [{}] seconds'.format(time.time() - start_time))

df_submit['is_attributed'] = predictions[:,1]
df_submit.describe()

df_submit.to_csv('random_forest_talking_data_v3.csv', index=False)