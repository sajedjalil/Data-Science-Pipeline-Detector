# Temi Babs
# 
# Naive Bayes
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import gc

print(os.listdir("../input"))

path = '../input/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

# we save only day 9
def procedure(x):
    if x == 1:
        print('load train....')
        train_df = pd.read_csv(path+"train.csv", dtype=dtypes, skiprows = range(1, 131886954),
                            usecols=['ip','app','device','os', 'channel', 'click_time',
                            'is_attributed'], parse_dates=['click_time'])
        print('load test....')
        test_df = pd.read_csv(path+"test.csv", dtype=dtypes,
                    usecols=['ip','app','device','os', 'channel',
                    'click_time', 'click_id'], parse_dates=['click_time'])
        len_train = len(train_df)
        train_df=train_df.append(test_df)
        del test_df; gc.collect()
    else:
        train_df = pd.read_csv(path+"train.csv", dtype=dtypes, nrows=40886954,
                            usecols=['ip','app','device','os', 'channel', 'click_time',
                            'is_attributed'], parse_dates=['click_time'])
    

    print('click time....')
    train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    train_df['next_click'] = (train_df.groupby(['ip', 'app','device',
                    'os']).click_time.shift(-1) - train_df.click_time).astype(np.float32)
    train_df['next_click'].fillna((train_df['next_click'].mean()), inplace=True)

    print('hour, day, wday....')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
    print('grouping by ip-day-hour combination....')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp; gc.collect()
    print('group by ip-app combination....')
    gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp; gc.collect()

    print('group by ip-app-os combination....')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app',
                'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp; gc.collect()
    print("vars and data type....")
    train_df['qty'] = train_df['qty'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    print("label encoding....")
    from sklearn.preprocessing import LabelEncoder
    train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
    print ('final part of preparation....')
    if x==1:
        test_df = train_df[len_train:]
        train_df = train_df[:len_train]
    y_train = train_df['is_attributed'].values
    if x==1:
        train_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)

    gc.collect()
 
    # train_df = get_keras_data(train_df)
    if x==1:
        return train_df, y_train, test_df
    else:
        return train_df, y_train
    
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB(priors=[0.05, 0.95])
for i in range(0, 2):
    print('Loading batch {}'.format(i))
    gc.collect()
    if i==0:
        train_df, y_train = procedure(i)
    else:
        train_df, y_train, test_df = procedure(i)
    model.fit(train_df, y_train)
    del train_df, y_train
    gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
# test_df = get_keras_data(test_df)
print("predicting....")
ans = model.predict_proba(test_df)
print(ans)
sub['is_attributed'] = [1.0-max(i) for i in ans]
print(sub)
del test_df; gc.collect()
print("writing....")
sub.to_csv('imbalanced_data.csv',index=False)