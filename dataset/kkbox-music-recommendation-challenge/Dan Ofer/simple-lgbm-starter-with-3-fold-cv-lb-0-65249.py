# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold

df_test = pd.read_csv('../input/test.csv',dtype={'msno' : 'category',
												'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                'song_id' : 'category'})

df_train = pd.read_csv('../input/train.csv',dtype={'msno' : 'category',
													'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})

df_members = pd.read_csv('../input/members.csv',dtype={'city' : 'category',
                                                      'bd' : 'category',
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'})



df_members['registration_year'] = df_members['registration_init_time'].apply(lambda x : int(str(x)[0:4]))
df_members['registration_year'] = pd.to_numeric(df_members['registration_year'],downcast='unsigned')

df_members['registration_month'] = df_members['registration_init_time'].apply(lambda x : int(str(x)[4:6]))
df_members['registration_month'] = pd.to_numeric(df_members['registration_month'],downcast='unsigned')

df_members['registration_date'] = df_members['registration_init_time'].apply(lambda x : int(str(x)[6:8]))
df_members['registration_date'] = pd.to_numeric(df_members['registration_date'],downcast='unsigned')

df_members.drop('registration_init_time',axis=1,inplace=True)

df_members['expiration_year'] = df_members['expiration_date'].apply(lambda x : int(str(x)[0:4]))
df_members['expiration_year'] = pd.to_numeric(df_members['expiration_year'],downcast='unsigned')

df_members['expiration_month'] = df_members['expiration_date'].apply(lambda x :int(str(x)[4:6]))
df_members['expiration_month'] = pd.to_numeric(df_members['expiration_month'],downcast='unsigned')

df_members['expiration_date'] = df_members['expiration_date'].apply(lambda x : int(str(x)[6:8]))
df_members['expiration_date'] = pd.to_numeric(df_members['expiration_date'],downcast='unsigned')

df_test = pd.merge(left = df_test,right = df_members,how='left',on='msno')
df_test.msno = df_test.msno.astype('category')
df_train = pd.merge(left = df_train,right = df_members,how='left',on='msno')
df_train.msno = df_train.msno.astype('category')

del df_members


df_songs = pd.read_csv('../input/songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})


df_test = pd.merge(left = df_test,right = df_songs,how = 'left',on='song_id')
df_test.song_length.fillna(200000,inplace=True)
df_test.song_length = df_test.song_length.astype(np.uint32)
df_test.song_id = df_test.song_id.astype('category')

df_train = pd.merge(left = df_train,right = df_songs,how = 'left',on='song_id')
df_train.song_length.fillna(200000,inplace=True)
df_train.song_length = df_train.song_length.astype(np.uint32)
df_train.song_id = df_train.song_id.astype('category')

del df_songs

import lightgbm as lgb

kf = KFold(n_splits=3)

predictions = np.zeros(shape=[len(df_test)])

for train_indices,val_indices in kf.split(df_train) : 
    train_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[train_indices,:],label=df_train.loc[train_indices,'target'])
    val_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[val_indices,:],label=df_train.loc[val_indices,'target'])
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.1 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 128,
        'max_depth': 10,
        'num_rounds': 200,
        } 
    
    bst = lgb.train(params, train_data, 100, valid_sets=[val_data])
    predictions+=bst.predict(df_test.drop(['id'],axis=1))
    del bst
    
predictions = predictions/3

submission = pd.read_csv('../input/sample_submission.csv')
submission.target=predictions
submission.to_csv('submission.csv',index=False)