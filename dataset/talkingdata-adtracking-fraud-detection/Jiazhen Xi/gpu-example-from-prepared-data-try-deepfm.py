## ADD: 
## deepFM model
## Reference paper: https://arxiv.org/abs/1703.04247
## fm part: https://www.kaggle.com/qqgeogor/keras-based-fm
## tensorflow implementation: https://github.com/ChenglongChen/tensorflow-DeepFM


BATCHSIZE = 2**14
EPOCHS = 3
LR = 1e-3

import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GaussianDropout, merge
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
import gc

import os
print(os.listdir("../input"))




##### THE TRAINING DATA #####

df = pd.read_pickle('../input/preprocessed-subsamples-including-all-positives/sample0.pkl.gz')

max_app = df['app'].max()+1
max_ch = df['channel'].max()+1
max_dev = df['device'].max()+1
max_os = df['os'].max()+1
max_h = df['hour'].max()+1
max_agg1 = df['nip_day_test_hh'].max()+1
max_agg2 = df['nip_day_hh'].max()+1
max_agg3 = df['nip_hh_os'].max()+1
max_agg4 = df['nip_hh_app'].max()+1
max_agg5 = df['nip_hh_dev'].max()+1

df['agg1_float'] = df['nip_day_test_hh'].astype('float32') / np.float32(max_agg1)
df['agg2_float'] = df['nip_day_hh'].astype('float32') / np.float32(max_agg2)
df['agg3_float'] = df['nip_hh_os'].astype('float32') / np.float32(max_agg3)
df['agg4_float'] = df['nip_hh_app'].astype('float32') / np.float32(max_agg4)
df['agg5_float'] = df['nip_hh_dev'].astype('float32') / np.float32(max_agg5)
print( df.info() )

y_train = df['is_attributed'].values
df.drop(['is_attributed'],1,inplace=True)

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'agg1': np.array(dataset.nip_day_test_hh),
        'agg2': np.array(dataset.nip_day_hh),
        'agg3': np.array(dataset.nip_hh_os),
        'agg4': np.array(dataset.nip_hh_app),
        'agg5': np.array(dataset.nip_hh_dev),
        'agg1_float': np.array(dataset.agg1_float),
        'agg2_float': np.array(dataset.agg2_float),
        'agg3_float': np.array(dataset.agg3_float),
        'agg4_float': np.array(dataset.agg4_float),
        'agg5_float': np.array(dataset.agg5_float),
    }
    return X
df = get_keras_data(df)




#### THE MODEL ####

emb_n = 50
dense_n1 = 1000
dense_n2 = 1400
dense_n3 = 200
dense_n4 = 40
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev) #emb_n+30
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
emb_h = Embedding(max_h, emb_n)(in_h) #emb_n-33
in_agg1 = Input(shape=[1], name = 'agg1')
emb_agg1 = Embedding(max_agg1, emb_n)(in_agg1) #emb_n-10
in_agg2 = Input(shape=[1], name = 'agg2')
emb_agg2 = Embedding(max_agg2, emb_n)(in_agg2) #emb_n-10
in_agg3 = Input(shape=[1], name = 'agg3')
emb_agg3 = Embedding(max_agg3, emb_n)(in_agg3) #emb_n-10
in_agg4 = Input(shape=[1], name = 'agg4')
emb_agg4 = Embedding(max_agg4, emb_n)(in_agg4) #emb_n-10
in_agg5 = Input(shape=[1], name = 'agg5')
emb_agg5 = Embedding(max_agg5, emb_n)(in_agg5) #emb_n-10

agg1_float = Input(shape=[1], dtype='float32', name='agg1_float')
agg2_float = Input(shape=[1], dtype='float32', name='agg2_float')
agg3_float = Input(shape=[1], dtype='float32', name='agg3_float')
agg4_float = Input(shape=[1], dtype='float32', name='agg4_float')
agg5_float = Input(shape=[1], dtype='float32', name='agg5_float')

emb_layers = [(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                (emb_agg1), (emb_agg2), (emb_agg3), (emb_agg4), (emb_agg5)]

fe = concatenate(emb_layers)
embs = GaussianDropout(0.2)(fe)
embs = Flatten()(embs)

#%% DEEP PART
x = concatenate([embs, agg1_float, agg2_float, agg3_float, agg4_float, agg5_float])
x = (BatchNormalization())(x)
x = GaussianDropout(0.2)(Dense(dense_n1,activation='relu')(x))
x = (BatchNormalization())(x)
x = GaussianDropout(0.3)(Dense(dense_n2,activation='relu')(x))
x = (BatchNormalization())(x)
x = GaussianDropout(0.25)(Dense(dense_n3,activation='relu')(x))
x = (BatchNormalization())(x)
deep_out = GaussianDropout(0.2)(Dense(dense_n4,activation='relu')(x))

#%% FM PART
first_order = [Flatten()(emb1) for emb1 in emb_layers]
from itertools import combinations
second_order = []
for emb1, emb2 in combinations(emb_layers, 2):
    dot_layer = merge([Flatten()(emb1), Flatten()(emb2)], 
                       mode='dot', dot_axes=1)
    second_order.append(dot_layer)
first_order = merge(first_order, mode='sum')
second_order = merge(second_order, mode='sum')

#%% TOTAL
total_out = concatenate([(deep_out), (first_order), (second_order)])

outp = Dense(1,activation='sigmoid')(total_out)
model = Model(inputs=[in_app, in_ch, in_dev, in_os, in_h,
                      in_agg1, in_agg2, in_agg3, in_agg4, in_agg5,
                      agg1_float, agg2_float, agg3_float, agg4_float, agg5_float], outputs=outp)

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=LR),metrics=['accuracy'])

model.summary()
#%% 



##### THE FIT #####

model.fit(df, y_train, batch_size=BATCHSIZE, epochs=EPOCHS, shuffle=True, verbose=2)




##### THE TEST DATA #####

del y_train
del df
gc.collect()

print('load test...')
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'click_id'      : 'uint32'
        }
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
df = pd.read_csv("../input/talkingdata-adtracking-fraud-detection/test.csv", dtype=dtypes, usecols=test_cols)

gc.collect()

print('data prep...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df.drop(['click_time'], axis=1, inplace=True)
gc.collect()

df['in_test_hh'] = (   3 
                     - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                     - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

print('group by : ip_day_test_hh')
gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
         'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_test_hh'})
df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
del gp
df.drop(['in_test_hh'], axis=1, inplace=True)
df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
gc.collect()

print('group by : ip_day_hh')
gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_hh'})
df = df.merge(gp, on=['ip','day','hour'], how='left')
del gp
df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
gc.collect()

print('group by : ip_hh_os')
gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_os'})
df = df.merge(gp, on=['ip','os','hour','day'], how='left')
del gp
df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
gc.collect()

print('group by : ip_hh_app')
gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_app'})
df = df.merge(gp, on=['ip','app','hour','day'], how='left')
del gp
df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
gc.collect()

print('group by : ip_hh_dev')
gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_dev'})
df = df.merge(gp, on=['ip','device','day','hour'], how='left')
del gp
df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
gc.collect()

df.drop( ['ip','day'], axis=1, inplace=True )
gc.collect()

df['agg1_float'] = df['nip_day_test_hh'].astype('float32') / np.float32(max_agg1)
df['agg2_float'] = df['nip_day_hh'].astype('float32') / np.float32(max_agg2)
df['agg3_float'] = df['nip_hh_os'].astype('float32') / np.float32(max_agg3)
df['agg4_float'] = df['nip_hh_app'].astype('float32') / np.float32(max_agg4)
df['agg5_float'] = df['nip_hh_dev'].astype('float32') / np.float32(max_agg5)




##### THE PREDICTION #####

sub = pd.DataFrame()
sub['click_id'] = df['click_id'].astype('int')
df.drop(['click_id'],1,inplace=True)
df = get_keras_data(df)
gc.collect()

print("predicting....")
sub['is_attributed'] = model.predict(df, batch_size=BATCHSIZE, verbose=2)
del df; gc.collect()
print(sub.head())

print("writing....")
sub.to_csv('gpu_test_dfm.csv', index=False, float_format='%.7f') #'%.9f'
print("Done")