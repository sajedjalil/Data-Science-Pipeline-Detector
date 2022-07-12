BATCHSIZE = 2**14
EPOCHS = 2
LR = 1e-3
NRUNS = 16
PUBLIC_CUTOFF = 4032690


import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GaussianDropout
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
import gc
import psutil
import os
from sklearn.metrics import roc_auc_score

process = psutil.Process(os.getpid())
print(os.listdir("../input"))


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
    


for i in range(NRUNS):


    ##### THE TRAINING DATA #####
    
    filename = '../input/preprocessed-train-subsamples-for-validation/train_sample' + str(i) + '.pkl.gz'
    df = pd.read_pickle(filename)
    
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

    y_train = df['is_attributed'].values
    df.drop(['is_attributed'],1,inplace=True)
    
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
    emb_dev = Embedding(max_dev, emb_n+30)(in_dev)
    in_os = Input(shape=[1], name = 'os')
    emb_os = Embedding(max_os, emb_n)(in_os)
    in_h = Input(shape=[1], name = 'h')
    emb_h = Embedding(max_h, emb_n-33)(in_h) 
    in_agg1 = Input(shape=[1], name = 'agg1')
    emb_agg1 = Embedding(max_agg1, emb_n-10)(in_agg1) 
    in_agg2 = Input(shape=[1], name = 'agg2')
    emb_agg2 = Embedding(max_agg2, emb_n-10)(in_agg2) 
    in_agg3 = Input(shape=[1], name = 'agg3')
    emb_agg3 = Embedding(max_agg3, emb_n-10)(in_agg3) 
    in_agg4 = Input(shape=[1], name = 'agg4')
    emb_agg4 = Embedding(max_agg4, emb_n-10)(in_agg4) 
    in_agg5 = Input(shape=[1], name = 'agg5')
    emb_agg5 = Embedding(max_agg5, emb_n-10)(in_agg5) 
    
    agg1_float = Input(shape=[1], dtype='float32', name='agg1_float')
    agg2_float = Input(shape=[1], dtype='float32', name='agg2_float')
    agg3_float = Input(shape=[1], dtype='float32', name='agg3_float')
    agg4_float = Input(shape=[1], dtype='float32', name='agg4_float')
    agg5_float = Input(shape=[1], dtype='float32', name='agg5_float')
    
    fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                      (emb_agg1), (emb_agg2), (emb_agg3), (emb_agg4), (emb_agg5)])
    embs = GaussianDropout(0.2)(fe)
    embs = Flatten()(embs)
    
    x = concatenate([embs, agg1_float, agg2_float, agg3_float, agg4_float, agg5_float])
    x = (BatchNormalization())(x)
    x = GaussianDropout(0.2)(Dense(dense_n1,activation='relu')(x))
    x = (BatchNormalization())(x)
    x = GaussianDropout(0.3)(Dense(dense_n2,activation='relu')(x))
    x = (BatchNormalization())(x)
    x = GaussianDropout(0.25)(Dense(dense_n3,activation='relu')(x))
    x = (BatchNormalization())(x)
    x = GaussianDropout(0.2)(Dense(dense_n4,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=[in_app, in_ch, in_dev, in_os, in_h,
                          in_agg1, in_agg2, in_agg3, in_agg4, in_agg5,
                          agg1_float, agg2_float, agg3_float, agg4_float, agg5_float], outputs=outp)
    
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=LR),metrics=['accuracy'])
    
    if i==0:
        model.summary()
    

    
    ##### THE FIT #####
    
    model.fit(df, y_train, batch_size=BATCHSIZE, epochs=EPOCHS, shuffle=True, verbose=2)
    
    
    
    ##### THE TEST DATA #####
    
    del y_train
    del df
    gc.collect()
    
    print('load test...')
    df = pd.read_pickle('../input/training-and-validation-data-pickle/validation.pkl.gz')
    df['click_id'] = df.index
    if i==0:
        y_test = df['is_attributed']
    df.drop(['is_attributed'],axis=1,inplace=True)

    gc.collect()
    
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
             'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    del gp
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    gc.collect()
    
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_day_hh'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
    gc.collect()
    
    gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_hh_os'})
    df = df.merge(gp, on=['ip','os','hour','day'], how='left')
    del gp
    df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
    gc.collect()
    
    gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour','day'], how='left')
    del gp
    df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
    gc.collect()
    
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
    
    if i==0:
        sub = pd.DataFrame()
        sub['click_id'] = df['click_id'].astype('int32')
        sub['is_attributed'] = (0*sub['click_id']).astype('float32')

    
    df.drop(['click_id'],1,inplace=True)
    df = get_keras_data(df)
    gc.collect()
    
    print("predicting from training dataset ", i, "..." )
    sub['is_attributed'] += model.predict(df, batch_size=BATCHSIZE, verbose=2).astype('float32').reshape(-1)
    del df; gc.collect()
    
    print('Total memory in use after cleanup: ', process.memory_info().rss/(2**20), ' MB\n')



sub['is_attributed'] /= NRUNS

print(sub.head())

y_pred = sub['is_attributed'].values
print(  "\n\nFULL VALIDATION SCORE:    ", 
        roc_auc_score( np.array(y_test), y_pred )  )
print(  "\nPUBLIC VALIDATION SCORE:  ", 
        roc_auc_score( np.array(y_test)[:PUBLIC_CUTOFF], y_pred[:PUBLIC_CUTOFF] )  )
print(  "\nPRIVATE VALIDATION SCORE: ",
        roc_auc_score( np.array(y_test)[PUBLIC_CUTOFF:], y_pred[PUBLIC_CUTOFF:] )  )

print("writing....")
sub.to_csv('gpu_val1.csv', index=False, float_format='%.9f')
print("Done")