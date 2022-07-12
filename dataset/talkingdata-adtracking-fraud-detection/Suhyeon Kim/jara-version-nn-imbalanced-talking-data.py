# good day, my friends
# in this kernel we try to continue development of our DL models
# thanks for people who share their works. i hope together we can create smth interest

# https://www.kaggle.com/baomengjiao/embedding-with-neural-network
# https://www.kaggle.com/gpradk/keras-starter-nn-with-embeddings
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787
# https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
# https://www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553
# https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755/versions#base=2202774&new=2519287


#======================================================================================
# we continue our work started in previos kernel "Deep learning support.."
# + we will try to find a ways which can help us increase specialisation of neural network on our task
# + we will try to work with different architect decisions for neural networks
# if you need a details about what we try to create follow the comments

print ('Good luck')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc

path = '../input/test-train/'
dtypes = {
        'ip' : 'uint32',
'app' :                  'uint16',
'device' :               'uint16',
'os' :                   'uint16',
'channel' :              'uint16',
'is_attributed' :        'uint8',
'hour' :                 'uint8',
'day' :                  'uint8',
'nip_day_hh' :           'uint16',
'nip_hh_os' :            'uint16',
'nip_hh_app' :          'uint16',
'nip_hh_dev' :           'uint32',
'ip_app_count' :         'uint64',
'ip_app_os_count' :      'uint64',
'ip_day_hour_count' :    'uint64',
'ip_device_count' :      'uint64',
'app_channel_count' :    'uint64',
'ip_channel_unique' :    'uint64',
'app_cluster_x' :        'uint64',
'channel_cluster_x' :    'uint64',
'device_cluster' :       'uint64',
'os_cluster' :           'uint64',
'next_click' :           'uint64'
        }
print('load train....')
# we save only day 9
train_df = pd.read_csv(path+"train_add_cols.csv", usecols=['ip','app','device','os','channel','is_attributed',
'hour','day','nip_day_hh','nip_hh_os' ,'nip_hh_app'
,'nip_hh_dev','ip_app_count','ip_app_os_count' ,'ip_day_hour_count','ip_device_count'
,'app_channel_count','ip_channel_unique','app_cluster_x','channel_cluster_x','device_cluster','os_cluster' ,'next_click'])
print ('is_attributed devided...')
y_train = train_df['is_attributed'].values 
train_df.drop(['is_attributed'],1,inplace=True) 


print ('neural network....')
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

max_ip = np.max([train_df['ip'].max(), test_df['ip'].max()])+1
max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_d = np.max([train_df['day'].max(), test_df['day'].max()])+1
max_iho = np.max([train_df['nip_hh_os'].max(), test_df['nip_hh_os'].max()])+1
max_iha = np.max([train_df['nip_hh_app'].max(), test_df['nip_hh_app'].max()])+1
max_ihd = np.max([train_df['nip_hh_dev'].max(), test_df['nip_hh_dev'].max()])+1
max_ia = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_iao = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1
max_idh = np.max([train_df['ip_day_hour_count'].max(), test_df['ip_day_hour_count'].max()])+1
max_id = np.max([train_df['ip_device_count'].max(), test_df['ip_device_count'].max()])+1
max_ac = np.max([train_df['app_channel_count'].max(), test_df['app_channel_count'].max()])+1
max_ic_u = np.max([train_df['ip_channel_unique'].max(), test_df['ip_channel_unique'].max()])+1
max_app_clu = np.max([train_df['app_cluster_x'].max(), test_df['app_cluster_x'].max()])+1
max_channel_clu = np.max([train_df['channel_cluster_x'].max(), test_df['channel_cluster_x'].max()])+1
max_device_clu = np.max([train_df['device_cluster'].max(), test_df['device_cluster'].max()])+1
max_os_clu = np.max([train_df['os_cluster'].max(), test_df['os_cluster'].max()])+1
max_nextclick = np.max([train_df['next_click'].max(), test_df['next_click'].max()])+1

def get_keras_data(dataset):
    X = {
        'ip' :                  np.array(dataset.ip),
        'app' :                  np.array(dataset.app),
        'device' :               np.array(dataset.device),
        'os' :                   np.array(dataset.os),
        'channel' :              np.array(dataset.channel),
        'hour' :                 np.array(dataset.hour),
        'day' :                  np.array(dataset.day),
        'nip_hh_os' :           np.array(dataset.nip_hh_os),
        'nip_hh_app' :          np.array(dataset.nip_hh_app),
        'nip_hh_dev' :          np.array(dataset.nip_hh_dev),
        'ip_app_count' :         np.array(dataset.ip_app_count),
        'ip_app_os_count' :     np.array(dataset.ip_app_os_count),
        'ip_day_hour_count' :   np.array(dataset.ip_day_hour_count),
        'ip_device_count' :     np.array(dataset.ip_device_count),
        'app_channel_count' :    np.array(dataset.app_channel_count),
        'ip_channel_unique' :    np.array(dataset.ip_channel_unique),
        'app_cluster_x' :        np.array(dataset.app_cluster_x),
        'channel_cluster_x' :    np.array(dataset.channel_cluster_x),
        'device_cluster' :       np.array(dataset.device_cluster),
        'os_cluster' :           np.array(dataset.os_cluster),
        'next_click' :           np.array(dataset.next_click)
    }
    return X
train_df = get_keras_data(train_df)

emb_n = 50
dense_n = 1000
in_ip = Input(shape=[1], name = 'ip')
emb_ip = Embedding(max_ip, emb_n)(in_ip)
in_app = Input(shape=[1], name = 'app')
emb_ap = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
emb_h = Embedding(max_h, emb_n)(in_h) 
in_d = Input(shape=[1], name = 'd')
emb_d = Embedding(max_d, emb_n)(in_d)
in_iho = Input(shape=[1], name = 'iho')
emb_iho = Embedding(max_iho, emb_n)(in_iho)
in_iha = Input(shape=[1], name = 'iha')
emb_iha = Embedding(max_iha, emb_n)(in_iha)
in_ihd = Input(shape=[1], name = 'ihd')
emb_ihd = Embedding(max_ihd, emb_n)(in_ihd)
in_ia = Input(shape=[1], name = 'ia')
emb_ia = Embedding(max_ia, emb_n)(in_ia)
in_iao = Input(shape=[1], name = 'iao')
emb_iao = Embedding(max_iao, emb_n)(in_iao)
in_idh = Input(shape=[1], name = 'idh')
emb_idh = Embedding(max_idh, emb_n)(in_idh)
in_id = Input(shape=[1], name = 'id')
emb_id = Embedding(max_id, emb_n)(in_id)
in_ac = Input(shape=[1], name = 'ac')
emb_ac = Embedding(max_ac, emb_n)(in_ac)
in_ic_u = Input(shape=[1], name = 'ic_u')
emb_ic_u = Embedding(max_ic_u, emb_n)(in_ic_u)
in_app_clu = Input(shape=[1], name = 'app_clu')
emb_app_clu = Embedding(max_app_clu, emb_n)(in_app_clu)
in_channel_clu = Input(shape=[1], name = 'channel_clu')
emb_channel_clu = Embedding(max_channel_clu, emb_n)(in_channel_clu)
in_device_clu = Input(shape=[1], name = 'device_clu')
emb_device_clu = Embedding(max_device_clu, emb_n)(in_device_clu)
in_os_clu = Input(shape=[1], name = 'os_clu')
emb_os_clu = Embedding(max_os_clu, emb_n)(in_os_clu)
in_nextclick = Input(shape=[1], name = 'nextclick')
emb_nextclick = Embedding(max_nextclick, emb_n)(in_nextclick)
fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                 (emb_d), (emb_idh), (emb_iho), (emb_iha), (emb_ihd), (emb_ia), (emb_iao), (emb_id), (emb_ac),
                 (emb_ic_u), (emb_app_clu), (emb_channel_clu), (emb_device_clu), (emb_os_clu), (emb_nextclick)])
s_dout = SpatialDropout1D(0.2)(fe)
fl1 = Flatten()(s_dout)
conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
fl2 = Flatten()(conv)
concat = concatenate([(fl1), (fl2)])
x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_iho,in_iha,in_ihd,
in_ia,in_iao,in_idh,in_id,in_ac,in_ic_u,in_app_clu,in_channel_clu,in_device_clu,in_os_clu,in_nextclick], outputs=outp)

batch_size = 50000
epochs = 2
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(list(train_df)[0]) / batch_size) * epochs
lr_init, lr_fin = 0.002, 0.0002
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.002, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()

class_weight = {0:.01,1:.99} # magic
model.fit(train_df, y_train, batch_size=batch_size, epochs=2, class_weight=class_weight, shuffle=True, verbose=2)
del train_df, y_train; gc.collect()
model.save_weights('imbalanced_data.h5')

print('load test....')
test_df = pd.read_csv(path+"test_add_cols.csv",usecols=['ip','app','device','os','channel',
'hour','day','nip_day_hh','nip_hh_os' ,'nip_hh_app'
,'nip_hh_dev','ip_app_count','ip_app_os_count' ,'ip_day_hour_count','ip_device_count'
,'app_channel_count','ip_channel_unique','app_cluster_x','channel_cluster_x','device_cluster','os_cluster' ,'next_click'])

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['is_attributed'],1,inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df; gc.collect()
print("writing....")
sub.to_csv('submission_0501.csv',index=False)


# write now i don't use validation set
# since the data is imbalanced i can't understand how we can separate data the right way