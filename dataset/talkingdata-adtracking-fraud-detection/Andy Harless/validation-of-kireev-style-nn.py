# This is validation code. The original submission run is described below.


# This kernel is based on Alexander Kireev's deep learning model:
#   https://www.kaggle.com/alexanderkireev/deep-learning-support-imbalance-architect-9671
# (See notes and references below.)
# I (Andy Harless) have made the following changes:
#   1. Add 2 more (narrower) layers on top
#   2. Eliminate "day" and "wday" variables (no variation in this sample)
#   3. Change target weight from 99 to 70
#   4. Add batch normalization
#   5. Only one epoch
#   6. Eliminate weight decay
#   7. Increase batch size
#   8. Increase dropout

# version 4:  adding ipcount
# version 5:  use gaussian dropout, and increase dimension of ipcount embedding
# version 6:  add raw values (min-max-scaled) for other counts,
#             reduce dimensions for the corresponding embeddings,
#             and revise dimensions for other embeddings


# good day, my friends
# in this kernel we try to continue development of our DL models
# =================================================================================================
# we continue our work
# this kernel is attempt to configure neural network for work with imbalanced data (see ~150th row)
# =================================================================================================
# thanks for people who share his works. i hope together we can create smth interest

# https://www.kaggle.com/baomengjiao/embedding-with-neural-network
# https://www.kaggle.com/gpradk/keras-starter-nn-with-embeddings
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787
# https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
# https://www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553
# https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755/versions#base=2202774&new=2519287

NDATA = 50000000
PUBLIC_CUTOFF = 4032690

print ('Good luck')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc
from sklearn.metrics import roc_auc_score
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GaussianDropout
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam


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
print('load train....')
train_df = pd.read_pickle('../input/training-and-validation-data-pickle/training.pkl.gz')[-NDATA:]
print('load test....')
test_df = pd.read_pickle('../input/training-and-validation-data-pickle/validation.pkl.gz')
test_df['click_id'] = test_df.index
y_test = test_df['is_attributed']
test_df.drop(['is_attributed'],axis=1,inplace=True)
len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df; gc.collect()

print('hour, day, wday....')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
print('grouping by ip alone....')
gp = train_df[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ipcount'})
train_df = train_df.merge(gp, on=['ip'], how='left')
del gp; gc.collect()
print('grouping by ip-day-hour combination....')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp; gc.collect()
print('group by ip-app combination....')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp; gc.collect()
print('group by ip-app-os combination....')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp; gc.collect()
print("vars and data type....")
train_df['ipcount'] = train_df['ipcount'].astype('uint32')
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
print("label encoding....")
from sklearn.preprocessing import LabelEncoder
train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)

max_app = train_df['app'].max()+1
max_ch = train_df['channel'].max()+1
max_dev = train_df['device'].max()+1
max_os = train_df['os'].max()+1
max_h = train_df['hour'].max()+1
max_ipcount = train_df['ipcount'].max()+1
max_qty = train_df['qty'].max()+1
max_c1 = train_df['ip_app_count'].max()+1
max_c2 = train_df['ip_app_os_count'].max()+1

train_df['qty_float'] = train_df['qty'].astype('float32') / np.float32(max_qty)
train_df['ipcount_float'] = train_df['ipcount'].astype('float32') / np.float32(max_ipcount)
train_df['c1_float'] = train_df['ip_app_count'].astype('float32') / np.float32(max_c1)
train_df['c2_float'] = train_df['ip_app_os_count'].astype('float32') / np.float32(max_c2)
print( train_df.info() )

print ('final part of preparation....')
test_df = train_df[len_train:]
train_df = train_df[:len_train]
y_train = train_df['is_attributed'].values
train_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)

print ('neural network....')

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'qty': np.array(dataset.qty),
        'ipcount': np.array(dataset.ipcount),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count),
        'ipcount_float': np.array(dataset.ipcount_float),
        'qty_float': np.array(dataset.qty),
        'c1_float': np.array(dataset.c1_float),
        'c2_float': np.array(dataset.c2_float)
    }
    return X
train_df = get_keras_data(train_df)

emb_n = 50
dense_n1 = 1000
dense_n2 = 1000
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
in_qty = Input(shape=[1], name = 'qty')
emb_qty = Embedding(max_qty, emb_n-10)(in_qty) 
in_ipcount = Input(shape=[1], name = 'ipcount')
emb_ipcount = Embedding(max_ipcount, 2*emb_n+10)(in_ipcount) 
in_c1 = Input(shape=[1], name = 'c1')
emb_c1 = Embedding(max_c1, emb_n-10)(in_c1) 
in_c2 = Input(shape=[1], name = 'c2')
emb_c2 = Embedding(max_c2, emb_n-10)(in_c2)

qty_float = Input(shape=[1], dtype='float32', name='qty_float')
ipcount_float = Input(shape=[1], dtype='float32', name='ipcount_float')
c1_float = Input(shape=[1], dtype='float32', name='c1_float')
c2_float = Input(shape=[1], dtype='float32', name='c2_float')

fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                  (emb_ipcount), (emb_qty), (emb_c1), (emb_c2)])
s_dout = SpatialDropout1D(0.2)(fe)
x = Flatten()(s_dout)
x = concatenate([x, qty_float, ipcount_float, c1_float, c2_float])
x = (BatchNormalization())(x)
x = GaussianDropout(0.2)(Dense(dense_n1,activation='relu')(x))
x = (BatchNormalization())(x)
x = GaussianDropout(0.3)(Dense(dense_n2,activation='relu')(x))
x = (BatchNormalization())(x)
x = GaussianDropout(0.25)(Dense(dense_n3,activation='relu')(x))
x = (BatchNormalization())(x)
x = GaussianDropout(0.2)(Dense(dense_n4,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app, in_ch, in_dev, in_os, in_h, in_ipcount,
                      qty_float, ipcount_float, c1_float, c2_float,
                      in_qty, in_c1, in_c2], outputs=outp)

batch_size = 65536
epochs = 1
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(list(train_df)[0]) / batch_size) * epochs
lr_init, lr_fin = 0.0013, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=lr_init, decay=lr_decay)
optimizer_adam_nodecay = Adam(lr=lr_init)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam_nodecay,metrics=['accuracy'])

model.summary()

class_weight = {0:.01,1:.70} # magic
model.fit(train_df, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight, shuffle=True, verbose=2)
del train_df, y_train; gc.collect()
model.save_weights('imbalanced_data.h5')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
test_preds = model.predict(test_df, batch_size=batch_size, verbose=2)
y_pred = test_preds.reshape(-1)

print(  "\n\nFULL VALIDATION SCORE:    ", 
        roc_auc_score( np.array(y_test), y_pred )  )
print(  "\nPUBLIC VALIDATION SCORE:  ", 
        roc_auc_score( np.array(y_test)[:PUBLIC_CUTOFF], y_pred[:PUBLIC_CUTOFF] )  )
print(  "\nPRIVATE VALIDATION SCORE: ",
        roc_auc_score( np.array(y_test)[PUBLIC_CUTOFF:], y_pred[PUBLIC_CUTOFF:] )  )

sub['is_attributed'] = test_preds
del test_df; gc.collect()
print(sub.head())
print("writing....")
sub.to_csv('dl_val1.csv', index=False, float_format='%.9f')

