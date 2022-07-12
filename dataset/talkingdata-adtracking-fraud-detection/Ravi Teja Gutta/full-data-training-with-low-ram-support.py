# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path = '../input/'
dtypes = {
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
cat_cols = ['app','channel', 'device', 'os']
maxvalue_dict = {}
encoder_dict = {}
epochs = 1


def encode_df(): 
    # encode cat_cols
    for col in cat_cols:
        train_col = pd.read_csv(os.path.join(path, 'train.csv'), usecols=[col], dtype=dtypes[col])
        test_col = pd.read_csv(os.path.join(path, 'test.csv'), usecols=[col], dtype=dtypes[col])
        train_col = train_col.append(test_col)
        del test_col
        enc = LabelEncoder()
        enc.fit(train_col)
        encoder_dict[col] = enc
        maxvalue_dict[col] = train_col.values.max() + 1
    train_col = pd.read_csv(os.path.join(path, 'train.csv'), usecols=['click_time'])
    test_col = pd.read_csv(os.path.join(path, 'test.csv'), usecols=['click_time'])
    train_col = train_col.append(test_col)
    del test_col
    
    tfeat =  pd.to_datetime(train_col.click_time).dt.hour.astype('uint8')
    enc = LabelEncoder()
    enc.fit(tfeat)
    encoder_dict['hour'] = enc
    maxvalue_dict['hour'] = tfeat.values.max() + 1
    
    tfeat =  pd.to_datetime(train_col.click_time).dt.day.astype('uint8')
    enc = LabelEncoder()
    enc.fit(tfeat)
    encoder_dict['day'] = enc
    maxvalue_dict['day'] = tfeat.values.max() + 1
    
    tfeat =  pd.to_datetime(train_col.click_time).dt.dayofweek.astype('uint8')
    enc = LabelEncoder()
    enc.fit(tfeat)
    encoder_dict['wday'] = enc
    maxvalue_dict['wday'] = tfeat.values.max() + 1
    
        
        
def get_model():
    # taken from https://www.kaggle.com/alexanderkireev/deep-learning-support-966/code
    emb_n = 50
    dense_n = 1000
    in_app = Input(shape=[1], name = 'app')
    emb_app = Embedding(maxvalue_dict['app'], emb_n)(in_app)
    in_ch = Input(shape=[1], name = 'ch')
    emb_ch = Embedding(maxvalue_dict['channel'], emb_n)(in_ch)
    in_dev = Input(shape=[1], name = 'dev')
    emb_dev = Embedding(maxvalue_dict['device'], emb_n)(in_dev)
    in_os = Input(shape=[1], name = 'os')
    emb_os = Embedding(maxvalue_dict['os'], emb_n)(in_os)
    in_h = Input(shape=[1], name = 'h')
    emb_h = Embedding(maxvalue_dict['hour'], emb_n)(in_h) 
    in_d = Input(shape=[1], name = 'd')
    emb_d = Embedding(maxvalue_dict['day'], emb_n)(in_d) 
    in_wd = Input(shape=[1], name = 'wd')
    emb_wd = Embedding(maxvalue_dict['wday'], emb_n)(in_wd) 
    # removed interactions for simplicity and also neural nets are capable of capturing interactions when trained properly
    fe = concatenate([(emb_app), (emb_ch), (emb_dev),
                      (emb_os), (emb_h), (emb_d), (emb_wd)])
    s_dout = SpatialDropout1D(0.2)(fe)
    x = Flatten()(s_dout)
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_wd], outputs=outp)
    return model


def transform_df(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['wday'] = pd.to_datetime(df.click_time).dt.dayofweek.astype('uint8')
    df.drop('click_time', axis=1, inplace=True)
    for col in cat_cols:
        df[col] = encoder_dict[col].transform(df[col])
    df['hour'] = encoder_dict['hour'].transform(df['hour'])
    df['day'] = encoder_dict['day'].transform(df['day'])
    df['wday'] = encoder_dict['wday'].transform(df['wday'])
    return df


def train():
    optimizer_adam = Adam(lr=0.001)
    print(maxvalue_dict)
    model = get_model()
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['binary_crossentropy'])
    for epoch in range(epochs):
        df_iter = pd.read_csv(os.path.join(path, 'train.csv'), usecols=['app', 'channel', 'device', 'os', 'click_time', 'is_attributed'],
                              dtype=dtypes, chunksize=1000000)
        
        steps = 10
        for step in range(steps):
            chunk = next(df_iter)
            # to run on entire data, change for loop to 'for chunk in df_iter:' and comment out chunk = next(df_iter)
            # transform chunk
            y_train = chunk['is_attributed']
            chunk.drop('is_attributed', axis=1, inplace=True)
            chunk = transform_df(chunk)
            model.fit([chunk['app'].values, chunk['channel'].values, chunk['device'].values,
                       chunk['os'].values, chunk['hour'].values, chunk['day'].values, chunk['wday'].values],
                       y_train, batch_size=50000)        
            
             

if __name__ == '__main__':
    encode_df()
    train()














