# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import math
import copy
import random

from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

import tensorflow as tf



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


augment_count = 25
batch_size = 1000
batch_size2 = 5000
optimizer = 'nadam'
num_models = 1
use_specz = False
valid_size = 0.1
max_epochs = 1000

limit = 1000000
sequence_len = 256

classes = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99], dtype='int32')
class_names = ['class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99']
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 1}

# LSST passbands (nm)  u    g    r    i    z    y      
passbands = np.array([357, 477, 621, 754, 871, 1004], dtype='float32')


def append_data(list_x, list_y = None):
    X = {}
    for k in list_x[0].keys():

        list = [x[k] for x in list_x]
        X[k] = np.concatenate(list)

    if list_y is None:
        return X
    else:
        return X, np.concatenate(list_y)
        
        
def get_wtable(df):
    
    all_y = np.array(df['target'], dtype = 'int32')

    y_count = np.unique(all_y, return_counts=True)[1]

    wtable = np.ones(len(classes))

    for i in range(0, y_count.shape[0]):
        wtable[i] = y_count[i] / all_y.shape[0]

    return wtable


def get_keras_data(itemslist):

    keys = itemslist[0].keys()
    X = {
            'id': np.array([i['id'] for i in itemslist], dtype='int32'),
            'meta': np.array([i['meta'] for i in itemslist]),
            'band': pad_sequences([i['band'] for i in itemslist], maxlen=sequence_len, dtype='int32'),
            'hist': pad_sequences([i['hist'] for i in itemslist], maxlen=sequence_len, dtype='float32'),
        }

    Y = to_categorical([i['target'] for i in itemslist], num_classes=len(classes))

    X['hist'][:,:,0] = 0 # remove abs time
#    X['hist'][:,:,1] = 0 # remove flux
#    X['hist'][:,:,2] = 0 # remove flux err
    X['hist'][:,:,3] = 0 # remove detected flag
#    X['hist'][:,:,4] = 0 # remove fwd intervals
#    X['hist'][:,:,5] = 0 # remove bwd intervals
#    X['hist'][:,:,6] = 0 # remove source wavelength
    X['hist'][:,:,7] = 0 # remove received wavelength

    return X, Y



def set_intervals(sample):

    hist = sample['hist']
    band = sample['band']

    hist[:,4] = np.ediff1d(hist[:,0], to_begin = [0])
    hist[:,5] = np.ediff1d(hist[:,0], to_end = [0])
    


def copy_sample(s, augmentate=True):
    c = copy.deepcopy(s)

    if not augmentate:
        return c

    band = []
    hist = []

    drop_rate = 0.3

    # drop some records
    for k in range(s['band'].shape[0]):
        if random.uniform(0, 1) >= drop_rate:
            band.append(s['band'][k])
            hist.append(s['hist'][k])

    c['hist'] = np.array(hist, dtype='float32')
    c['band'] = np.array(band, dtype='int32')

    set_intervals(c)
            
    new_z = random.normalvariate(c['meta'][5], c['meta'][6] / 1.5)
    new_z = max(new_z, 0)
    new_z = min(new_z, 5)

    dt = (1 + c['meta'][5]) / (1 + new_z)
    c['meta'][5] = new_z

    # augmentation for flux
    c['hist'][:,1] = np.random.normal(c['hist'][:,1], c['hist'][:,2] / 1.5)

    # multiply time intervals and wavelength to apply augmentation for red shift
    c['hist'][:,0] *= dt
    c['hist'][:,4] *= dt
    c['hist'][:,5] *= dt
    c['hist'][:,6] *= dt

    return c


def normalize_counts(samples, wtable, augmentate):
    maxpr = np.max(wtable)
    counts = maxpr / wtable

    res = []
    index = 0
    for s in samples:

        index += 1
        print('Normalizing {0}/{1}   '.format(index, len(samples)), end='\r')

        res.append(s)
        count = int(3 * counts[s['target']]) - 1

        for i in range(0, count):
            res.append(copy_sample(s, augmentate))

    print()

    return res




def augmentate(samples, gl_count, exgl_count):

    res = []
    index = 0
    for s in samples:

        index += 1
        
        if index % 1000 == 0:
            print('Augmenting {0}/{1}   '.format(index, len(samples)), end='\r')

        count = gl_count if (s['meta'][8] == 0) else exgl_count

        for i in range(0, count):
            res.append(copy_sample(s))

    print()
    return res


def get_data(data_df, meta_df, extragalactic=None, use_specz=False):

    samples = []
    groups = data_df.groupby('object_id')

    for g in groups:

        id = g[0]

        sample = {}
        sample['id'] = int(id)

        #object_id,ra,decl,gal_l,gal_b,ddf,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target
        #615,349.046051,-61.943836,320.796530,-51.753706,1,0.0000,0.0000,0.0000,nan,0.017,92
        meta = meta_df.loc[meta_df['object_id'] == id]

        if extragalactic == True and float(meta['hostgal_photoz']) == 0:
            continue

        if extragalactic == False and float(meta['hostgal_photoz']) > 0:
            continue


        if 'target' in meta:
            sample['target'] = np.where(classes == int(meta['target']))[0][0]
        else:
            sample['target'] = len(classes) - 1

        sample['meta'] = np.zeros(10, dtype = 'float32')

        sample['meta'][4] = meta['ddf']
        sample['meta'][5] = meta['hostgal_photoz']
        sample['meta'][6] = meta['hostgal_photoz_err']
        sample['meta'][7] = meta['mwebv']
        sample['meta'][8] = float(meta['hostgal_photoz']) > 0

        sample['specz'] = float(meta['hostgal_specz'])

        if use_specz:
            sample['meta'][5] = float(meta['hostgal_specz'])
            sample['meta'][6] = 0.0

        z = float(sample['meta'][5])

        #object_id,mjd,passband,flux,flux_err,detected
        #615,59750.4229,2,-544.810303,3.622952,1


        mjd      = np.array(g[1]['mjd'],      dtype='float32')
        band     = np.array(g[1]['passband'], dtype='int32')
        flux     = np.array(g[1]['flux'],     dtype='float32')
        flux_err = np.array(g[1]['flux_err'], dtype='float32')
        detected = np.array(g[1]['detected'], dtype='float32')

        mjd -= mjd[0]
        mjd /= 100 # Earth time shift in day*100
        mjd /= (z + 1) # Object time shift in day*100


        received_wavelength = passbands[band] # Earth wavelength in nm
        received_freq = 300000 / received_wavelength # Earth frequency in THz
        source_wavelength = received_wavelength / (z + 1) # Object wavelength in nm


        sample['band'] = band + 1

        sample['hist'] = np.zeros((flux.shape[0], 8), dtype='float32')
        sample['hist'][:,0] = mjd
        sample['hist'][:,1] = flux
        sample['hist'][:,2] = flux_err
        sample['hist'][:,3] = detected

        sample['hist'][:,6] = (source_wavelength/1000)
        sample['hist'][:,7] = (received_wavelength/1000)

        set_intervals(sample)


        flux_max = np.max(flux)
        flux_min = np.min(flux)
        flux_pow = math.log2(flux_max - flux_min)
        sample['hist'][:,1] /= math.pow(2, flux_pow)
        sample['hist'][:,2] /= math.pow(2, flux_pow)
        sample['meta'][9] = flux_pow / 10


        samples.append(sample)

        if len(samples) % 1000 == 0:
            print('Converting data {0}'.format(len(samples)), end='\r')

        if len(samples) >= limit:
            break


    print()
    return samples
        



print('Loading train data...')

train_meta = pd.read_csv('../input/training_set_metadata.csv')
train_data = pd.read_csv('../input/training_set.csv')


wtable = get_wtable(train_meta)

def mywloss(y_true,y_pred):
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss
    
    
def multi_weighted_logloss(y_ohe, y_p, wtable):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    nb_pos = wtable

    if nb_pos[-1] == 0:
        nb_pos[-1] = 1

    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss / y_ohe.shape[0]
    


def get_model(X, Y, size=80):

    hist_input = Input(shape=X['hist'][0].shape, name='hist')
    meta_input = Input(shape=X['meta'][0].shape, name='meta')
    band_input = Input(shape=X['band'][0].shape, name='band')

    band_emb = Embedding(8, 8)(band_input)

    hist = concatenate([hist_input, band_emb])
    hist = TimeDistributed(Dense(40, activation='relu'))(hist)

    rnn = Bidirectional(CuDNNGRU(size, return_sequences=True))(hist)
    rnn = SpatialDropout1D(0.5)(rnn)

    gmp = GlobalMaxPool1D()(rnn)
    gmp = Dropout(0.5)(gmp)

    x = concatenate([meta_input, gmp])
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(15, activation='softmax')(x)

    model = Model(inputs=[hist_input, meta_input, band_input], outputs=output)

    return model


def train_model(i, samples_train, samples_valid):

    samples_train += augmentate(samples_train, augment_count, augment_count)
    patience = 1000000 // len(samples_train) + 5

    train_x, train_y = get_keras_data(samples_train)
    del samples_train
    valid_x, valid_y = get_keras_data(samples_valid)
    del samples_valid

    model = get_model(train_x, train_y)

    if i == 1: model.summary()
    model.compile(optimizer=optimizer, loss=mywloss, metrics=['accuracy'])


    print('Training model {0} of {1}, Patience: {2}'.format(i, num_models, patience))
    filename = 'model_{0:03d}.hdf5'.format(i)
    callbacks = [EarlyStopping(patience=patience, verbose=1), ModelCheckpoint(filename, save_best_only=True)]

    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=max_epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)

    model = load_model(filename, custom_objects={'mywloss': mywloss})

    preds = model.predict(valid_x, batch_size=batch_size2)
    loss = multi_weighted_logloss(valid_y, preds, wtable)
    acc = accuracy_score(np.argmax(valid_y, axis=1), np.argmax(preds,axis=1))
    print('MW Loss: {0:.4f}, Accuracy: {1:.4f}'.format(loss, acc))



samples = get_data(train_data, train_meta, use_specz=use_specz)

for i in range(1, num_models+1):

    samples_train, samples_valid = train_test_split(samples, test_size=valid_size, random_state=42*i)

    train_model(i, samples_train, samples_valid)

