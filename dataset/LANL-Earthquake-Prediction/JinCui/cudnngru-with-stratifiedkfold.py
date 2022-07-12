'''
Reference:

https://www.kaggle.com/mayer79/rnn-starter-for-huge-time-series

'''

import numpy as np 
import pandas as pd
import gc
import os
print(os.listdir("../input"))
from tqdm import tqdm
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, CuDNNLSTM, Dropout, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
os.environ['PYTHONHASHSEED'] = str(42)
seed(639)
from tensorflow import set_random_seed
set_random_seed(5944)



Q = list(range(0,110,10))

def extract_features(z):
     return np.c_[z.mean(axis=1), 
                  np.transpose(np.percentile(np.abs(z), q=Q, axis=1)),
                  z.std(axis=1)]


def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1).astype(np.float32) - 5 ) / 3

    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 10:]),
                 extract_features(temp[:, -step_length // 100:])]
                 

def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # rows are the last index, I made sure it's dividable by 150000 so that the data is always chosen from a 150000 chunk
        base = np.random.randint( (min_index+n_steps*step_length)//150000, max_index//150000, size=batch_size)
        rows = base*150000
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row - 1, 1]
        yield samples, targets
        
batch_size = 32

histories = []
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

'''
# This is what I did to generate the train & valid data:
## Done on local machine because of lack of memory on kernels!

float_data = pd.read_csv('../input/LANL-Earthquake/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

## The basic idea of sampling_idx is that I want to make sure each fold have a smiliar composition of low and high time_to_failure values

sampling_idx = float_data[:,1].astype('int')
sampling_idx[sampling_idx==16] = 15

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, sampling_idx)):
    print('Fold_{}'.format(i))
    range_trn_idx = [*map(lambda x: list(range(x*150000, (x+1)*150000)), trn_idx)]
    range_trn_idx = sum(range_trn_idx, [])
    range_val_idx = [*map(lambda x: list(range(x*150000, (x+1)*150000)), val_idx)]
    range_val_idx = sum(range_val_idx, [])
    trn_data = float_data[range_trn_idx, :]
    val_data = float_data[range_val_idx, :]
    print(trn_data.shape)
    print(val_data.shape)
    np.save('stratified_4_fold_trn_{}'.format(i), trn_data)
    np.save('stratified_4_fold_val_{}'.format(i), val_data)
'''



for n_fold in range(4):
    train_data = np.load("../input/stratified-150000-segment/stratified/stratified_4_fold_trn_{}.npy".format(n_fold))
    valid_data = np.load("../input/stratified-150000-segment/stratified/stratified_4_fold_val_{}.npy".format(n_fold))
    n_features = create_X(valid_data[0:150000]).shape[1]
    
    train_gen = generator(train_data, batch_size=batch_size)
    valid_gen = generator(valid_data, batch_size=batch_size)
    del train_data, valid_data
    gc.collect()
    print('Fold_{} starts with {} features'.format(n_fold, n_features))
    
    ckpt = ModelCheckpoint("RNN_model_{}.hdf5".format(n_fold), save_best_only=True, period=3)
    es = EarlyStopping(monitor='val_loss',patience=10)
    model = Sequential()
    
    model.add(CuDNNGRU(128, input_shape=(None, n_features), return_sequences=True))
    model.add(Dropout(0.4))
    #model.add(CuDNNGRU(64, return_sequences=True))
    #model.add(Dropout(0.3))
    model.add(CuDNNGRU(64))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    OPT = SGD(lr=0.0005, momentum=0.0, decay=1e-6, nesterov=False)
    #OPT = adam(lr=0.0005, decay=1e-6)
    #OPT = RMSprop(lr=0.0005, decay=1e-6)
    model.compile(optimizer=OPT, loss="mae", metrics=['mae'])
    
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=1000,
                                  epochs=32,
                                  verbose=2,
                                  callbacks=[ckpt, es],
                                  validation_data=valid_gen,
                                  validation_steps=250
                                  )
    histories.append(history)
    for i, seg_id in enumerate(tqdm(submission.index)):
        seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')
        x = seg['acoustic_data'].values
        submission.time_to_failure[i] += model.predict(np.expand_dims(create_X(x), 0))/4

    del history, train_gen, valid_gen
    gc.collect()


def perf_plot(history, what='loss', label_idx=0):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1
    
    plt.plot(epochs, x, 'bo', label = "Training " + what)
    plt.plot(epochs, val_x, 'b', label = "Validation " + what)
    plt.title("Fold {} Training and validation ".format(label_idx) + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('RNN_result_{}.png'.format(label_idx), dpi=300)
    plt.show()
    return None


submission.to_csv('submission.csv')

for i in range(len(histories)):
    plt.subplots()
    perf_plot(histories[i], label_idx=i)

