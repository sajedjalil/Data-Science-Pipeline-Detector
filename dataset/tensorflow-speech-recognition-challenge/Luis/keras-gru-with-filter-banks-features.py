
import os
import numpy as np
np.random.seed(1969)
import tensorflow as tf
tf.set_random_seed(1969)


from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.layers import GRU, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, Conv3D, ConvLSTM2D
from keras.callbacks import TensorBoard
from keras.models import Sequential
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')


def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames



def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=1000):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))



labels, fnames = list_wavs_fname(train_data_path)
new_sample_rate=16000
y_train = []
x_train = np.zeros((64727,99,26),np.float32)
G = []
ix = 0
for label, fname in tqdm(zip(labels, fnames)):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else:
        n_samples = [samples]
    for samples in n_samples:
        filter_banks = logfbank(samples)
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        x_train[ix,:,:] = filter_banks
    y_train.append(label)
    group = fname.split('_')[0]
    G.append(group)
    ix += 1

y_train = label_transform(y_train)
label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
G = np.array(G)

del labels, fnames
gc.collect()



model = Sequential()
model.add(GRU(256,input_shape=(99,26)))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['categorical_accuracy'])
model.summary()
weights = model.get_weights()

NUM_FOLDS = 4
EPOCHS = 30
BATCH_SIZE = 512
BAGS = 16

kf = GroupKFold(n_splits=NUM_FOLDS)

shape = None

for bag in range(BAGS):
    fold = 0

    val_loss = np.ones((EPOCHS,NUM_FOLDS),np.float32)

    for train, val in kf.split(x_train,y_train,G):
        model.set_weights(weights)
        model.reset_states()
        tensorboard = TensorBoard(log_dir='logs/gru_fold_{}_bag_{}'.format(fold,bag))
        history = model.fit(x_train[train], y_train[train], batch_size=BATCH_SIZE, validation_data=(x_train[val], y_train[val]), epochs=EPOCHS, shuffle=True, verbose=1, callbacks=[tensorboard])
        val_loss[:,fold] = history.history['val_loss']
        fold += 1

    val_mean = np.mean(val_loss,axis=1)
    best_loss = np.min(val_mean)
    best_epoch = np.argmin(val_mean)
    print('Best epoch: {} Best loss: {}'.format(best_epoch,best_loss))
    model.set_weights(weights)
    model.reset_states()
    tensorboard = TensorBoard(log_dir='logs/gru_bag_{}'.format(bag))
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=best_epoch, shuffle=True, verbose=1, callbacks=[tensorboard])

    model.save('models/gru_{}_{}.h5'.format(bag+1,best_loss))

    def test_data_generator(batch=32):
        fpaths = glob(os.path.join(test_data_path, '*wav'))
        i = 0
        for path in fpaths:
            if i == 0:
                imgs = []
                fnames = []
            i += 1
            rate, samples = wavfile.read(path)
            samples = pad_audio(samples)
            filter_banks = logfbank(samples)
            filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
            imgs.append(filter_banks)
            fnames.append(path.split('/')[-1])
            if i == batch:
                i = 0
                imgs = np.array(imgs)
                yield fnames, imgs
        if i < batch:
            imgs = np.array(imgs)

            yield fnames, imgs
        raise StopIteration()



    gc.collect()

    index = []
    results = []
    probs = []
    for fnames, imgs in tqdm(test_data_generator(batch=32)):
        predicts = model.predict(imgs)
        probs.extend(predicts)
        predicts = np.argmax(predicts, axis=1)
        predicts = [label_index[p] for p in predicts]
        index.extend(fnames)
        results.extend(predicts)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'subs/gru_sub_{}_{}.csv'.format(bag+1,best_loss)), index=False)
    probs = np.array(probs)
    np.save('probs/gru_probs_{}.npy'.format(bag+1),probs)

