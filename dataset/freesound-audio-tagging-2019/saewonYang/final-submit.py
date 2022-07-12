
import numpy as np 
import pandas as pd 
import multiprocessing
from joblib import Parallel, delayed
import os
import sys
import gc

import librosa

from keras.models import load_model
from pathlib import Path
from keras.utils import Sequence
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, concatenate
from keras.models import Model
from tqdm import tqdm
from time import time
import keras as kr


def create_mel64_model():
    mn = MobileNetV2(include_top=False, weights=None)
    mn.layers.pop(0)
    inp = Input(shape=(64, None, 1))
    x = BatchNormalization()(inp)
    x = Conv2D(10, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu')(x)
    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(80, activation='softmax')(x)
    model = Model(inputs=[inp], outputs=x)
    return model

class TestGenerator(kr.utils.Sequence):
    def __init__(self,
                 mel_files,
                 batch_size=64,
                 mel_data=None,
                 req_mel_len=None):

        self.mel_files = mel_files 
        self.batch_size = batch_size
        self.mel_data = mel_data

        self.one_set_size = int(np.ceil(len(self.mel_files) / self.batch_size))

        self.req_mel_len = req_mel_len
        self.on_epoch_end()

    def load_one_mel(self, filename):
        x = self.mel_data[filename].copy()
        x = uni_len(x, self.req_mel_len)
        x = x[..., np.newaxis]
        return x

    def load_mels_for_batch(self, filelist):
        this_batch_data = [self.load_one_mel(x) for x in filelist]
        return np.array(this_batch_data)

    def __len__(self):
        return self.one_set_size

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.mel_files))
        tmp = []
        for one_req_len in [self.req_mel_len]:
            self.req_mel_len = one_req_len
            tmp.append(self.load_mels_for_batch([
                  self.mel_files[i] for i in np.arange(len(self.mel_files))
            ]))
        self.mel_this_epoch = tmp

    def __data_generation(self, batch_num):

        this_set = int(batch_num / self.one_set_size)
        this_index = batch_num % self.one_set_size
        this_indices = self.indexes[this_index*self.batch_size:(this_index+1)*self.batch_size]

        this_batch_mel = self.mel_this_epoch[this_set][this_indices, :, :]

        return this_batch_mel

def mel_0_1(x):
    min_val = -90.5
    max_val = 39.21
    return (x - min_val) / (max_val - min_val)

def uni_len(x, reqlen):
    x_len = x.shape[1]
    if reqlen < x_len:
        max_offset = x_len - reqlen
        offset = np.random.randint(max_offset)
        x = x[:, offset:(reqlen+offset)]
        return x
    elif reqlen == x_len:
        return x
    else:
        total_diff = reqlen - x_len
        offset = np.random.randint(total_diff)
        left_pad = offset
        right_pad = total_diff - offset
        return np.pad(x, (
            (0, 0), (left_pad, right_pad)
        ), 'symmetric')
        
def trim_mel64_resample(dir, filename, n_mels=64):
    x, sr = librosa.load(dir + filename, sampling_rate)
    x = librosa.effects.trim(x)[0]
    x = librosa.resample(x, 44100, 22050)
    melspec = librosa.feature.melspectrogram(x,
                                             sr=22050,
                                             n_fft=1764,
                                             hop_length=220,
                                             n_mels=n_mels)
    logmel = librosa.core.power_to_db(melspec)
    logmel = mel_0_1(logmel)
    return logmel

    
test_metadata = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')

sampling_rate = 44100

num_cores = 2

x_test64 = Parallel(n_jobs=num_cores)(
    delayed(lambda x: trim_mel64_resample('../input/freesound-audio-tagging-2019/test/', x))(x) for x in test_metadata.fname.values)


mel_test_64 = dict()

for i, fname in enumerate(test_metadata.fname.values):
    mel_test_64[fname] = x_test64[i]

del x_test64
gc.collect()


fold_dict_64 = {
              'fold0': Path('../input/fsat-mel-all/64fold0.h5'),
              'fold1': Path('../input/fsat-mel-all/64fold1.h5'),
              'fold2': Path('../input/fsat-mel-all/64fold2.h5'),
              'fold3': Path('../input/fsat-mel-all/64fold3.h5'),
              'fold4': Path('../input/fsat-mel-all/64fold4.h5'),
             }

        
def fold_load_model(path, mel64:bool):
    if mel64:
        model = create_mel64_model()
        model.load_weights(path)
    else:
        model = create_mel128_model()
        model.load_weights(path)
    return model
    
    

time_list = [263, 363, 463, 563, 663, 763]

batch_size = 16
prediction = np.log(np.ones((len(test_metadata), 80)))


for fold, path in fold_dict_64.items():
    model = fold_load_model(path, mel64=True)
    
    for req_mel_len in time_list:
        
        for _ in range(1):    
            
            test_generator = TestGenerator(
                    test_metadata.fname.values,
                    batch_size=batch_size,
                    mel_data=mel_test_64,
                    req_mel_len=req_mel_len)

            this_pred = model.predict_generator(
                    test_generator,
                    steps=len(test_generator),
                    max_queue_size=1,
                    workers=2,
                    use_multiprocessing=False)
            prediction = prediction + np.log(this_pred + 1e-38)
            del test_generator, this_pred
            gc.collect()
    del model

del mel_test_64
del fold_dict_64
gc.collect()

# print(np.min(prediction))
# print(np.max(prediction))

prediction = prediction / 60.
prediction = np.exp(prediction)
# print(np.min(prediction))
# print(np.max(prediction))

test_metadata.iloc[:, 1:] = prediction
test_metadata.to_csv('submission.csv', index=False)
test_metadata.head()
