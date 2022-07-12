
"""
# referebce 
[1] pipeline: https://www.kaggle.com/chewzy/gru-w-attention-baseline-model-curated
"""

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

import librosa
import librosa.display

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Embedding, Input, Dense, CuDNNGRU, CuDNNLSTM, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, concatenate 
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler


class Attention(Layer):
    def __init__(
            self, step_dim,
            W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):

        self.supports_masking = True

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = None
        super(Attention, self).__init__(**kwargs)

        self.param_W = {
            'initializer': initializers.get('glorot_uniform'),
            'name': '{}_W'.format(self.name),
            'regularizer': regularizers.get(W_regularizer),
            'constraint': constraints.get(W_constraint)
        }
        self.W = None

        self.param_b = {
            'initializer': 'zero',
            'name': '{}_b'.format(self.name),
            'regularizer': regularizers.get(b_regularizer),
            'constraint': constraints.get(b_constraint)
        }
        self.b = None

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]
        self.W = self.add_weight((input_shape[-1],), **self.param_W)

        if self.bias:
            self.b = self.add_weight((input_shape[1],), **self.param_b)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        step_dim = self.step_dim
        features_dim = self.features_dim

        eij = K.reshape(
            K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
            (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def get_model(input_shape=(636, 128), output_shape=80):
    
    sequence_input = Input(shape=input_shape, dtype='float32')
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(sequence_input)
    att = Attention(input_shape[0])(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x) 
    x = concatenate([att, avg_pool, max_pool])

    hidden = Dense(2048, activation='relu')(x)
    output = Dense(output_shape, activation='sigmoid')(hidden)
    model = Model(sequence_input, output)
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(0.005), 
        metrics=['categorical_accuracy'])
    model.summary()
    return model


def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf['sampling_rate'])
    # trim silence
    y_len = len(y)
    conf_samples = conf['samples']

    if 0 < y_len:  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)

    # make it unified length to conf.samples
    if y_len > conf_samples:  # long enough
        if trim_long_data:
            y = y[0:0 + conf_samples]

    else:  # pad blank
        padding = (conf_samples - y_len) // 2   # add padding at both ends
        y = np.pad(y, (padding, conf_samples - y_len - padding), 'constant')
        
    return y


def normalize(img, eps=0.001):
    """
    Normalizes an array
    (subtract mean and divide by standard deviation)
    """
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
        
    return img


def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(
        audio, 
        sr=conf['sampling_rate'],
        n_mels=conf['n_mels'],
        hop_length=conf['hop_length'],
        n_fft=conf['n_fft'],
        fmin=conf['fmin'],
        fmax=conf['fmax'])
    spectrogram = librosa.power_to_db(spectrogram)
    # print(spectrogram.shape)
    spectrogram = normalize(spectrogram)
    return spectrogram.astype(np.float32).transpose()


def read_as_melspectrogram(pathname, conf, trim_long_data):
    x = read_audio(conf, pathname, trim_long_data)
    return audio_to_melspectrogram(conf, x)


def transform(queue, dir_path, conf=None, shape=(636, 128), trim_long_data=True):
    func = partial(read_as_melspectrogram, conf=conf, trim_long_data=trim_long_data)

    data = np.zeros((len(queue),) + shape)
    for i, r in enumerate(map(func, [os.sep.join([dir_path, q]) for q in queue])):
        data[i, :r.shape[0], :r.shape[1]] = r

    return data


def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def labelize(series):
    return series.apply(
        lambda x: pd.Series(index=x.split(',')).fillna(1)).fillna(0).astype(np.int8)


def split_index(series, min_sample=5):
    train_split_index = series.apply(hash)
    tmp = train_split_index.value_counts()
    minor_split_index = tmp.loc[tmp < min_sample]
    rep = {f: -999 for f in minor_split_index.index.tolist()}
    train_split_index.replace(rep, inplace=True)
    return train_split_index


def get_configs():
    conf = {
        'sampling_rate': 44100,
        'duration': 5,
        'hop_length': 347, # to make time steps 128
        'fmin': 20,
        'fmax': 22000,
        'n_mels': 128
    }
    conf['n_fft'] = conf['n_mels'] * 20
    conf['samples'] = conf['sampling_rate'] * conf['duration']
    return conf


def run_model(train_x, train_y, label_columns, df_split_index, model=None):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, valid_index in sss.split(df_split_index, df_split_index):
        x_train, y_train = train_x[train_index], train_y[train_index]
        x_valid, y_valid = train_x[valid_index], train_y[valid_index]
        print(x_train.shape, x_valid.shape)
        break

    if model is None:
        model = get_model(
            input_shape=(train_x.shape[-2], train_x.shape[-1]),
            output_shape=len(label_columns))

    file_weights = 'model.hdf5'
    check_point = ModelCheckpoint(file_weights, save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=10)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1, cooldown=0, min_lr=0.0001)

    model.fit(
        x_train,
        y_train,
        batch_size=512,
        epochs=100,
        validation_data=(x_valid, y_valid),
        verbose=0,
        callbacks=[early_stopping, check_point, reduce_lr])

    model.load_weights(file_weights)
    y_train_pred = model.predict(x_train)
    y_valid_pred = model.predict(x_valid)

    train_lwlrap = calculate_overall_lwlrap_sklearn(y_train, y_train_pred)
    valid_lwlrap = calculate_overall_lwlrap_sklearn(y_valid, y_valid_pred)
    print(f'LWLRAP: training {train_lwlrap:.4f}, valid {valid_lwlrap:.4f}')

    return model


def load_train(data_file_path, data_dir_path, conf):

    df = pd.read_csv(data_file_path).set_index('fname')

    df_split_index = split_index(df['labels'])

    train_y = labelize(df['labels'])
    label_columns = sorted(train_y.columns.tolist())
    train_y = train_y[label_columns].values

    train_x = transform(df.index.tolist(), data_dir_path, conf)
    print(train_x.shape, train_y.shape)
    return train_x, train_y, label_columns, df_split_index


def main():

    train_dict = {
        'curated': {
            'data_dir': '../input/train_curated/',
            'data_list': '../input/train_curated.csv',
            'data_npy_x': '../input/train_curated.npy',
        },

        'nosiy':{
            'data_dir': '../input/train_noisy/',
            'data_list': '../input/train_noisy.csv',
            'data_npy_x': '../input/train_noisy.npy',
        },
    }

    conf = get_configs()
    train = train_dict['curated']
    train_x, train_y, label_columns, df_split_index = load_train(
        train['data_list'], train['data_dir'], conf=conf)

    model = run_model(train_x, train_y, label_columns, df_split_index)

    test_path = '../input/test/'
    test = pd.read_csv('../input/sample_submission.csv').set_index('fname')
    test_x = transform(test.index.tolist(), test_path, conf)
    print(test_x.shape)

    predictions = model.predict(test_x)
    test[label_columns] = predictions
    test.to_csv('submission.csv', index=True)


if __name__ == '__main__':
    main()
    