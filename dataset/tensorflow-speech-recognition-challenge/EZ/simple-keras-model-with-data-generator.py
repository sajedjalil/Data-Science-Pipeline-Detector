
# coding: utf-8


from pathlib import Path
import time

from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])
    
    return df


def prepare_data(df):
    '''Transform data into something more useful.'''
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    words = df.word.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'
    
    return df


def get_specgrams(paths, nsamples=16000):
    '''
    Given list of paths, return specgrams.
    '''
    
    # read the wav files
    wavs = [wavfile.read(x)[1] for x in paths]

    # zero pad the shorter samples and cut off the long ones.
    data = [] 
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
        data.append(d)

    # get the specgram
    specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    specgram = [s.reshape(129, 124, -1) for s in specgram]
    
    return specgram


def batch_generator(X, y, batch_size=16):
    '''
    Return a random image from X, y
    '''
    
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]
        
        specgram = get_specgrams(im)


        yield np.concatenate([specgram]), label


def get_model(shape):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)

    model = BatchNormalization()(inputlayer)
    model = Conv2D(16, (3, 3), activation='elu')(model)
    model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Flatten()(model)
    model = Dense(32, activation='elu')(model)
    model = Dropout(0.25)(model)
    
    # 11 because background noise has been taken out
    model = Dense(11, activation='sigmoid')(model)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model


train = prepare_data(get_data('../input/train/audio/'))
shape = (129, 124, 1)
model = get_model(shape)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# create training and test data.

labelbinarizer = LabelBinarizer()
X = train.path
y = labelbinarizer.fit_transform(train.word)
X, Xt, y, yt = train_test_split(X, y, test_size=0.3, stratify=y)

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32)



train_gen = batch_generator(X.values, y, batch_size=32)
valid_gen = batch_generator(Xt.values, yt, batch_size=32)

model.fit_generator(
    generator=train_gen,
    epochs=1,
    steps_per_epoch=X.shape[0] // 32,
    validation_data=valid_gen,
    validation_steps=Xt.shape[0] // 32,
    callbacks=[tensorboard])


# Create a submission

test = prepare_data(get_data('../input/test/'))

predictions = []
paths = test.path.tolist()

for path in paths:
    specgram = get_specgrams([path])
    pred = model.predict(np.array(specgram))
    predictions.extend(pred)


labels = [labelbinarizer.inverse_transform(p.reshape(1, -1), threshold=0.5)[0] for p in predictions]
test['labels'] = labels
test.path = test.path.apply(lambda x: str(x).split('/')[-1])
submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
submission.to_csv('submission.csv', index=False)