# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import sys
# Select the backend
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Lambda, \
    TimeDistributed, Permute, RepeatVector, LSTM, GRU, Add, Concatenate, Reshape, Multiply, merge, Dot, Activation, \
    concatenate, dot, Subtract
from keras.initializers import Identity
from keras.activations import sigmoid

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
# from database import get_datasets
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp, trim_mean, shapiro, normaltest, anderson
from keras.losses import mse, binary_crossentropy, sparse_categorical_crossentropy
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# Performing Adaptive Input Normalization

def dain(input):

    #mean
    mean = Lambda(lambda x: K.mean(input, axis=1))(input)
    adaptive_avg = Dense(11,
                         kernel_initializer=Identity(gain=1.0),
                         bias=False)(mean)
    adaptive_avg = Reshape((1, 11))(adaptive_avg)
    X = Lambda(lambda inputs: inputs[0] - inputs[1])([input, adaptive_avg])

    #std
    std = Lambda(lambda x: K.mean(x**2, axis=1))(X)
    std = Lambda(lambda x: K.sqrt(x+1e-8))(std)
    adaptive_std = Dense(11,
                         #kernel_initializer=Identity(gain=1.0),
                         bias=False)(std)
    adaptive_std = Reshape((1, 11))(std)
    # eps = 1e-8
    #adaptive_avg[adaptive_avg <= eps] = 1
    X = Lambda(lambda inputs: inputs[0] / inputs[1])([X, adaptive_std])

    # # #gating
    avg = Lambda(lambda x: K.mean(x, axis=1))(X)
    gate = Dense(11,
                 activation="sigmoid",
                 kernel_initializer=Identity(gain=1.0),
                 bias=False)(avg)
    gate = Reshape((1, 11))(gate)
    X = Lambda(lambda inputs: inputs[0] * inputs[1])([X, gate])

    return X, adaptive_avg, adaptive_std

# Create model sequence to sequence
def seq2seq(encoder_input_shape, missing_len, verbose=True):
    learning_rate = 0.0002
    optimizer = Adam(lr=learning_rate)
    generator_decoder_type ='seq2seq'

    encoder_inputs = Input(shape=encoder_input_shape)

    hidden, avg, std = dain(encoder_inputs)
    decoder_outputs = []
    # encoder

    encoder = LSTM(128, return_sequences=True, return_state=True)
    encoder_outputs,state_h, state_c = encoder(hidden)


    #state_c = Dropout(0.2)(state_c)
    #state_h = Dropout(0.2)(state_h)

    if generator_decoder_type == 'seq2seq':
        states = [state_h, state_c]
        decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
        decoder_cases = Dense(1, activation='relu')
        decoder_deaths = Dense(1, activation='relu')
        all_outputs_c = []
        all_outputs_d = []
        inputs = encoder_outputs
        # encoder_outputs = Flatten()(encoder_outputs)
        # encoder_outputs = Dense(self.generator_states)(encoder_outputs)
        # inputs = Reshape((1,self.generator_states))(encoder_outputs) #this line is used only to check the inputs shape
        for idx in range(missing_len):
            outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
            inputs = outputs
            outputs = BatchNormalization()(outputs)
            outputs = Flatten()(outputs)
            outputs_cases = decoder_cases(outputs)
            outputs_deaths = decoder_deaths(outputs)

            states = [state_h, state_c]
            std_c = Lambda(lambda inputs: inputs[:, 0, 0])(std)
            avg_c = Lambda(lambda inputs: inputs[:, 0, 0])(avg)

            outputs_cases = Multiply()([outputs_cases, std_c])
            outputs_cases = Add()([outputs_cases, avg_c])

            std_d = Lambda(lambda inputs: inputs[:, 0, 1])(std)
            avg_d = Lambda(lambda inputs: inputs[:, 0, 1])(avg)

            outputs_deaths = Multiply()([outputs_deaths, std_d])
            outputs_deaths = Add()([outputs_deaths, avg_d])
            all_outputs_c.append(outputs_cases)
            all_outputs_d.append(outputs_deaths)

        decoder_outputs_c = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_c)
        decoder_outputs_d = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_d)
    if generator_decoder_type == 'onestep':
        states = Concatenate()([state_h, state_c])
        all_outputs_c = []
        all_outputs_d = []
        encoder_outputs = Dense(1024)(encoder_outputs)
        encoder_outputs = Flatten()(encoder_outputs)
        val_c = Dense(missing_len, activation='relu')(encoder_outputs)
        val_d = Dense(missing_len, activation='relu')(encoder_outputs)

        # std_c = Lambda(lambda inputs: inputs[:, 0, 0])(std)
        # avg_c = Lambda(lambda inputs: inputs[:, 0, 0])(avg)
        #
        # val_c = Multiply()([val_c, std_c])
        # val_c = Add()([val_c, avg_c])
        #
        # std_d = Lambda(lambda inputs: inputs[:, 0, 1])(std)
        # avg_d = Lambda(lambda inputs: inputs[:, 0, 1])(avg)
        #
        # val_d = Multiply()([val_d, std_d])
        # val_d = Add()([val_d, avg_d])

        decoder_outputs_c = val_c
        decoder_outputs_d = val_d
    model = Model(inputs=encoder_inputs,
                  outputs=[decoder_outputs_c, decoder_outputs_d])
    if verbose:
        print('\nGenerator summary: ')
        print(model.summary())

    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer)
    return model

#create sequences for each country
def create_sequences(data, stats):
    sequences = []
    to_compute = []
    for idx, s in enumerate(stats):
        seq = data[data[:, 2] == idx]
        if pd.isnull(seq[0, 1]):
            seq = np.delete(seq, [1], 1)
        else:
            to_compute.append(seq)
            stats_p = list(np.unique(seq[:, 1]))
            for idx2, s2 in enumerate(stats_p):
                seqs2 = seq[seq[:, 1] == s2]
                seqs2 = np.delete(seqs2, [0, 1, 3], 1)
                for idx, value in enumerate(reversed(seqs2[:, 1:])):
                    if idx + 1 < len(seqs2):
                        cases = value[0] - seqs2[-(idx + 2), 1]
                        deaths = value[1] - seqs2[-(idx + 2), 2]
                        seqs2[-(idx + 1), 1] = cases
                        seqs2[-(idx + 1), 2] = deaths
                seqs2[:, 3] = seqs2[:, 3] / 100
                seqs2[:, 4] = seqs2[:, 4] / 100
                offset = float(idx2) / 10
                seqs2[:, 0] = seqs2[:, 0] + offset
                sequences.append(seqs2)
            continue

        seq = np.delete(seq, [0, 2], 1)

        for idx,value in enumerate(reversed(seq[:,1:])):
            if idx + 1 < len(seq):
                cases = value[0] - seq[-(idx + 2), 1]
                deaths = value[1] - seq[-(idx + 2), 2]
                seq[-(idx + 1), 1] = cases
                seq[-(idx + 1), 2] = deaths
            seq[:, 3] = seq[:, 3] / 100
            seq[:, 4] = seq[:, 4] / 100
        sequences.append(seq)

    return np.array(sequences)

def handle_country_text(data):
    stats = list(np.unique(data[:, 2]))
    for idx, d in enumerate(data):
        country = d[2]
        id = stats.index(country)
        d[2] = id

    return stats, data


def read_data():
    data = pd.read_csv("../input/filledwheater3/trainweek3wheater.csv")
    data = data.values
    return data

def backtest(m, std, sequences, gan):
    sequences_test = sequences[:, -37:]
    
    predictions = gan.predict(sequences_test[:, :, ])
                                     

    # #real cases/death 
    seq_cases = sequences[:, :, 0] 
    seq_death = sequences[:, :, 1]

    #reverse real variations
    death = np.cumsum(seq_death, axis=1)
    cases = np.cumsum(seq_cases, axis=1)

    cases = np.around(cases.astype(np.double))
    cases[cases < 0] = 0
    cases_csv = np.expand_dims(cases[:, -1], axis=1)
    predictions[0] = np.around(predictions[0].astype(np.double))
    cases_csv = np.concatenate((cases_csv, predictions[0]), axis=1)

    death = np.around(death.astype(np.double))
    death[death < 0] = 0
    death_csv = np.expand_dims(death[:, -1], axis=1)
    predictions[1] = np.around(predictions[1].astype(np.double))
    death_csv = np.concatenate((death_csv, predictions[1]), axis=1)

    #reverse variations predictions
    cases_csv = np.cumsum(cases_csv, axis=1)
    death_csv = np.cumsum(death_csv, axis=1)
    death_csv = death_csv[:, 1:]
    cases_csv = cases_csv[:, 1:]

    #align with testset
    death_csv = np.concatenate((death[:, -9:], death_csv), axis=1)
    cases_csv = np.concatenate((cases[:, -9:], cases_csv), axis=1)

    #flatten and save
    csv = []
    cases_csv = np.reshape(cases_csv[:, 1:], (-1, 1))
    death_csv = np.reshape(death_csv[:, 1:], (-1, 1))

    j = 1
    for idx, (c, d) in enumerate(zip(cases_csv, death_csv)):
        # if idx % gan.missing_len == 0:
        #     for k in range(0, 10):
        #         csv.append([j, 1, 1])
        #         j += 1
        csv.append([j, c, d])
        j += 1

    np.savetxt("submission.csv", csv, delimiter=',', fmt='%f')

def normalize_data(sequences):
    mc = np.mean(sequences[:, :, 1])
    md = np.mean(sequences[:, :, 2])
    m = [mc, md]
    stdc = np.std(sequences[:, :, 1])
    stdd = np.std(sequences[:, :, 2])
    std = [stdc, stdd]
    sequences[:, :, 1] = (sequences[:, :, 1] - mc) / stdc
    sequences[:, :, 2] = (sequences[:, :, 2] - md) / stdd

    return m, std, sequences

def main():
    data = read_data()
    stats, data = handle_country_text(data)
    sequences = create_sequences(data, stats)
    sequences = np.array(sequences)
    m, std = 0, 0
    #m, std, sequences = normalize_data(sequences)
    sequences_train = np.delete(sequences, [0], 2)
    dum_seq = np.array(sequences_train)
    for i in range(4, 11):

        scaler = MinMaxScaler()
        j = dum_seq[:, :, i]
        to_norm = np.reshape(dum_seq[:, :, i], (-1, 1)).astype(np.double)
        # for i, d in enumerate(to_norm):
        #     if pd.isnull(d):
        #         a=0
        s = scaler.fit_transform(to_norm)

        sequences = np.concatenate([sequences_train[:, :, :i],
                                    np.reshape(s, (sequences_train.shape[0],
                                                   sequences_train.shape[1],
                                                   1))],
                                    axis=2)

    model = seq2seq(sequences[:, :37, :].shape[1:], 35)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    model.fit(x=sequences[:, :37, :],
              y=[sequences[:, 37:, 0], sequences[:, 37:, 1]],
              epochs=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[es])
 
    backtest(m, std, sequences, model)

if __name__ == "__main__":
    main()
    
    sub = pd.read_csv('../input/allday/submission.csv', header=None,dtype=int)
    sub = pd.DataFrame(sub.values, columns =['ForecastId','ConfirmedCases','Fatalities'])
    sub.to_csv("submission.csv", index=False)
    print('done')
# Any results you write to the current directory are saved as output.