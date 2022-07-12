"""
This is a test script I wrote to validate the idea of using Keras LSTM NNs.

The code is mostly adapted from the LSTM examples available from the Keras github. 
It is mean to allow the true data to be subbed in easily.

I was able to get good results at about 1000 epochs.  This will not work without modificatio
due to issues others have detailed such as the length of sequences, and int sizes.

Kaggle lost a bunch of stuff I wrote but I'll rewrite later.

"""
import pandas as pd
import sklearn.cross_validation as skcv
import random

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import l2


def parse_fake(st, ln):
    seq = list(range(st, st + ln))
    last = seq.pop()
    return 0, seq, last


fake_len = 20
samples = 1e3
sample_max = 1e3
train_data = [parse_fake(random.randint(0, sample_max), fake_len) for i in range(1, int(samples))]

df = pd.DataFrame(train_data, columns=['sid', 'sequence', 'nint'])
maxlen = int(round(df.sequence.apply(len).max()))

X_train, X_test, y_train, y_test = skcv.train_test_split(df['sequence'].values, df['nint'].values, test_size=.1)

X_train = pad_sequences(X_train, dtype='float', maxlen=maxlen)
X_test = pad_sequences(X_test, dtype='float', maxlen=maxlen)

X_train_rshp = X_train.reshape(X_train.shape + (1,))
X_test_rshp = X_test.reshape(X_test.shape + (1,))

print('X_train Reshaped shape:', X_train_rshp.shape)
print('X_test shape:', X_train_rshp.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, 1), return_sequences=False, go_backwards=False,
               W_regularizer=l2(0.005), U_regularizer=l2(0.005),
               inner_init='glorot_normal', init='glorot_normal', activation='tanh'))  # try using a GRU instead, for fun
model.add(Dense(1, activation='linear'))

# try using different optimizers and different optimizer configs
model.compile(loss='mse', optimizer='rmsprop')
print('Train...')
model.fit(X_train_rshp, y_train, batch_size=32, nb_epoch=5)
print(model.evaluate(X_test_rshp, y_test))
y_pred = model.predict(X_test_rshp)

