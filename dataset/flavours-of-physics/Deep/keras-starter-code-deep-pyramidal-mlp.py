import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler


def get_training_data():
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits']
    f = open('../input/training.csv')
    data = []
    y = []
    ids = []
    for i, l in enumerate(f):
        if i == 0:
            labels = l.rstrip().split(',')
            label_indices = dict((l, i) for i, l in enumerate(labels))
            continue

        values = l.rstrip().split(',')
        filtered = []
        for v, l in zip(values, labels):
            if l not in filter_out:
                filtered.append(float(v))

        label = values[label_indices['signal']]
        ID = values[0]

        data.append(filtered)
        y.append(float(label))
        ids.append(ID)
    return ids, np.array(data), np.array(y)


def get_test_data():
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits']
    f = open('../input/test.csv')
    data = []
    ids = []
    for i, l in enumerate(f):
        if i == 0:
            labels = l.rstrip().split(',')
            continue

        values = l.rstrip().split(',')
        filtered = []
        for v, l in zip(values, labels):
            if l not in filter_out:
                filtered.append(float(v))

        ID = values[0]
        data.append(filtered)
        ids.append(ID)
    return ids, np.array(data)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

# get training data
ids, X, y = get_training_data()
print('Data shape:', X.shape)

# shuffle the data
np.random.seed(1337)
np.random.shuffle(X)
np.random.seed(1337)
np.random.shuffle(y)

print('Signal ratio:', np.sum(y) / y.shape[0])

# preprocess the data
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
nb_train_sample = int(len(y) * 0.95)
X_train = X[:nb_train_sample]
X_eval = X[nb_train_sample:]
y_train = y[:nb_train_sample]
y_eval = y[nb_train_sample:]

print('Train on:', X_train.shape[0])
print('Eval on:', X_eval.shape[0])

L1 = 100
L2 = 75
L3 = 50
L4 = 25
D1 = 0.13
D2 = 0.11
D3 = 0.09
D4 = 0.07

# deep pyramidal MLP, narrowing with depth
model = Sequential()
model.add(Dropout(D1))
model.add(Dense(X_train.shape[1], L1))
model.add(PReLU((L1,)))

model.add(Dropout(D2))
model.add(Dense(L1, L2))
model.add(PReLU((L2,)))

model.add(Dropout(D3))
model.add(Dense(L2, L3))
model.add(PReLU((L3,)))

model.add(Dropout(D4))
model.add(Dense(L3, L4))
model.add(PReLU((L4,)))

model.add(Dense(L4, 2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# train model
model.fit(X_train, y_train, batch_size=64, nb_epoch=85, validation_data=(X_eval, y_eval), verbose=2, show_accuracy=True)

# generate submission
ids, X = get_test_data()
print('Data shape:', X.shape)
X, scaler = preprocess_data(X, scaler)
preds = model.predict(X, batch_size=256)[:, 1]
with open('submission.csv', 'w') as f:
    f.write('id,prediction\n')
    for ID, p in zip(ids, preds):
        f.write('%s,%.8f\n' % (ID, p))
