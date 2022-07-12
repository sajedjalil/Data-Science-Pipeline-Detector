import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD

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
np.random.seed(369)
np.random.shuffle(X)
np.random.seed(369)
np.random.shuffle(y)

print('Signal ratio:', np.sum(y) / y.shape[0])

# preprocess the data
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
nb_train_sample = int(len(y) * 0.97)
X_train = X[:nb_train_sample]
X_eval = X[nb_train_sample:]
y_train = y[:nb_train_sample]
y_eval = y[nb_train_sample:]

print('Train on:', X_train.shape[0])
print('Eval on:', X_eval.shape[0])

# deep pyramidal MLP, narrowing with depth
model = Sequential()
model.add(Dense(X_train.shape[1], 32, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(32, 16, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
#model.add(Dense(16, 8, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0))
model.add(Dense(16, 2, init='uniform'))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer="adadelta")
model.compile(loss='mean_squared_error', optimizer="adadelta")


# train model
model.fit(X_train, y_train, batch_size=256, nb_epoch=50, validation_data=(X_eval, y_eval), verbose=2, show_accuracy=True)

# generate submission
ids, X = get_test_data()
print('Data shape:', X.shape)
X, scaler = preprocess_data(X, scaler)
preds = model.predict(X, batch_size=256)[:, 1]
with open('keras_mlp_sgd.csv', 'w') as f:
    f.write('id,prediction\n')
    for ID, p in zip(ids, preds):
        f.write('%s,%.8f\n' % (ID, p))
