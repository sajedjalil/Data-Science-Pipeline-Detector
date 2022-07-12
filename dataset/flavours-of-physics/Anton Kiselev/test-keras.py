import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

def get_training_data():
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits', 'IP', 'IPSig', 'isolationc']
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
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits', 'IP', 'IPSig', 'isolationc']
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
np.random.seed(46) # used to be 369
np.random.shuffle(X)
np.random.seed(46)
np.random.shuffle(y)

print('Signal ratio:', np.sum(y) / y.shape[0])

# preprocess the data
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
#nb_train_sample = int(len(y) * 0.78) # used to be 0.97, 0.78 is better, 0.83 possible
X_train = X
#X_eval = X
y_train = y
#y_eval = y

print('Train on:', X_train.shape[0])
#print('Eval on:', X_eval.shape[0])


a=50
b=45
c=30

# deep pyramidal MLP, narrowing with depth
model = Sequential()
model.add(Dropout(0.13, input_shape=(X_train.shape[1],)))
model.add(Dense(input_shape=(X_train.shape[1],), output_dim=a))
model.add(PReLU())

model.add(Dropout(0.11))
model.add(Dense(input_shape=(a,), output_dim=b))
model.add(PReLU())

model.add(Dropout(0.09))
model.add(Dense(input_shape=(b,), output_dim=c))
model.add(PReLU())

model.add(Dense(input_shape=(c,), output_dim=2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# train model
model.fit(X_train, y_train, batch_size=128, nb_epoch=50, verbose=2, show_accuracy=True)
# nb_epoch = 50, 100 fine

# generate submission
ids, X = get_test_data()
print('Data shape:', X.shape)
X, scaler = preprocess_data(X, scaler)
predskeras = model.predict(X, batch_size=256)[:, 1]

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Make predictions on the test set")
# test_probs = (0.35*rf.predict_proba(test[features])[:,1]) + (0.35*gbm.predict(xgb.DMatrix(test[features])))+(0.15*predskeras) + (0.15*fb_preds) 
test_probs = predskeras 
# test_probs = (0.25*rf.predict_proba(test[features])[:,1]) + (0.25*gbm.predict(xgb.DMatrix(test[features])))+(0.25*predskeras) + (0.25*fb_preds)
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("keras_v1.csv", index=False)