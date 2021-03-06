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
np.random.seed(671)
np.random.shuffle(X)
np.random.seed(671)
np.random.shuffle(y)

print('Signal ratio:', np.sum(y) / y.shape[0])

# preprocess the data
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
nb_train_sample = int(len(y) * 0.5) 
X_train = X[:nb_train_sample]
X_eval = X[nb_train_sample:]
y_train = y[:nb_train_sample]
y_eval = y[nb_train_sample:]

print('Train on:', X_train.shape[0])
print('Eval on:', X_eval.shape[0])

# deep pyramidal MLP, narrowing with depth
model = Sequential()
model.add(Dropout(0.13))
model.add(Dense(X_train.shape[1], 75))
model.add(PReLU((75,)))

model.add(Dropout(0.11))
model.add(Dense(75, 50))
model.add(PReLU((50,)))

model.add(Dropout(0.09))
model.add(Dense(50, 30))
model.add(PReLU((30,)))

model.add(Dropout(0.07))
model.add(Dense(30, 25))
model.add(PReLU((25,)))

model.add(Dense(25, 2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# train model
model.fit(X_train, y_train, batch_size=128, nb_epoch=200, validation_data=(X_eval, y_eval), verbose=2, show_accuracy=True)


# generate submission
ids, X = get_test_data()
print('Data shape:', X.shape)
X, scaler = preprocess_data(X, scaler)
predskeras = model.predict(X, batch_size=256)[:, 1]

print("Load the training/test data...")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

features = list(train.columns[1:-5])
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
clf = UGradientBoostingClassifier(loss=loss, n_estimators=100, subsample=0.1, # n_estimators = 75
                                  max_depth=7, min_samples_leaf=10,
                                  learning_rate=0.1, train_features=features, random_state=11)
clf.fit(train[features + ['mass']], train['signal'])
fb_preds = clf.predict_proba(test[features])[:,1]
print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, criterion="entropy", random_state=1)
rf.fit(train[features], train["signal"]) 


params = {"objective": "binary:logistic",
          "eta": 0.15,
          "max_depth": 6, 
          "min_child_weight": 1,
          "silent": 1,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=200
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)


test_probs = (0.3*rf.predict_proba(test[features])[:,1]) + (0.3*gbm.predict(xgb.DMatrix(test[features])))+(0.2*predskeras) + (0.2*fb_preds) #is better
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("submission.csv", index=False)