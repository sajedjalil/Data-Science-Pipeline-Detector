import numpy as np
import pandas as pd
import tensorflow as tf
import keras

train_data = pd.read_csv('../input/train.csv')
train_data = train_data.values

features = train_data[:, 1:55]
targets = train_data[:, 55]

oneHotTargets = np.zeros([15120, 7])

for i in range(0, 15120):
    for j in range(0, 7):
        if targets[i] == j + 1:
            oneHotTargets[i][j] = 1
            
targets = oneHotTargets

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

model = Sequential()

model.add(Dense(20, input_shape = (54,)))
model.add(BatchNormalization())

model.add(Dense(50, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.05))

model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.05))

model.add(Dense(80, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.05))

model.add(Dense(30, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(x=features, y=targets, epochs=50)

test_data = pd.read_csv('../input/test.csv')
test_data = test_data.values

features = test_data[:, 1:55]

predictions = model.predict(features)
Ids = np.array([i for i in range(15121, 581013)])

testPredictions1 = []

for i in range(0, 565892):
    testPredictions1.append(list.index(list(predictions[i]), max(predictions[i])) + 1)

testPredictions = pd.DataFrame(Ids, columns={"Id"})
testPredictions['Cover_Type'] = testPredictions1
testPredictions = testPredictions[["Id", "Cover_Type"]]
testPredictions.to_csv('sample_submission.csv', sep=',', index=False)