'''
This script makes use of Neural networks using Keras library
Piece of code for encoding categorical variables is taken from Abhishek's script

@author: Harshaneel Gokhale

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
from keras.layers.advanced_activations import PReLU

# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Drop uselss columns and store useful columns.
train = train.drop('Id', axis=1)
target = train.Hazard.tolist()  
train = train.drop('Hazard', axis=1)

idx = test.Id.tolist()
test = test.drop('Id', axis=1)

# Encode few columns together for train and test.
columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# Encode categorical variables.
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

# scale
s = StandardScaler()
s.fit(train)
train = s.transform(train)
test = s.transform(test)

# Create NN model.
nn_model = Sequential()

# Add layers of computation.
nn_model.add(Dense(train.shape[1], 256, init='glorot_uniform', W_regularizer=l2(0.003)))
nn_model.add(PReLU((256,)))
nn_model.add(BatchNormalization((256)))
nn_model.add(Dropout(0.5))

nn_model.add(Dense(256, 256, init='glorot_uniform', W_regularizer=l2(0.003)))
nn_model.add(PReLU((256)))
nn_model.add(BatchNormalization(256))
nn_model.add(Dropout(0.5))

nn_model.add(Dense(256, 1, W_regularizer=l1l2(0.01, 0.01)))


# Compile and train model.
nn_model.compile(loss='mse', optimizer='rmsprop')
nn_model.fit(train, target, nb_epoch=20, batch_size=32, verbose=2, validation_split=0.2, shuffle=True)

# Create Predictions.
preds = np.expm1(nn_model.predict(test, verbose=0).flatten())

# Create Submission.
df =pd.DataFrame({'Id':idx, 'Hazard': preds})
df.to_csv('submission_nn.csv', index=False)