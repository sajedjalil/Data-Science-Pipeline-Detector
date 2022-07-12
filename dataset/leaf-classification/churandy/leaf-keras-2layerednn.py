# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

np.random.seed(1337)

# Load train and test sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Target
le = LabelEncoder()
target = le.fit_transform(train.species)
print("Target shape: {}".format(target.shape))
print(np.unique(target).size)
print(target[:5])

# Training data
#print(train.columns)
species = np.sort(train.species.unique())
train.drop(['id', 'species'], axis=1, inplace=True)
print('Train data shape: {}'.format(train.shape))

# Scale training data
scaler = StandardScaler().fit(train.values)
train = scaler.transform(train.values)

# Create random train and validation sets out of 15% samples
Xtrain, Xval, ytrain, yval = train_test_split(train, target, test_size=0.15,
                                            stratify=target, random_state=11)
print('\nXtrain, ytrain shapes ' + str((Xtrain.shape, ytrain.shape)))
print('Xval, yval shapes ' + str((Xval.shape, yval.shape)))

# Training parameters
print('Model Parameters:')
batch_size = 128
nb_epoch = 50
nb_classes = np.unique(target).size
print('Batch size: %d, epochs: %d, n classes: %d, ' % (batch_size, nb_epoch, nb_classes))

# convert class vectors to binary class matrices (one-hot encoder)
ytrain = np_utils.to_categorical(ytrain, nb_classes)
yval = np_utils.to_categorical(yval, nb_classes)
print(ytrain.shape, yval.shape)

print('Training neural network...')
# Create model
# First hidden layer
nhidden1 = 800
print('(hidden layer 1: %d units)' % nhidden1)
model = Sequential()
model.add(Dense(nhidden1, input_shape=(192,)))
model.add(Activation('relu'))
model.add(Dropout(0.5)) # Prevent overfitting
# second hidden layer
nhidden2 = 200
print('(hidden layer 2: %d units)' % nhidden2)
model.add(Dense(nhidden2, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation("softmax"))
# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model
model.fit(Xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(Xval, yval))
score = model.evaluate(Xval, yval, verbose=0)
print('Validation loss: %0.5f' % score[0])
print('Validation accuracy: %0.2f' % (100*score[1]))

print('\nIterating through the training and cross validation sets...')
cv = StratifiedKFold(target, n_folds=8, shuffle=True)
losses = []
accurs = []
target = np_utils.to_categorical(target, nb_classes)
for traincv, testcv in cv:
    model.fit(train[traincv], target[traincv], batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0)#, validation_data=(Xval, yval))
    score = model.evaluate(train[testcv], target[testcv], verbose=0)
    print('Validation loss: %0.5f, Validation accuracy: %0.5f' % (score[0], score[1]))
    losses.append(score[0])
    accurs.append(score[1])
    
#print out the mean of the cross-validated results
print("Mean log-loss: %0.5f. Mean accuracy: %0.5f" % 
                        (np.array(losses).mean(),np.array(accurs).mean()))
                        
# Test predictions
ids = test.id
test.drop(['id'], axis=1, inplace=True)
test = scaler.transform(test.values)
preds = model.predict_proba(test)

# Submit
submission = pd.DataFrame(preds,index=ids,columns=species)
submission.to_csv('Leaf_Keras_NN.csv')
