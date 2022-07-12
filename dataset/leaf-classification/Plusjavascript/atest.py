
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

## Read data from the CSV file
data = pd.read_csv('../input/train.csv')
parent_data = data.copy()
ID = data.pop('id')

## Since the labels are textual, so we encode them categorically
y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation
y_cat = to_categorical(y)

## Most of the learning algorithms are prone to feature scaling
## Standardising the data to give zero mean =)
X = StandardScaler().fit(data).transform(data)

model = Sequential()
model.add(Dense(300, input_dim=192,  init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(99, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])
history = model.fit(X, y_cat, batch_size=100, nb_epoch=500, verbose=0, validation_split=0.1)

# summarize history for loss
## Plotting the loss with the number of iterations
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## read test file
test = pd.read_csv('../input/test.csv')
index = test.pop('id')
test = StandardScaler().fit(test).transform(test)
yPred = model.predict_proba(test)
cols = parent_data.species.unique()

## Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred,index=index,columns=sorted(cols))

fp = open('submission_nn_kernel.csv','w')
fp.write(yPred.to_csv())

from subprocess import check_output
print(check_output(["ls", "."]).decode("utf8"))
