'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
import csv
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

ID = 0
NAME = 1
DATE = 2
OUTCOME = 3
OUTCOME_SUB = 4
TYPE = 5
SEX = 6
AGE = 7
BREED = 8
COLOR = 9

def generateDict():
    f = open('../input/train.csv', 'r')
    num_of_col = 10  ## Here is a result
    csvReader = csv.reader(f)
    count = []
    for i in range(0,num_of_col):
        count.append(0)
    dict = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

    row_idx = 0
    for row in csvReader:
        if (row[1] == 'Name'):
            continue
        for col in range(0, num_of_col):
            if (col == ID) or (col == DATE): ## Ignore ID and DATE
                continue
            if None == dict[col].get(row[col], None):    ### Get Histogram Map
                dict[col][row[col]] = count[col]
                count[col] = count[col] + 1
            #print('[', row_idx, col, '] - ', row[col], ' - ', dict[col][row[col]])
        row_idx = row_idx + 1
    print(count)
    f.close
    return dict

def load_train_data(dict, begin, end):
    f = open('../input/train.csv', 'r')
    num_of_col = 10  ## Here is a result
    csvReader = csv.reader(f)
    column = [[0]*num_of_col for x in range(end - begin)]
    result = [0 for x in range(end - begin)]

    row_idx = 0
    idx = 0;
    for row in csvReader:
        idx = idx + 1
        if (row[1] == 'Name'):
            continue
        if (idx < begin or idx >= end): # To separate Train_data_set and  Validation_data_set
            continue
        for col in range(0, num_of_col):
            if (col == ID) or (col == DATE):
                continue

            if (col != OUTCOME and col != OUTCOME_SUB):
                column[row_idx][col] = dict[col][row[col]]  ### Input data
            elif (col == OUTCOME):
                result[row_idx] = dict[col][row[col]]  ### Result
            #print('[', row_idx, col, '] - ', row[col], ' - ', dict[col][row[col]])
        row_idx = row_idx + 1
    del column[row_idx:(end - begin)]
    del result[row_idx:(end - begin)]
    f.close
    return column, result

def load_test_data(dict):
    f = open('../input/test.csv', 'r')
    num_of_col = 10  ## Here is no result
    csvReader = csv.reader(f)
    column = [[0]*num_of_col for x in range(30000)]

    row_idx = 0
    for row in csvReader:
        if (row[1] == 'Name'):
            continue
        for col in range(0, num_of_col):
            if (col == ID) or (col == DATE):
                continue
            if (col == OUTCOME) or (col == OUTCOME_SUB):
                continue

            if (col >= OUTCOME_SUB):  ## Because the row[]( = test.csv) doesn't include OUTCOME, OUTCOME_SUB
                row_col = row[col-2]
            else:
                row_col = row[col]
            if None == (dict[col].get(row_col, None)):  ### If new key, update dict
                dict[col][row_col] = len(dict[col])
            column[row_idx][col] = dict[col][row_col]  ### Input data
            #print('[', row_idx, col, '] - ', row[col], ' - ', dict[col][row[col]])
        row_idx = row_idx + 1
    del column[row_idx:30000]
    f.close
    return column

batch_size = 128
nb_classes = 5
nb_epoch = 20

dict = generateDict()
X_train, Y_train = load_train_data(dict, 0, 30000)
X_train = np.array(X_train, np.int32)
X_train = X_train.astype('float32')
X_train /= 6375
print(X_train.shape[0], 'train samples')

X_test, Y_test = load_train_data(dict, 8001, 10000)
X_test = np.array(X_test, np.int32)
X_test = X_test.astype('float32')
X_test /= 6375

print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

print('Y_train: ', Y_train[0:20], '...')
print('X_train[0] Length: ', len(X_train[0]))
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(len(X_train[0]),)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
print('---------------')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

X_test = load_test_data(dict)
preds = model.predict_proba(X_test)

of = open('./prediction.csv', 'w')
csvWriter = csv.writer(of, delimiter=',')
csvWriter.writerow(['AnimalID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner' ,'Transfer'])
for i, row in enumerate(preds):
    csvWriter.writerow([i+1] + list(row))
of.close
