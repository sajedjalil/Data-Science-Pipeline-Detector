# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.preprocessing.text import Tokenizer
from keras import utils
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

trainfile = "../input/train.json"
data = pd.read_json(trainfile)
data['ingredients'] = data.ingredients.map(lambda x: " ".join(x) )
max_chars_train = data.ingredients.map(lambda x: sum(len(i) for i in x) ).max()
max_words_train = data.ingredients.map(lambda x: (len(x)) ).max()
print('data shape',data.shape)

test_file = DATA_PATH = '../input/test.json'
test_data = pd.read_json(test_file)
test_data['ingredients'] = test_data.ingredients.map(lambda x: " ".join(x) )
X_test = test_data['ingredients']
max_chars_test = test_data.ingredients.map(lambda x: sum(len(i) for i in x) ).max()
max_words_test = test_data.ingredients.map(lambda x: (len(x)) ).max()

unique_targets = data.cuisine.unique()
num_classes = len(unique_targets)

y = data['cuisine']
class_weights = class_weight.compute_class_weight('balanced', np.unique(y),  y)

X = data['ingredients']
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X, y, test_size=0.16, random_state=42)
del X
del y

max_words = max(max_chars_train, max_chars_test, 1000)
tokenize = Tokenizer(num_words=max_words)

tokenize.fit_on_texts(X_train_split) # only fit on train
X_train = tokenize.texts_to_matrix(X_train_split)
X_valid = tokenize.texts_to_matrix(X_valid_split)
X_test = tokenize.texts_to_matrix(X_test)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(y_train_split)
y_train = encoder.transform(y_train_split)
y_valid = encoder.transform(y_valid_split)

# Converts the labels to a one-hot representation
y_train = utils.to_categorical(y_train, num_classes)
y_valid = utils.to_categorical(y_valid, num_classes)

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])

model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=2,
                    validation_data=(X_valid, y_valid), shuffle=True, initial_epoch=0, class_weight=class_weights)
                    
text_labels = encoder.classes_ 
SUBMISSION = "sample_submission.csv"
prediction = model.predict(X_test)
predicted_label = text_labels[np.argmax(prediction, axis=1)]
test_data['cuisine'] = predicted_label
test_data.drop(['ingredients'], axis=1, inplace=True)
test_data.to_csv(SUBMISSION, index=False)