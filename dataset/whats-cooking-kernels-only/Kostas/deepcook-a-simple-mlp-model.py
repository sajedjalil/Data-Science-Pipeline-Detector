import gensim
import numpy as np
import pandas as pd

import keras
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Read data
data = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

# Vectorise text in training and test sets using TFIDF 
tfidf = TfidfVectorizer(lowercase=False)
training_data = tfidf.fit_transform([' '.join(doc).lower() for doc in data.ingredients])
td = training_data.toarray()

X_test = tfidf.transform([' '.join(doc).lower() for doc in test.ingredients])
X_test = X_test.toarray()

# encode the target variable
lb = LabelEncoder()
target = data.cuisine
y = lb.fit_transform(target)
one_hot_lb = to_categorical(y)

# MLP
model = Sequential()
model.add(layers.Dense(256, input_shape=(td.shape[1], )))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(20))
model.add(layers.Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(td, one_hot_lb, batch_size=32, epochs=18)

# Predictions
uniques, ids = np.unique(y, return_inverse=True)
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(uniques[y_test.argmax(1)])

test_id = [id_ for id_ in test.id]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('mlp_output.csv', index=False)