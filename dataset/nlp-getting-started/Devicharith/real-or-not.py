import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
train_text = train.iloc[:, 3].values
test_text = test.iloc[:, 3].values


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential

embed = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/3', trainable=True)

def model_(embed):
    model = tf.keras.Sequential()
    model.add(Input(shape=(),dtype=tf.string))
    model.add(embed)
    model.add(Dense(128, activation='relu'))
    model.add( BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = model_(embed)
y = train.target.values

model.fit(
    train_text,y,epochs=10,verbose=1,batch_size=32)

y_pred = model.predict_classes(test_text,batch_size = 100)
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv('withoutsubmission.csv', index=False)