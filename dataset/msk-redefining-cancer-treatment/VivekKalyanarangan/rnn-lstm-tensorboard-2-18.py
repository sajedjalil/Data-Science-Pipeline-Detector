# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:20:59 2017

@author: VK046010
"""
import os
import pickle

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model as plot
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import load_model

# Hyper params
MAXLEN = 100
MAX_FEATURES = 200000
BATCH_SIZE = 32
MAX_NB_WORDS=1000
EPOCHS=1

MODEL_ARCH_FOLDER = 'model_architecture/'
MODEL_CHECKPOINT_FOLDER = 'checkpoint/'
TOKENIZER = 'tokenizer.pkl'
MODEL_NAME = 'weights-improvement-.hdf5'
print('Hyper params set')

# Load train
data_train = pd.read_csv('training_text',sep=r"\|(\|)",engine='python')
data_train=data_train.drop(['|'],axis=1)
print('Training Data Loaded')

# Load variants
data_training_variants = pd.read_csv('training_variants')
print('Variants loaded')

# Merge X and Y
data_train_merge = pd.merge(data_train,data_training_variants,how='inner',on='ID')
print('Data merged')

# Dense layer can be encoded to 9. Add 1 to prediction class
data_train_merge['Class'] = data_train_merge['Class']-1
print('output variable differenced')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

print('Tokenizing...')
tokenizer.fit_on_texts(data_train_merge['Text'])

# Persist tokenizer for test use
with open(MODEL_CHECKPOINT_FOLDER+TOKENIZER,'wb') as tokenizer_pkl:
    pickle.dump(tokenizer,tokenizer_pkl)

print('Converting text to sequences...')
sequences_train = tokenizer.texts_to_sequences(data_train_merge['Text'])

word_index = tokenizer.word_index

print('Preparing data...')
x = sequence.pad_sequences(sequences_train, maxlen=MAXLEN)
y = np.array(data_train_merge['Class'])

y_binary = to_categorical(y)

print('Split train and test...')
rng = np.random.RandomState(42)
n_samples = len(x)
indices = np.arange(n_samples)
rng.shuffle(indices)
x_shuffled = x[indices]
y_shuffled = y[indices]

x_train = x_shuffled[:int(n_samples*0.8)]
x_test = x_shuffled[int(n_samples*0.8):]

y_train = y_shuffled[:int(n_samples*0.8)]
y_test = y_shuffled[int(n_samples*0.8):]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Build model...')
model = Sequential()
model.add(Embedding(MAX_FEATURES, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='sigmoid'))

model_pretrained_path = MODEL_CHECKPOINT_FOLDER + MODEL_NAME
if os.path.exists(model_pretrained_path):
    model = load_model(model_pretrained_path)
else:
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    plot(model, to_file=MODEL_ARCH_FOLDER+'model.png', show_shapes=True)
    json_string = model.to_json()
    open(MODEL_ARCH_FOLDER+'model_architecture.json', 'w').write(json_string)

print('Train...')
tensorBoardCallback = TensorBoard(log_dir='./tb_logs', write_graph=True, histogram_freq=1)
filepath = MODEL_CHECKPOINT_FOLDER + "weights-improvement-.hdf5" #{epoch:02d}-{val_acc:.2f}
checkPointCallback = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [tensorBoardCallback, checkPointCallback]

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))