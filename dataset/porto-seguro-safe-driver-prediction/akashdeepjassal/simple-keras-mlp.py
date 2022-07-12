#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import keras
import sklearn.model_selection
import numpy as np
import pandas as pd


# Load Datasets
df_train  = pd.read_csv('../input/train.csv')
df_test   = pd.read_csv('../input/test.csv')
df_submit = pd.read_csv('../input/sample_submission.csv')

# To numpy array - dataset of train
x_all = df_train.drop(['target', 'id'], axis=1).values
y_all = keras.utils.np_utils.to_categorical(df_train['target'].values)

# For imbalanced data, better-way maybe exist!
# Please tell me better way by comment! Thanks!!
y_all_0 = y_all[y_all[:,1]==0]
y_all_1 = y_all[y_all[:,1]==1]
x_all   = np.concatenate([x_all[y_all[:,1]==0], np.repeat(x_all[y_all[:,1]==1], repeats=int(len(y_all_0)/len(y_all_1)), axis=0)], axis=0)
y_all   = np.concatenate([y_all[y_all[:,1]==0], np.repeat(y_all[y_all[:,1]==1], repeats=int(len(y_all_0)/len(y_all_1)), axis=0)], axis=0)

# Split train/valid datasets
x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.3, random_state=0)

# Define model
model = keras.models.Sequential()
model.add(keras.layers.normalization.BatchNormalization(input_shape=tuple([x_train.shape[1]])))
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.core.Dense(2,   activation='sigmoid'))
model.compile(loss="categorical_crossentropy", optimizer="adadelta",metrics=["accuracy"])
print(model.summary())

# Use Early-Stopping
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

# Train model
model.fit(x_train, y_train, batch_size=1024, epochs=200, validation_data=(x_valid, y_valid), verbose=1, callbacks=[callback_early_stopping])

# Predict test dataset
x_test = df_test.drop(['id'], axis=1).values
y_test = model.predict(x_test)

# Output
df_submit['target'] = y_test[:, 1]
df_submit.to_csv('submission_output.csv', index=False)