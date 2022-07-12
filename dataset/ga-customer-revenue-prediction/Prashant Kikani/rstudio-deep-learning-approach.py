import os
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
import gc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math as mt
from pylab import savefig

model_path = "keras_baseline.h5"
print("train read")
X = pd.read_csv('../input/rstudio-data/X.csv')
y = pd.read_csv('../input/rstudio-data/y.csv', header = None)
print("test read")
X_test = pd.read_csv('../input/rstudio-data/X_test.csv')
testid = pd.read_csv('../input/rstudio-data/testid.csv')
print("done")

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint

def model():
    model = Sequential()
    #input layer
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    # hidden layers
    model.add(Dense(input_dims))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.4))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.4))
    
    model.add(Dense(input_dims//4, activation=act_func))
    
    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))
    
    # compile this model
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                 )
    # Visualize NN architecture
    print(model.summary())
    return model
    
act_func = 'tanh'
input_dims = X.shape[1]
estimator = KerasRegressor(
    build_fn=model, 
    nb_epoch=300, 
    batch_size=30,
    verbose=1
)
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=20,
            mode='min',
            verbose=2),
        ModelCheckpoint(model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min',
        verbose=2)
]
# fit estimator
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state = 7)

history = estimator.fit(
        X_train, 
        y_train,
        validation_data=(X_valid,y_valid),
        epochs=50,
        verbose=2,
        callbacks=callbacks,
        shuffle=True
    )

estimator = load_model(model_path)
ans = estimator.predict(X_test)

submission = testid
submission.loc[:, 'PredictedLogRevenue'] = ans
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('dl.csv',index=False)

'''
# prepare callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', 
            patience=20,
            mode='min',
            verbose=2),
    ModelCheckpoint(model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min',
        verbose=2)
]

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=7
)

# fit estimator
history = estimator.fit(
    X_tr, 
    y_tr, 
    epochs=500,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)

# summarize history for R^2
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model accuracy')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_accuracy.png")

# summarize history for loss
fig_loss = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig("model_loss.png")

# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    estimator = load_model(model_path)

# Plot in blue color the predicted data and in green color the
# actual data to verify visually the accuracy of the model.
predicted = estimator.predict(X_val)
fig_verify = plt.figure(figsize=(100, 50))
plt.plot(predicted, color="blue")
plt.plot(y_val, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()
fig_verify.savefig("model_verify.png")

# predict results
res = estimator.predict(X_test).ravel()
'''