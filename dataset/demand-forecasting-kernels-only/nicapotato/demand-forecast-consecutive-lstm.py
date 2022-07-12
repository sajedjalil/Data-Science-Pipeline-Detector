## Demand Forecast: Consecutive LSTM Solution
# By Nick Brooks, July 2018

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import gc

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import TimeDistributed
# Utility
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers

print("Data Load Stage")
training = pd.read_csv('../input/train.csv', parse_dates = ["date"])
testing = pd.read_csv('../input/test.csv', parse_dates = ["date"])

# Scale
scaler = MinMaxScaler()
print(scaler.fit(training['sales'].values.reshape(-1, 1)))
training['sales'] = scaler.transform(training['sales'].values.reshape(-1, 1))

# Merge
df = pd.concat([training,testing.drop("id",axis=1)],axis=0, sort=True)
# Reshape
df = df.groupby(["item","store","date"]).sum().reset_index()
train = (df.loc[df.date<pd.to_datetime('2018-01-01')].drop(["item","store","date"],axis=1).values
                       .reshape(500,1826,df.drop(["item","store","date"],axis=1).shape[1]))
print("Train Shape: ", train.shape)
test = (df.loc[df.date>=pd.to_datetime('2018-01-01')].drop(["item","store","date"],axis=1).values
                       .reshape(500,90,df.drop(["item","store","date"],axis=1).shape[1]))
print("Test Shape: ", test.shape)

# 3D Matrix
n_samples = 500

print("For Prediction")
y_train = train[:,-1,:]
X_train = train[:,-1823:,:]
print("y train Shape: ",y_train.shape)
print("X train Shape: ",X_train.shape)

print("\nCreate Validation Set")
y_valid = train[:,-2,:]
X_valid = train[:,:-3,:]
print("y Valid Shape: ",y_valid.shape)
print("X Valid Shape: ",X_valid.shape)

# Model Input Shape
inputshape = (X_train.shape[1], X_train.shape[2])
print(inputshape)

LSTM_PARAM = {"batch_size":30,
              "verbose":2,
              "epochs":10}

# Model Architecture
model_lstm = Sequential([
    LSTM(100, input_shape=inputshape),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation = 'linear')
])

# Objective Function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Compile
opt = optimizers.Adam()
model_lstm.compile(optimizer=opt, loss="mae")

modelstart = time.time()
callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=4, mode='auto')]
hist = model_lstm.fit(X_valid, y_valid,
                      validation_data=(X_train, y_train),
                      callbacks=callbacks_list,
                      **LSTM_PARAM)

# Model Evaluation
best = np.argmin(hist.history["val_loss"])
print("Optimal Epoch: ",best+1)
print("Train Score: {}, Validation Score: {}".format(hist.history["loss"][best],hist.history["val_loss"][best]))

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.title("Train and Validation Error")
plt.legend()
plt.savefig("Train and Validation MSE Progression.png")
plt.show()

# Final Params
LSTM_PARAM = {"batch_size":30,
              "verbose":0,
              "epochs":2}

# Consecutive LSTM
print("Train Shape Before: ", train.shape)
for iteration in range(90):
    # 3D Matrix
    n_samples = 500
    y = train[:,-1,:]
    X = train[:,-372:-2,:]
    test = train[:,-370:,:]
    # Model Input Shape
    inputshape = (X.shape[1], X.shape[2])
                  
    # Model Architecture
    model_lstm = Sequential([
        LSTM(100, input_shape=inputshape),
        Activation('relu'),
        Dropout(0.5),
        Dense(10),
        Activation('relu'),
        Dropout(0.5),
        Dense(1, activation = 'linear')
    ])

    # Compile
    opt = optimizers.Adam()
    model_lstm.compile(optimizer=opt, loss="mae")

    modelstart = time.time()
    hist = model_lstm.fit(X,y,**LSTM_PARAM)
    pred = model_lstm.predict(test)
    scaled = scaler.inverse_transform(np.array([x for sl in pred for x in sl]).reshape(-1, 1))
    reshape_pred = scaled.reshape(scaled.shape[0], 1, 1)
    train = np.concatenate([train, reshape_pred], axis=1)
    print("Iteration: ", iteration + 1)
    del model_lstm
print("Train Shape After: ", train.shape)

# Submit
submission = pd.Series(train[:,-90:,:].reshape(90*500))
submission.rename("sales", inplace=True)
submission.index.name = "id"
submission.to_csv("LSTM_submission.csv",index=True,header=True)
submission.head()
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))