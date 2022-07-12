# BASIC IDEA OF THE KERNEL

# The data consists of a one dimensional time series x with 600 Mio data points. 
# At test time, we will see a time series of length 150'000 to predict the next earthquake.
# The idea of this kernel is to randomly sample chunks of length 150'000 from x, derive some
# features and use them to update weights of a recurrent neural net with 150'000 / 1000 = 150
# time steps. 

# 重新修改了一下代码。框架仍然为初始时的框架。

import numpy as np 
import pandas as pd
import os
from tqdm import tqdm

# Import
float_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values

# Fix seeds
from numpy.random import seed
seed(639)
from tensorflow import set_random_seed
set_random_seed(5944)

# NEW
# 以10个数据为一组，求平均值。
# 原int变成了float
def databinning(data, test=0):
    i=0
    l=len(data)
    l = l-10
    #print(l)
    if test==0:
        data2=np.zeros(int((l+10)/10*2), dtype=float).reshape(-1,2)
        while 10*i <= l:
            data2[i][0]=(np.mean(data[:,0][10*i:10*i+10]))
            data2[i][1]=np.mean(data[:,1][10*i:10*i+10])
            i = i+1
            #print(i)
    ### 数据结构可能存在混淆之处
    elif test==1:
        data2=np.zeros(int(l/10), dtype=float)
        while 10*i <= l:
            data2[i]=(np.mean(data[:,0][10*i:10*i+10]))
            i = i+1
    return data2

# Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.
# Can easily be extended. Expects a two dimensional array.
def extract_features(z, step_length):
     return np.c_[z.mean(axis=1), 
                  np.quantile(z, 0.05, axis=1),
                  np.quantile(z, 0.99, axis=1),
                  #z.max(axis=1),
                  z.std(axis=1),
                  z[:, -step_length // 10:].mean(axis=1), 
                  np.quantile(z[:, -step_length // 10:], 0.05, axis=1),
                  z[:, -step_length // 10:].max(axis=1),
                  z[:, -step_length // 10:].std(axis=1),
                  z[:, -step_length // 100:].mean(axis=1), 
                  np.quantile(z[:, -step_length // 100:], 0.05, axis=1),
                  np.quantile(z[:, -step_length // 100:], 0.95, axis=1),
                  #z[:, -step_length // 100:].max(axis=1),
                  #z[:, -step_length // 100:].std(axis=1),
                  ]

# For a given ending position "last_index", we split the last 150'000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
# From each piece, a set features are extracted. This results in a feature matrix 
# of dimension (150 time steps x features).  
def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(temp, step_length)]

# Query "create_X" to figure out the number of features
n_features = create_X(float_data[0:150000]).shape[1]
print("Our RNN is based on %i features"% n_features)
    
# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
         
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size)
        
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row - 1, 1]
        yield samples, targets
        
        
# Position of second (of 16) earthquake. Used to have a clean split
# between train and validation
second_earthquake = 50085877
float_data[second_earthquake, 1]

# Initialize generators
#train_gen = generator(float_data, batch_size=32) # Use this for better score
train_gen = generator(float_data, batch_size=32, min_index=second_earthquake + 1)
valid_gen = generator(float_data, batch_size=16, max_index=second_earthquake)

# Define model
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]

model = Sequential()
model.add(CuDNNGRU(48, input_shape=(None, n_features)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,
                              epochs=30,
                              verbose=0,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=200)

# Visualize accuracies
import matplotlib.pyplot as plt

def perf_plot(history, what = 'loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1
    
    plt.plot(epochs, x, 'bo', label = "Training " + what)
    plt.plot(epochs, val_x, 'b', label = "Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    plt.savefig('loss.jpg')
    print(history.history)
    print(history.epoch)
    print(history.history['val_loss'])

    return None

perf_plot(history)

# Load submission file
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

#submission.head()

# Save
submission.to_csv('submission.csv')