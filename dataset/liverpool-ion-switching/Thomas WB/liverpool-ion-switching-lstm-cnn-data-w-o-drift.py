# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Use a stacked LSTM model
# Use data preprocessed by Chris Deotte (https://www.kaggle.com/cdeotte/data-without-drift)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

os.system('pip install tensorflow_addons')

#__print__ = print
#def print(string):
#    os.system(f'echo \"{string}\"')
#    __print__(string)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
# Any results you write to the current directory are saved as output.

experience_results_path = '.'

import tensorflow as tf
from sklearn import preprocessing
import random
import time
import random

class DataRange():
  def __init__(self, data, range_idx, start_offset=0, scaler=None):
    self.range_idx = range_idx

    number_of_records_per_range = 500000
    idx_start = start_offset+range_idx*number_of_records_per_range
    idx_end = idx_start + number_of_records_per_range

    self.range_data = data[idx_start:idx_end]

  def get_data(self):
      return self.range_data

class DataRangeGenerator():

    def __init__(self, range_idx, n, step, data_range, scaler):
        self.range_idx = range_idx
        self.n = n
        self.step = step
        self.data_range = data_range
        self.scaler = scaler

        data = data_range.get_data()
        self.example_number = ((data.shape[0]-n)//step)+1

        self.x = None
        self.y = None

    def compute_range(self, data, example_number, range_idx):

        x = np.zeros((example_number, self.n, 1))
        y = np.zeros((example_number,1))

        for i in range(example_number):

            # Compute indexes
            input_start_idx = i * self.step
            input_end_idx = input_start_idx + self.n
            output_idx = input_end_idx-1

            # Get corresponding data
            x[i,:,:] = data[input_start_idx:input_end_idx,0].reshape((1, self.n, 1))
            y[i,:]  = int(data[output_idx,1])

            # Standardize input
            x[i,:,:] = (x[i,:,:] - self.scaler.mean_[0]) / self.scaler.scale_[0]

        return x, y

    def run(self):
        print('Generate range: {0} with {1} examples - Start'.format(self.range_idx,self.example_number))
        self.x, self.y = self.compute_range(self.data_range.get_data(),self.example_number, self.range_idx)
        print('Generate range: {0} - End'.format(self.range_idx))

    def get_data(self):
        return self.x, self.y

def transform_dataset(n, step):
    """ 
    From the signal/open channels value generate vector of signal 
    samples associated to an open channels value:
        n is the number of signal samples
        step is the number of time steps between signal samples
    """

    #----------------------------------------------------------------------------
    # Read the data from the original file
    #----------------------------------------------------------------------------
    df_train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')

    #----------------------------------------------------------------------------
    # Get only the signal and open_channels columns (2nd and 3rd column)
    #----------------------------------------------------------------------------
    data = df_train.values[:,:]
    data = data[:,1:]

    #----------------------------------------------------------------------------
    # Define the scaler for data standardization
    #----------------------------------------------------------------------------
    # Normally we should measure mean and scale only on the train data.
    # But at this moment we don't known the training examples
    scaler = preprocessing.StandardScaler().fit(data)
    print('Mean = {}'.format(scaler.mean_))
    print('Scale = {}'.format(scaler.scale_))

    #----------------------------------------------------------------------------
    # Define Data Range Parameters
    #----------------------------------------------------------------------------
    number_of_ranges = 10
    start_offset = 0

    #----------------------------------------------------------------------------
    # Prepare the list of data range for train, validation and evaluation
    #----------------------------------------------------------------------------
    # Ranges are separated in order to be able to multiplex data during
    # data generation
    data_range_list = list()
    for i in range(number_of_ranges):
        data_range_list.append(DataRange(data=data, range_idx=i, start_offset=start_offset))

    #----------------------------------------------------------------------------
    # Iterate over data, by multiplexing ranges
    #----------------------------------------------------------------------------
    x = None
    y = None
    generator_list = list()    
    print('Generate data per range')
    for range_idx in range(number_of_ranges):
        g = DataRangeGenerator(range_idx, n, step, data_range_list[range_idx], scaler)
        generator_list.append(g)
        g.run()

    print('Stack data')
    for range_idx in range(number_of_ranges):
        g = generator_list[range_idx]
        if range_idx == 0:
            x, y = g.get_data()
        else:
            x_tmp, y_tmp = g.get_data()

            x = np.vstack((x, x_tmp))
            y =  np.vstack((y, y_tmp))

    y = y.astype('int64')

    print('Genration done')

    return x, y, scaler.mean_, scaler.scale_

def generate_datasets(n):

    #----------------------------------------------------------------------------
    # 1) Generate the full example dataset depending on n (number_of_samples) and
    #    s (step_increment), but without hot encoding open channels yet, keep the
    #    integer value for the moment. This shall give a dataset of ((m-n)//s)+1
    #    examples, m being the number of samples in the train.csv data file,
    #    5000000 in our case.
    #----------------------------------------------------------------------------
    print('Generate the full example dataset')

    start =  time.time()
    x, y, mean, scale = transform_dataset(sample_input_length, 1)
    end =  time.time()
    print('Data generation duration: {}'.format(end - start))
    print('X & y shapes: {0} {1}'.format(x.shape, y.shape))

    #----------------------------------------------------------------------------
    # 2) Identify all open channels values.
    #----------------------------------------------------------------------------
    print('Identify all open channels values')

    bin_counts = np.bincount(y.reshape((y.shape[0],)))
    print('Open channels bin counts: {}'.format(bin_counts))

    number_of_open_channels_values = bin_counts.shape[0]
    print('Open channels values from 0 to {}'.format(number_of_open_channels_values-1))

    #----------------------------------------------------------------------------
    # 3) Find in the whole original data the smallest bin counts for open channels.
    #    Use this bin counts as number of example per open channels values so that
    #    we have the same number of examples for each open channels values. We will
    #    call this number example_number_per_label.
    #----------------------------------------------------------------------------
    print('Identify smallest category of open channels values')

    min_bin_count_idx = np.argmin(bin_counts)
    min_bin_count = bin_counts[min_bin_count_idx]
    print('Smallest category is {0} with {1} elements'.format(min_bin_count_idx,min_bin_count))

    #----------------------------------------------------------------------------
    # 4) For each open channels value:
    #    a) Filter the data set to keep only examples of the selected open channels
    #       value.
    #    b) Select example_number_per_label examples in this filtered list,
    #       uniformally distributed in the list (random selection).
    #    c) Concatenate this selected list to a our new dataset.
    #----------------------------------------------------------------------------
    print('Resample data')
    start =  time.time()
    x_new = None
    y_new =None
    for open_channels_value in range(number_of_open_channels_values):

        # Get index of exemples where the open channels value is
        # equal to open_channels_value
        value_indexes = np.where(y == open_channels_value)

        # Get corresponding signal inputs
        input_values = x[value_indexes[0],:,:]

        # Get a random selection (uniform) of inputs
        index_uniform_selection = np.random.choice(input_values.shape[0], min_bin_count, replace=False)

        # Stack selected data
        x_tmp = input_values[index_uniform_selection,:,:]
        y_tmp = np.ones((min_bin_count,1)) * open_channels_value
        if x_new is None:
            x_new = x_tmp
            y_new = y_tmp
        else:
            x_new = np.vstack((x_new, x_tmp))
            y_new = np.vstack((y_new, y_tmp))

    y_new = y_new.astype('int64')
    end =  time.time()
    print('Data resample duration: {}'.format(end - start))

    print('New x & y shapes: {0} {1}'.format(x_new.shape, y_new.shape))

    bin_counts = np.bincount(y_new.reshape((y_new.shape[0],)))
    print('New open channels bin counts: {}'.format(bin_counts))

    #----------------------------------------------------------------------------
    # 5) Convert open channels values into one hot encoded values
    #----------------------------------------------------------------------------
    print('Convert open channels values into one hot encoded values')
    y_new_one_hot = tf.keras.utils.to_categorical(y_new, num_classes=11)

    #----------------------------------------------------------------------------
    # 6) Shuffle this new dataset
    #----------------------------------------------------------------------------
    data_new = np.c_[x_new.reshape(len(x_new), -1), y_new_one_hot.reshape(len(y_new_one_hot), -1)]
    np.random.shuffle(data_new)
    x_new = data_new[:, :x_new.size//len(x_new)].reshape(x_new.shape)
    y_new_one_hot = data_new[:, x_new.size//len(x_new):].reshape(y_new_one_hot.shape)

    #----------------------------------------------------------------------------
    # 7) Split the data set in t% training, v% validation, e% evaluation,
    #    by default t=70, v=21, e=9
    #----------------------------------------------------------------------------
    training_percentage = 70
    validation_percentage = 21
    
    training_ratio = training_percentage/100.0
    training_split_idx = int(training_ratio*x_new.shape[0])

    x_train, x_tmp = np.split(x_new, [training_split_idx])
    y_train, y_tmp = np.split(y_new_one_hot, [training_split_idx])

    validation_ratio = validation_percentage / (100.0-training_percentage)
    validation_split_idx = int(validation_ratio*x_tmp.shape[0])

    x_valid, x_eval = np.split(x_tmp, [validation_split_idx])
    y_valid, y_eval = np.split(y_tmp, [validation_split_idx])

    print('Train shapes x & y = {0} {1}'.format(x_train.shape, y_train.shape))
    print('Valid shapes x & y = {0} {1}'.format(x_valid.shape, y_valid.shape))
    print('Eval shapes x & y = {0} {1}'.format(x_eval.shape, y_eval.shape))
    
    return x_train, y_train, x_valid, y_valid, x_eval, y_eval, mean, scale

def compute_predictions(model, sample_input_length, mean=None, scale=None):
    #----------------------------------------------------------------------------
    # Read the test data from the original file
    #----------------------------------------------------------------------------
    print('Get test data')
    df_test = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv')

    #----------------------------------------------------------------------------
    # Get only the signal column (2nd column)
    #----------------------------------------------------------------------------
    data = df_test.values[:,:]
    data = data[:,1]

    #----------------------------------------------------------------------------
    # Prepare result array
    #----------------------------------------------------------------------------
    # Keep only first coulumn (time)
    data_result = df_test.values[:,:]
    data_result = data_result[:,0]
    data_result = data_result.reshape((data_result.shape[0],1))

    # Add a column for the open_channels
    data_result = np.insert(data_result, 1,  0, axis=1)

    #----------------------------------------------------------------------------
    # Standardize train data which is signal & open_channels
    #----------------------------------------------------------------------------
    print('Standardize test data')
    # Mean and scale are the ones computed during standardization of
    # training data.
    data_scaled = (data - mean[0]) / scale[0]

    #----------------------------------------------------------------------------
    # Process each test data
    #----------------------------------------------------------------------------
    print('Make predictions')

    n = sample_input_length
    num_samples = data_scaled.shape[0]
    num_steps = num_samples-n+1

    print('Number of samples: {0}'.format(num_samples))
    print('Number of steps: {0}'.format(num_steps))

    x = np.zeros((num_steps, n, 1))
    for i in range(num_steps):
        x[i,:,:] = data_scaled[i:i+n].reshape((1, n, 1))

    # Make prediction
    print('Predict x {0}'.format(x.shape))
    y_one_hot = model.predict(x, verbose=0)
    print('y_one_hot.shape={0}'.format(y_one_hot.shape))

    # Revert one-hot vector
    y = np.argmax(y_one_hot, axis=1)
    print('y.shape={0}'.format(y.shape))

    # Save prediction
    data_result[n-1:,1] = y

    return data_result

def save_submission(path, data_result):

    submission_data_set_columns=['time', 'open_channels']

    df_submission = pd.DataFrame(data_result, columns=submission_data_set_columns)
    df_submission.open_channels = df_submission.open_channels.astype('int64')

    filename = '{0}/submission.csv'.format(path)

    df_submission.to_csv(filename, sep=',', mode='a', index=False, header=True, float_format='%.4f')

    print('Results saved into file {0}'.format(filename))

#-----------------------------------------------------
# Main
#-----------------------------------------------------
print('== Experience Start ==============================================')

#----------------------------------------------------------------------------
# Process script arguments
#----------------------------------------------------------------------------
sample_input_length = 70
batch_size = 1000
epochs = 100
steps_per_epoch = None
validation_steps = None

print('Signal sample input number: {0}'.format(sample_input_length))
print('Batch size: {0}'.format(batch_size))
print('Epochs: {0}'.format(epochs))
print('Steps per epochs: {0}'.format(steps_per_epoch))
print('Validation steps: {0}'.format(validation_steps))

#----------------------------------------------------------------------------
# Initialize random seeds
#----------------------------------------------------------------------------
os.environ['PYTHONHASHSEED'] = str(1234)
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

#----------------------------------------------------------------------------
# Create the model
#----------------------------------------------------------------------------
print('Create the model')

# Use Tensorflow 2.0 and Tensorflow.Keras
# Importing libraries for the model
import tensorflow_addons as tfa
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Input

n_steps = sample_input_length
n_features = 1 # only signal
n_units = 128

# define model
inp = Input(shape = (n_steps, n_features))
x = LSTM(n_units, activation='tanh', return_sequences=True)(inp)
x = LSTM(n_units, activation='tanh', return_sequences=True)(x)
x = LSTM(n_units, activation='tanh', return_sequences=True)(x)
x = LSTM(n_units, activation='tanh', return_sequences=True)(x)
x = LSTM(n_units, activation='tanh', return_sequences=True)(x)
x = LSTM(n_units, activation='tanh')(x)
out1 = Dense(11, activation = 'softmax', name = 'out1')(x)

y = Conv1D(32, 3, activation = 'relu')(inp)
y = MaxPooling1D(pool_size = 2)(y)
y = Dropout(0.20)(y)
y = Conv1D(64, 3, activation = 'relu')(y)
y = MaxPooling1D(pool_size = 2)(y)
y = Dropout(0.20)(y)
y = Flatten()(y)
y = Dense(units = 256, activation = 'relu')(y)
y = Dense(units = 64, activation = 'relu')(y)
y = Dense(units = 64, activation = 'relu')(y)
y = Dense(units = 64, activation = 'relu')(y)
y = Dense(units = 64, activation = 'relu')(y)
out2 = Dense(11, activation='softmax')(y)

out = Multiply()([out1, out2])

model = tf.keras.models.Model(inputs = inp, outputs = out)

opt = tf.keras.optimizers.Adam(lr = 0.001, amsgrad=True)
opt = tfa.optimizers.SWA(opt)
model.compile(optimizer=opt,loss="categorical_crossentropy", metrics = ['accuracy'])

model.summary()

#----------------------------------------------------------------------------
# Generate Data Sets
#----------------------------------------------------------------------------
print('Generate data sets')

x_train, y_train, x_valid, y_valid, x_eval, y_eval, mean, scale = generate_datasets(sample_input_length)

print('Mean = {}'.format(mean))
print('Scale = {}'.format(scale))

#----------------------------------------------------------------------------
# Train the model
#----------------------------------------------------------------------------
print('Train the model')

callbackEarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience = 5
    )

# fit model
start =  time.time()
history = model.fit(
    x_train,
    y_train,
    shuffle=True,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(x_valid, y_valid),
    validation_steps=validation_steps,
    verbose=0,
    callbacks=[callbackEarlyStopping]
    )
end =  time.time()
model_fit_duration_in_seconds = end - start
print('Fit duration: {}'.format(model_fit_duration_in_seconds))

#----------------------------------------------------------------------------
# Display model performance over epochs
#----------------------------------------------------------------------------
#%matplotlib inline
print('Display model performance over epochs')
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15,5))
title = 'Loss and Accuracy'
fig.suptitle(title, fontsize=16)
ax[0].set_title('loss')
if history.history["loss"] is not None:
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
if history.history["val_loss"] is not None:
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[0].legend()
ax[1].set_title('acc')
if history.history["accuracy"] is not None:
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
if history.history["val_accuracy"] is not None:
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
ax[1].legend()
fig_filename = '{0}/Loss_Accuracy.png'.format(
    experience_results_path
    )
plt.savefig(fig_filename)
plt.close(fig)

#----------------------------------------------------------------------------
# Evaluate model
#----------------------------------------------------------------------------
model_evaluation_duration_in_seconds = None
print('Evaluate model')
start =  time.time()
score = model.evaluate(
    x_eval,
    y_eval,
    batch_size=batch_size,
    verbose=0
    )
end =  time.time()
model_evaluation_duration_in_seconds = end - start
print('Evaluation duration: {}'.format(model_evaluation_duration_in_seconds))
print('Score: {0} - Metrics: {1}'.format(score, model.metrics_names))

# Make prediction
print('Predict x {0}'.format(x_eval.shape))
y_pred_one_hot = model.predict(x_eval)

# Revert one-hot vector
y_pred = np.argmax(y_pred_one_hot, axis=1)
y_ground_truth = np.argmax(y_eval, axis=1)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(confusion_matrix(y_ground_truth,y_pred))
print(classification_report(y_ground_truth,y_pred))
#print('Accuracy score: ', accuracy_score(y_ground_truth,y_pred))

#----------------------------------------------------------------------------
# Compute predictions
#----------------------------------------------------------------------------
compute_prediction_duration_in_seconds = None
print('Compute predictions')
start =  time.time()
data_result = compute_predictions(
    model=model,
    sample_input_length=sample_input_length,
    mean=mean,
    scale=scale
    )
end =  time.time()
compute_prediction_duration_in_seconds = end - start
print('Prediction duration: {}'.format(compute_prediction_duration_in_seconds))

#----------------------------------------------------------------------------
# Save predictions
#----------------------------------------------------------------------------
print('Save predictions')
save_submission(
    path=experience_results_path,
    data_result=data_result
    )

print('== Experience End ==============================================')
