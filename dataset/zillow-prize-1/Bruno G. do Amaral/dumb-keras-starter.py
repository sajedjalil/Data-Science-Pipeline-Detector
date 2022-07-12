import numpy as np
import pandas as pd

import gc

from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Dense, Reshape, Lambda, Dropout

# Number of rows to load (None for all dataset)
nrows = None

# Swap out for the small models to check it's working
train = pd.read_csv('../input/train_2016.csv', nrows=nrows) # pd.read_csv('./smalltrain.csv')
prop = pd.read_csv('../input/properties_2016.csv', nrows=nrows) # pd.read_csv('./smallprop.csv', header=0)

# Discard all non-numeric data
prop = prop.select_dtypes([np.number])
train = train.select_dtypes([np.number])

gc.collect()

# To check the script works, create a small dataset first
# prop[:100].to_csv('smallprop.csv', index=False, float_format='%.4f')
# train[:100].to_csv('smalltrain.csv', index=False, float_format='%.4f')

# The land parcel id is useless info for prediction

# Rows are not ordered, so join on parcelid
temp = pd.merge(left=train, right=prop, on=('parcelid'), how='left')

# Convert to numpy array for Keras
y_train = temp['logerror'].values.astype(np.float32)

# Casually fill in missing data with junk
fill_mean = temp.mean()
temp = temp.fillna(fill_mean).fillna(0).drop(['parcelid', 'logerror'], axis=1)
train_columns = temp.columns
x_train = temp.values

del temp, train
gc.collect()

# Normalize (across the whole dataframe cos we dun care)
mean_x = x_train.mean(axis=0).astype(np.float32)
std_x = x_train.std(axis=0).astype(np.float32)

# Remove zero-std features
std_x_zero = (std_x == 0)
mean_x = mean_x[~std_x_zero]
std_x = std_x[~std_x_zero]

y_min, y_max = np.percentile(y_train, [1, 99])

def normalize(x):
    x = x[:,~std_x_zero]
    x -= mean_x
    x /= std_x
    return x
    
def normalize_y(y):
    return ((y-y_min) / (y_max-y_min))

def denormalize_y(y):
    return y * (y_max-y_min) + y_min

# Build a simple model
model = Sequential([
	Dense(100, input_shape=(mean_x.shape[0], ), activation='relu'),
    Dense(50, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(loss='mse', optimizer='sgd')

x_train = normalize(x_train)

print("x_train shape", x_train.shape)
# 76 batch size works on my 4GB GTX960m but lower for kaggle
model.fit(x_train, normalize_y(y_train), batch_size=32, epochs=200, verbose=2, validation_split=0.2)

del x_train, y_train
gc.collect()

# Prepare the submission data
sample = pd.read_csv('../input/sample_submission.csv', nrows=nrows)
print("sample shape:", sample.shape)
sample = sample.select_dtypes([np.number])
sample['parcelid'] = sample['ParcelId']
del sample['ParcelId']

df_test = pd.merge(sample, prop, on='parcelid', how='left')
df_test = df_test.fillna(fill_mean)

x_test = df_test[train_columns].values
x_test = normalize(x_test)

p_test = denormalize_y(model.predict(x_test))

sub = pd.read_csv('../input/sample_submission.csv', nrows=nrows)
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('keras_starter.csv', index=False, float_format='%.5f')
