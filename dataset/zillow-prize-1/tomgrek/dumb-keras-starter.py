import numpy as np
import pandas as pd
from sklearn import preprocessing

import gc

from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Dense, Reshape, Lambda, Dropout

# Swap out for the small models to check it's working
train = pd.read_csv('../input/train_2016.csv', nrows=13200) # pd.read_csv('./smalltrain.csv')
prop = pd.read_csv('../input/properties_2016.csv', nrows=13200) # pd.read_csv('./smallprop.csv', header=0)
sample = pd.read_csv('../input/sample_submission.csv')

prop['propertycountylandusecode'] = prop['propertycountylandusecode'].apply(lambda x: str(x))
encoder = preprocessing.LabelEncoder()
encoder.fit(prop['propertycountylandusecode'])
prop['propertycountylandusecode'] = encoder.transform(prop['propertycountylandusecode'])

prop['propertyzoningdesc'] = prop['propertyzoningdesc'].apply(lambda x: str(x))
encoder2 = preprocessing.LabelEncoder()
encoder2.fit(prop['propertyzoningdesc'])
prop['propertyzoningdesc'] = encoder2.transform(prop['propertyzoningdesc'])

# Discard all non-numeric data
prop = prop.select_dtypes([np.number])
train = train.select_dtypes([np.number])
sample = sample.select_dtypes([np.number])

gc.collect()

# To check the script works, create a small dataset first
# prop[:100].to_csv('smallprop.csv', index=False, float_format='%.4f')
# train[:100].to_csv('smalltrain.csv', index=False, float_format='%.4f')

# The land parcel id is useless info for prediction
x_train = prop.drop(['parcelid'], axis=1)

gc.collect()

# Store for prediction phase
train_columns = x_train.columns

# Rows are not ordered, so join on parcelid
temp = pd.merge(left=train, right=prop, on=('parcelid'), how='outer')

# Casually fill in missing data with junk
temp = temp.fillna(0)

# Convert to numpy array for Keras
x_train = temp.drop(['parcelid', 'logerror'], axis=1).values
y_train = temp['logerror'].values

gc.collect()

scaler = preprocessing.StandardScaler()
# x_train.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
x_train = scaler.fit_transform(x_train)

# Normalize (across the whole dataframe cos we dun care)
mean_x = x_train.mean().astype(np.float32)
std_x = x_train.std().astype(np.float32)

mean_y = y_train.mean().astype(np.float32)
std_y = y_train.std().astype(np.float32)

def normalize(x):
    return (x-mean_x)/std_x

def normalize_y(y):
    return (y-mean_y)/std_y

def de_normalize_y(y):
    return (y*std_y) + mean_y

y_train = normalize(y_train)

# Build a simple model
model = Sequential([
    #Lambda(normalize,input_shape=(52, )),
	Dense(60,input_shape=(54, )),
    BatchNormalization(),
    Dropout(0.08),
	Dense(160, activation='relu'),
	BatchNormalization(),
    Dropout(0.38),
    Dense(20, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
model.compile(loss='mean_squared_error', optimizer='adam')

# 76 batch size works on my 4GB GTX960m but lower for kaggle
model.fit(x_train, y_train, batch_size=24, epochs=15)

# Prepare the submission data
sample['parcelid'] = sample['ParcelId']
del sample['ParcelId']

df_test = pd.merge(sample, prop, on='parcelid', how='left')
df_test = df_test.fillna(0)

x_test = df_test[train_columns]

p_test = model.predict(x_test.values)
p_test = de_normalize_y(p_test)

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('keras_starter.csv', index=False, float_format='%.5f')
