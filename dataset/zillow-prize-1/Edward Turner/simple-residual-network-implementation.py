# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from keras.layers.noise import GaussianNoise
from keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.

#########Loading the Files################

props = pd.read_csv('../input/properties_2016.csv')
train_df = pd.read_csv("../input/train_2016_v2.csv")
test_df = pd.read_csv("../input/sample_submission.csv")
test_df = test_df.rename(columns={'ParcelId': 'parcelid'})

######Merge Operation#####

train = train_df.merge(props, how = 'left', on = 'parcelid')
test = test_df.merge(props, on='parcelid', how='left')

for c in train.columns:
    if train[c].dtype == 'float64':
        train[c] = train[c].values.astype('float32')
        

print("Done with Merged Operation")        
#####Removing Outliers, Total Features#####        
train=train[ train.logerror > -0.4 ]
train=train[ train.logerror < 0.4 ]



do_not_include = ['parcelid', 'logerror', 'transactiondate']

feature_names = [f for f in train.columns if f not in do_not_include]

print("We have %i features."% len(feature_names))

#####Getting the same number of columns for Train, Test######

y = train['logerror'].values

train = train[feature_names].copy()
test = test[feature_names].copy()

#####Encoding the Categorical Variables#####
#####Handling Missing Values#####    
for col in train.columns:
    if train.loc[:,col].dtype == 'object':
        train.loc[:, col] = train.loc[:, col].fillna( 'NA' )
        test.loc[:, col] = test.loc[:, col].fillna( 'NA' )
        
        col_val_dict = train.loc[:, col].append(test.loc[:, col]).value_counts()
        
        n = int( 0.1*len( col_val_dict.keys() ) )
        
        n = min( n, 10 )
        
        values_keep = list( col_val_dict[n:].keys() )
        
        train.loc[ train.loc[:, col].isin(values_keep), col] = 'other'
        
    else:
        mean_ = train.loc[:,col].mean()
        
        train.loc[:, col] = train.loc[:, col].fillna( mean_ )
        test.loc[:, col] = test.loc[:, col].fillna( mean_ )  

train = pd.get_dummies( train )
test = pd.get_dummies( test )

cols_common = list( set(train.columns) & set(test.columns) )

train = train.loc[:, cols_common]
test = test.loc[:, cols_common]

print("Done with the Encoding")        
####Normalizing the values####

n = train.shape[1]

x_train = train.values
x_test =test.values

x_train = x_train[:,:,np.newaxis]
x_test = x_test[:,:,np.newaxis]

#####Artificial Neural Networks Implementation#####
print("Starting Neural Network")

input_ = Input( shape = (n,1) )
x = BatchNormalization()(input_)

#data augmentation
x = GaussianNoise(1.0)(x)

filt_size = 32

#residual node
left = Conv1D(filt_size, 3, padding = 'same', activation = 'elu', 
                      kernel_initializer = 'he_normal')(x)
left = BatchNormalization()(left)

#concatenation instead of addition
x =  concatenate([x, left])

for i in range(5):
    left = Conv1D(filt_size, 3, padding = 'same', activation = 'elu', 
                      kernel_initializer = 'he_normal')(left)
    left = BatchNormalization()(left)

    x =  concatenate([x, left])

#fc-layer
x = Flatten()(x)

m = int( 0.5*(n + 1) )

x = Dense(m, activation = 'elu', kernel_initializer = 'he_normal')(x)
x = BatchNormalization()(x)
x = GaussianNoise(1.0)(x)

x = Dense(m, activation = 'elu', kernel_initializer = 'he_normal' )(x)
x = BatchNormalization()(x)
x = GaussianNoise(1.0)(x)

output = Dense(1, activation = 'linear' )(x)

model = Model( inputs = input_, outputs = output)

model.compile(loss = 'mse', optimizer = Adadelta(), metrics=['mae'])
        
model.fit(x_train, y, epochs=5, batch_size=10, verbose = 2, callbacks = [EarlyStopping(monitor='val_loss', patience=1)], validation_split = 0.25)

predict_test_NN = model.predict(x_test)

#####Writing the Results######

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = predict_test_NN

print('Writing csv ...')
sub.to_csv('NN_submission.csv', index=False, float_format='%.4f')