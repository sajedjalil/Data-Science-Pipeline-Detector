# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import kerastuner as kt
from keras import layers

from kerastuner.tuners import RandomSearch, Hyperband
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

import numpy as np
import pandas as pd
from random import choices
from tqdm import tqdm

import janestreet

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
#train = train.query('date > 85').reset_index(drop = True) 
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
train.fillna(train.mean(),inplace=True)
train = train.query('weight > 0').reset_index(drop = True)
#train['action'] = (train['resp'] > 0).astype('int')
train['action'] =  (  (train['resp_1'] > 0.00001 ) & (train['resp_2'] > 0.00001 ) & (train['resp_3'] > 0.00001 ) & (train['resp_4'] > 0.00001 ) &  (train['resp'] * train['weight'] > 0.001 )).astype('int')

def lower_sample_data(df, percent=1):
    data1 = df[df['action'] == 1]
    data0 = df[df['action'] == 0]
    index = np.random.randint(len(data0), size=percent * (len(df) - len(data0))) #randomly pick the sample with action=0
    lower_data0 = data0.iloc[list(index)]
    return(pd.concat([lower_data0, data1]))

train = lower_sample_data(train)

features = [c for c in train.columns if 'feature' in c]

X = train[['date'] + features]
y = train[['date','action']] 

X_train = X.loc[(X['date']>400) & (X['date']<=480)][features].values
X_validation = X.loc[X['date']>490][features].values
#X_validation = X.loc[(X['date']>480) & (X['date']<=485)][features].values
#X_test = X.loc[X['date']>490][features].values

y_train = y.loc[(y['date']>400) & (y['date']<=480)]['action'].values
y_validation = y[y['date']>490]['action'].values
#y_validation = y.loc[(y['date']>480) & (y['date']<=485)]['action'].values
#y_test = y[y['date']>490]['action'].values

model = Sequential()
model.add(Dense(320,input_shape = (X_train.shape[-1],),activation='relu'))
model.add(layers.Dropout(rate=0.236))
model.add(Dense(128,activation='relu'))
model.add(layers.Dropout(rate=0.231))
model.add(Dense(1312,activation='relu'))
model.add(layers.Dropout(rate=0.418))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

opt = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train,y_train,
                    validation_data=(X_validation,y_validation),
                    batch_size=2048, epochs=50,
                   callbacks=[EarlyStopping('val_loss',patience=5,restore_best_weights=True)])



env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

for (test_df, sample_prediction_df) in iter_test:
    test_df.fillna(test_df.mean(),inplace=True)
    X = test_df[features].values
    action = model(X,training=False).numpy()[0,0] > 0.5
    sample_prediction_df.action = int(action)
    env.predict(sample_prediction_df)

