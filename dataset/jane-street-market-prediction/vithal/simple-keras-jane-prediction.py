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



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import janestreet
from keras.layers import Dense
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa

df = pd.read_csv('../input/jane-street-market-prediction/train.csv')
                

#print('Read csv')

train = df[df['weight']>0]
train = train.fillna(0.5)

y_train = ((train['resp'].values) > 0).astype(int)

x_train = train.drop(['resp','resp_1','resp_2','resp_3','resp_4','resp_4','ts_id','date'],axis = 1,inplace = False)
#print(x_train.shape)
#print(x_train.head())

input_shape =x_train.shape[1]

#print('Fitting model')


#print('Creating model')

model = Sequential()

model.add(Dense(150, activation='swish', input_shape=(input_shape,)))

# Add the second hidden layer
model.add(Dense(150, activation='swish',input_shape=(150, )))

model.add(Dense(1,activation='sigmoid',input_shape=(150, )))

model.compile(
optimizer = tfa.optimizers.RectifiedAdam(learning_rate=1e-3),
metrics = tf.keras.metrics.AUC(name="AUC"),
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=1e-2),
)


# you should split it on batches!!!!
model.fit(x_train.values,y_train,epochs=50, batch_size=4096,shuffle=False,verbose=2)

env = janestreet.make_env()
iter_test = env.iter_test()
for (test, sample_prediction) in tqdm(iter_test):
    test = test.fillna(0.5)
    
    if test['weight'].item() > 0:
        
        X_test = test.drop(['date'],axis = 1).values
        #print(X_test.shape)
        pred = (model.predict(X_test))
        pred = np.median(pred)
        sample_prediction['action'] = np.where(pred >= 0.5, 1, 0).astype(int)
        

    else:
         sample_prediction['action'] = 0
    
    env.predict(sample_prediction)






