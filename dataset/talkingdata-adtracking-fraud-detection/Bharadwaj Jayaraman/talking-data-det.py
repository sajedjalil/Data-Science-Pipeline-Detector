# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
class Encoder:
    @staticmethod
    def encode_data(x_train, numerical_columns, d, enc_create=True):
        print("Numerical columns"+str(numerical_columns))
        if numerical_columns is not None:
            cat_cols = x_train.drop(numerical_columns, axis=1).fillna('NA')
        else:
            cat_cols = x_train

        label_enc_map = {}

        if enc_create:
            enc = ()
            cat_data = cat_cols.apply(lambda x: d[x.name].fit_transform(x))
        else:
            cat_data = cat_cols.apply(lambda x: d[x.name].transform(x))

        if numerical_columns is not None:
            df = pd.concat([x_train[numerical_columns].fillna(0), cat_data], axis=1)
        else:
            df = cat_data
        return df.values
        
        
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pandas as pd,re, pickle, scipy
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import *
import numpy as np
from sklearn.feature_selection import *
from sklearn.feature_extraction import *
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.kernel_ridge import *
from sklearn import tree
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.model_selection import *
from sklearn.linear_model import *
from collections import defaultdict

from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
# from url_utils import URLUtils
from keras.layers import *
from keras.models import  *
from keras.utils import to_categorical
from keras.metrics import *
# from encoder import Encoder
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))



def convert_val(val):
    return (val+1)/2

def parse_dates(date):
    parts=date.split(" ")
    date_parts = parts[0].split("-")
    time_parts = parts[1].split(':')
    new_date =date_parts[2]+date_parts[1]+date_parts[0]+time_parts[0]+time_parts[1]
    return int(new_date)

d = defaultdict(LE)
mms=MinMaxScaler()

sample_path='../input/train_sample.csv'
path='../input/train.csv'

df = pd.read_csv(sample_path)
# df = df.loc[df['is_attributed']==1]
df['click_time']= df['click_time'].map(parse_dates)
data_cols = list(df.columns)
data_cols.remove('attributed_time')
data_cols.remove('is_attributed')
smote = SMOTE(ratio='minority',k_neighbors=15, n_jobs=-1)
label_cols = u'is_attributed'
print("Original Class dist:\n",df[label_cols].value_counts())
print('Data generated')
y = df[label_cols].values
X = df[data_cols].values# Any results you write to the current directory are saved as output.





X = mms.fit_transform(X)
x_train, y_train = smote.fit_sample(X, y)
x_train.shape,y_train.shape


df2=pd.DataFrame()

df2[label_cols]=y_train
print("After SMOTE Class dist:\n",df2[label_cols].value_counts())


model = Sequential()
num_columns=x_train.shape[1]
model.add(Dense(num_columns, input_shape=(num_columns,), kernel_initializer='normal', activation='relu'))
model.add(Dense(5000, kernel_initializer='normal', activation='relu'))
# model.add(Dense(100, kernel_initializer='normal', activation='relu'))

model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',mae])
model.fit(x_train,to_categorical(y_train), verbose=3, validation_split=0.2, shuffle=True, epochs=7)
model.evaluate(x_train,to_categorical(y_train))
model.save('model_5000.h5')
model = load_model('model_5000.h5')
print('Loaded model')
df = pd.read_csv('../input/test.csv')
df['click_time']= df['click_time'].map(parse_dates)
print('Test data generated')



x_test = mms.transform(df[data_cols].values)
y1_test=model.predict(x_test)

y_test = np.argmax(y1_test, axis=1)
print(y1_test)
print(y_test)

df['is_attributed'] = y_test
results = df[['click_id', 'is_attributed']]
results.to_csv('submission.csv', index=None)
print('done')

