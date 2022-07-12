# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

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

#####Handling Missing Values#####     

for i in range(len(train.columns)):
    train.iloc[:,i] = (train.iloc[:,i]).fillna(0)

for i in range(len(test.columns)):
    test.iloc[:,i] = (test.iloc[:,i]).fillna(0)    
    
#####Encoding the Categorical Variables#####

lbl = LabelEncoder()

for c in train.columns:
    if train[c].dtype == 'object':
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))

for c in test.columns:
    if test[c].dtype == 'object':
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))     
        
print("Done with the Encoding")        
####Normalizing the values####

mmScale = MinMaxScaler()

n = train.shape[1]


x_train = mmScale.fit_transform(train)
x_test = mmScale.fit_transform(test)

#####Artificial Neural Networks Implementation#####
print("Starting Neural Network")

model_n = Sequential()

#Want to use an expotential linear unit instead of the usual relu
model_n.add( Dense( n, activation='relu', input_shape=(n,) ) )
model_n.add( Dense( int(0.5*n), activation='relu' ) )
model_n.add(Dense(1, activation='linear'))
model_n.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
        

model_n.fit(x_train, y, epochs=5, batch_size=10)

predict_test_NN = model_n.predict(x_test)

#####Writing the Results######

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = predict_test_NN

print('Writing csv ...')
sub.to_csv('NN_submission.csv', index=False, float_format='%.4f')
        
        
        