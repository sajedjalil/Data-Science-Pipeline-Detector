import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df_train = pd.read_csv('../input/train_V2.csv')
df_test = pd.read_csv('../input/test_V2.csv')

df_train.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
df_test.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

df_train = pd.get_dummies(df_train,columns=['matchType'], drop_first=True)
df_test = pd.get_dummies(df_test,columns=['matchType'], drop_first=True)

columns_with_nan = df_train.columns[df_train.isna().any()].tolist()
for column in columns_with_nan:
    mean_value = df_train[column].mean()
    df_train[column] = df_train[column].fillna(mean_value)

columns_with_nan = df_test.columns[df_test.isna().any()].tolist()
for column in columns_with_nan:
    mean_value = df_test[column].mean()
    df_test[column] = df_test[column].fillna(mean_value)

y = df_train.loc[:,['winPlacePerc']].values
df_train_dummy = df_train.pop('winPlacePerc')

x = df_train.loc[:,:].values
df_train['winPlacePerc'] = df_train_dummy

x_test = df_test.loc[:,:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
x_test = sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

NN_model = Sequential()
NN_model.add(Dense(20, init='uniform',input_dim = x.shape[1], activation='relu'))
NN_model.add(Dense(20, init='uniform',activation='relu'))
NN_model.add(Dense(20, init='uniform',activation='relu'))
NN_model.add(Dense(1, init='uniform',activation='linear'))
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.fit(x, y, epochs=5, batch_size=64, validation_split = 0.2)

predictions = NN_model.predict(x_test)
refined_predictions = []
for i in predictions:
    refined_predictions.append(i[0])
    
my_submission = pd.DataFrame({'Id':pd.read_csv('../input/test_V2.csv').Id,'winPlacePerc':refined_predictions})
my_submission.to_csv('sample_submission.csv', index=False)

