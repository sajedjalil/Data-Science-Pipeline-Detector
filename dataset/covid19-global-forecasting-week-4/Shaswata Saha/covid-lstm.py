import math
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LinearRegression,BayesianRidge, Lasso
from statistics import mean
from math import sqrt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Bidirectional
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sklearn

import datetime
import warnings
from tqdm import tqdm
from pathlib import Path
import time
from copy import deepcopy
import os

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
df_train['Province_State'] = df_train['Province_State'].fillna('')
df_train['Date_datetime'] = df_train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
df_train['Country_Province'] = df_train['Country_Region'] + '_' + df_train['Province_State']
df_train['Country_Province'] = df_train['Country_Province'].apply(lambda x:(x[:-1] if x.endswith('_') else x))
df_train = df_train.drop(columns=['Date','Province_State','Country_Region'])
df_train = df_train[df_train["Date_datetime"]<datetime.datetime.strptime('2020-04-12', '%Y-%m-%d')]
print(len(df_train['Country_Province'].unique()))

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_test['Province_State'] = df_test['Province_State'].fillna('')
df_test['Date_datetime'] = df_test['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
df_test['Country_Province'] = df_test['Country_Region'] + '_' + df_test['Province_State']
df_test['Country_Province'] = df_test['Country_Province'].apply(lambda x:(x[:-1] if x.endswith('_') else x))
df_test = df_test.drop(columns=['Date','Province_State','Country_Region'])
df_test.head()
print(len(df_test['Country_Province'].unique()))

num_steps = 7
lags = range(num_steps,0,-1)
lag_cols = []

days_in_sequence = 7

temporal_cols = ['ConfirmedCases_Day','Fatalities_Day']
output_cols = ['ConfirmedCases','Fatalities']
norm_cols = ['ConfirmedCases_Norm','Fatalities_Norm']

df_enriched = df_train.copy()
print("Df read")
print(df_enriched)

df_enriched_copy = df_enriched.copy()

## making day count ##

start_time = time.time()
with tqdm(total = len(df_enriched.index)) as pbar:
    for idx,row in df_enriched.iterrows():
        for col in output_cols:
            mask  = (df_enriched['Date_datetime'] == row['Date_datetime']) & (df_enriched['Country_Province'] == row['Country_Province'])
            mask_prev = (df_enriched['Date_datetime'] == (row['Date_datetime'] - pd.Timedelta(days=1))) & (df_enriched['Country_Province'] == row['Country_Province'])
            try:
              df_enriched.loc[mask,col+'_Day'] = df_enriched.loc[mask,col].values[0] - df_enriched.loc[mask_prev,col].values[0]
            except:
              df_enriched.loc[mask,col+'_Day'] = 0
        pbar.update(1)

print('Time spent for getting daily data is {} minutes'.format(round((time.time()-start_time)/60,1)))
print(df_enriched)

df_enriched_copy = df_enriched.copy()

## making test data ##

test_df = df_test.copy()
embed_cols = temporal_cols + output_cols
# print(embed_cols)
test_df = test_df.join(pd.DataFrame(columns = embed_cols))

test_overlap_mask = (test_df['Date_datetime'] <= df_enriched_copy['Date_datetime'].max())
train_overlap_mask = (df_enriched_copy['Date_datetime'] >= test_df['Date_datetime'].min())
test_df.loc[test_overlap_mask, embed_cols] = df_enriched_copy.loc[train_overlap_mask, embed_cols].values

pred_dt_range = pd.date_range(start = df_enriched_copy['Date_datetime'].max() + pd.Timedelta(days=1), 
                              end = test_df['Date_datetime'].max(), freq = '1D')


query_mask = ((df_enriched['ConfirmedCases']>0) & 
              (df_enriched['Fatalities']>=0))
df_enriched = df_enriched[query_mask]

## normalising data

confprop_day = [df_enriched['ConfirmedCases_Day'].min(),
            df_enriched['ConfirmedCases_Day'].max()-df_enriched['ConfirmedCases_Day'].min()]
fatprop_day = [df_enriched['Fatalities_Day'].min(),
           df_enriched['Fatalities_Day'].max()-df_enriched['Fatalities_Day'].min()]

df_enriched['ConfirmedCases_Norm'] = ((df_enriched['ConfirmedCases']-confprop_day[0])*1.0)/confprop_day[1]
df_enriched['Fatalities_Norm'] = ((df_enriched['Fatalities']-fatprop_day[0])*1.0)/fatprop_day[1]

test_df['ConfirmedCases_Norm'] = ((test_df['ConfirmedCases_Day']-confprop_day[0])*1.0)/confprop_day[1]
test_df['Fatalities_Norm'] = ((test_df['Fatalities_Day']-fatprop_day[0])*1.0)/fatprop_day[1]
print(test_df)

print(df_enriched['ConfirmedCases_Norm'].isna().sum(),df_enriched['Fatalities_Norm'].isna().sum())

print("Df normalised")
print(df_enriched)

## making temporal cols ##

start_time = time.time()
temp_final = []

with tqdm(total = len(df_enriched.index)) as pbar:
  for idx,row in df_enriched.iterrows():
      i = row['Country_Province']
      d = row['Date_datetime']

      temp_arr = []       
      for lag in lags:
          day_arr = []
          mask_org = (df_enriched['Date_datetime'] == (d - pd.Timedelta(days=lag))) & (df_enriched['Country_Province'] == i)
          
          for temp_col in norm_cols:
              try:
                  day_arr = day_arr + [df_enriched.loc[mask_org, temp_col].values[0]]
              except:
                  if temp_col == 'ConfirmedCases_Norm':
                    part1 = (-1.0*confprop_day[0])/confprop_day[1]
                    day_arr = day_arr + [part1]
                  else:
                    part1 = (-1.0*fatprop_day[0])/fatprop_day[1]
                    day_arr = day_arr + [part1]
          
          temp_arr = temp_arr + [day_arr]
      
      # print(temp_arr)
      temp_final = temp_final + [temp_arr]

      pbar.update(1)

df_enriched['Temporal_cols'] = temp_final
print('Time spent for building temporal cols is {} minutes'.format(round((time.time()-start_time)/60,1)))

print("Df temporal")
print(df_enriched)

training_percentage = 0.9
sequence_length = days_in_sequence

trend_df = df_enriched.copy()
training_item_count = int(len(trend_df)*training_percentage)
validation_item_count = len(trend_df)-training_item_count
training_df = trend_df[:training_item_count]
training_df = sklearn.utils.shuffle(training_df)

validation_df = trend_df[training_item_count:]
validation_df = sklearn.utils.shuffle(validation_df)

validation_item_count = len(validation_df)

X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in training_df["Temporal_cols"].values]),(training_item_count,len(temporal_cols),sequence_length)),(0,2,1) )).astype(np.float32)
Y_cases_train = np.asarray([np.asarray(x) for x in training_df["ConfirmedCases_Norm"]]).astype(np.float32)
Y_fatalities_train = np.asarray([np.asarray(x) for x in training_df["Fatalities_Norm"]]).astype(np.float32)

X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in validation_df["Temporal_cols"]]),(validation_item_count,len(temporal_cols),sequence_length)),(0,2,1)) ).astype(np.float32)
Y_cases_test = np.asarray([np.asarray(x) for x in validation_df["ConfirmedCases_Norm"]]).astype(np.float32)
Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_df["Fatalities_Norm"]]).astype(np.float32)

## building the model ##

temporal_input_layer = Input(shape=(days_in_sequence,len(temporal_cols)))
main_rnn_layer = layers.LSTM(32, return_sequences=True, recurrent_dropout=0.2)(temporal_input_layer)

#cases output branch
# rnn_c = layers.LSTM(16)(main_rnn_layer)
# dense_c_1 = layers.Dense(64)(rnn_c)
# dropout_c_1 = layers.Dropout(0.3)(dense_c_1)
# dense_c_2 = layers.Dense(16)(dropout_c_1)
# dropout_c_2 = layers.Dropout(0.3)(dense_c_2)
cases = layers.Dense(1, activation='relu',name="cases")(main_rnn_layer)

#fatality output branch
# rnn_f = layers.LSTM(16)(main_rnn_layer)
# dense_f_1 = layers.Dense(64)(rnn_f)
# dropout_f_1 = layers.Dropout(0.3)(dense_f_1)
# dense_f_2 = layers.Dense(16)(dropout_f_1)
# dropout_f_2 = layers.Dropout(0.3)(dense_f_2)
fatalities = layers.Dense(1, activation='relu', name="fatalities")(main_rnn_layer)

model = Model(temporal_input_layer, [cases,fatalities])
model.summary()


callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, 
                                   factor=0.6),
             EarlyStopping(monitor='val_loss',patience=20,verbose=1),
            ModelCheckpoint(filepath='best_model.h5', 
                        monitor='val_loss', save_best_only=True)]

opti = optimizers.Adam(lr=1e-3)
model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),
                    tf.keras.losses.MeanSquaredLogarithmicError()], 
              optimizer=opti)

## training model ##

history = model.fit(x=X_temporal_train, 
                    y=[Y_cases_train, Y_fatalities_train], 
                    epochs = 50, 
                    batch_size = 8, 
                    validation_data=(X_temporal_test,  
                                      [Y_cases_test, Y_fatalities_test]),
                    callbacks = callbacks)

model.load_weights("best_model.h5")

## predicting future values
for country in df_enriched.Country_Province.unique():
    for date in pred_dt_range:
      mask = (test_df['Date_datetime'] == date) & (test_df['Country_Province'] == country)

      temp_input = []
      for lag in lags:
          mask_org = (test_df['Date_datetime'] == (date - pd.Timedelta(days=lag))) & (test_df['Country_Province'] == country)
          day_input = []

          for col in norm_cols:
            try:
                day_input.append(test_df.loc[mask_org, col].values[0])
            except:
                  if temp_col == 'ConfirmedCases_Norm':
                    part1 = (-1.0*confprop_day[0])/confprop_day[1]
                    day_input.append(part1)
                  else:
                    part1 = (-1.0*fatprop_day[0])/fatprop_day[1]
                    day_input.append(part1)

          temp_input = temp_input + [day_input]

      temp_input = np.array([temp_input])
    
      result = model.predict(temp_input)

      result[0] = np.nan_to_num(result[0])
      result[1] = np.nan_to_num(result[1])
        
      test_df.loc[mask,'ConfirmedCases_Norm'] = result[0][0][0]
      test_df.loc[mask,'Fatalities_Norm'] = result[1][0][0]
        
      confday = result[0][0][0]*confprop_day[1] + confprop_day[0]      
      fatday = result[1][0][0]*fatprop_day[1] + fatprop_day[0]

      mask_prev = (test_df['Date_datetime'] == (date - pd.Timedelta(days=1))) & (test_df['Country_Province'] == country)
      test_df.loc[mask,'ConfirmedCases'] = test_df.loc[mask_prev,'ConfirmedCases'].values[0] + confday
      test_df.loc[mask,'Fatalities'] = test_df.loc[mask_prev,'Fatalities'].values[0] + fatday

      print(country,date,
            confday,fatday,
            test_df.loc[mask,'ConfirmedCases'].values,
            test_df.loc[mask,'Fatalities'].values)

      test_df.loc[mask,'ConfirmedCases_Day'] = confday
      test_df.loc[mask,'Fatalities_Day'] = fatday
        

test_df = test_df[['ForecastId','ConfirmedCases','Fatalities']]

sub = test_df.copy()
sub = sub.sort_values('ForecastId')
sub.to_csv('submission.csv',index=False)