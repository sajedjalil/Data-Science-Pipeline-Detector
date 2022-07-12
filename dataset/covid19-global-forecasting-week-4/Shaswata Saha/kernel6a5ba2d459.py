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

df_lockdown = pd.read_csv('/kaggle/input/covid19/lockdown.csv')
df_lockdown['Province'] = df_lockdown['Province'].fillna('')
df_lockdown['Date'] = df_lockdown['Date'].fillna('08/05/2020')
df_lockdown['Date_datetime'] = df_lockdown['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%d/%m/%Y')))
df_lockdown['Country_Province'] = df_lockdown['Country/Region'] + '_' + df_lockdown['Province']
df_lockdown['Country_Province'] = df_lockdown['Country_Province'].apply(lambda x:(x[:-1] if x.endswith('_') else x))
df_lockdown = df_lockdown.drop(columns=['Date','Province','Country/Region','Type','Reference'])
print(df_lockdown)

df_enriched = pd.read_csv('/kaggle/input/covid-19-enriched-dataset-week-2/enriched_covid_19_week_2.csv')
df_enriched['Province_State'] = df_enriched['Province_State'].fillna('')
df_enriched['Date_datetime'] = df_enriched['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
df_enriched['Country_Province'] = df_enriched['Country_Region']
df_enriched['Country_Province'] = df_enriched['Country_Province'].apply(lambda x:(x[:-1] if x.endswith('_') else x))
df_enriched = df_enriched.drop(columns=['Date','Province_State','Country_Region'])
df_enriched.head()

countryTrain = df_train['Country_Province'].unique()
countryTest = df_test['Country_Province'].unique()
countryIntersect = set(countryTrain) & set(countryTest)
countryLockdown = df_lockdown['Country_Province'].unique()

lockdownIntersect = set(countryLockdown) & countryIntersect
print(set(lockdownIntersect).difference(countryIntersect))

countryEnriched = df_enriched['Country_Province'].unique()
print(set(countryEnriched).difference(countryIntersect))

enrichedCountries = sorted(lockdownIntersect & set(countryEnriched))
lockdownCountries = sorted(lockdownIntersect.difference(set(countryEnriched)))
print('Model 1 : ',enrichedCountries)
print('Model 2 : ',lockdownCountries)
print(len(lockdownCountries),len(enrichedCountries))

lockdown_dates = {}

for _,i in df_lockdown.iterrows():
  lockdown_dates[i['Country_Province']] = i['Date_datetime']

print(lockdown_dates)

num_steps = 7
lags = range(num_steps,0,-1)
lag_cols = []

days_in_sequence = 7

temporal_cols = ['ConfirmedCases_Day_Norm','Fatalities_Day_Norm','restrictions','quarantine','schools']
day_cols = ['ConfirmedCases_Day','Fatalities_Day']
output_cols = ['ConfirmedCases','Fatalities']

trend_list = []
demographic_cols = ['age_0-4','age_5-9',
                   'age_10-14','age_15-19',
                   'age_20-24','age_25-29',
                   'age_30-34','age_35-39',
                   'age_40-44','age_45-49',
                   'age_50-54','age_55-59',
                   'age_60-64','age_65-69',
                   'age_70-74','age_75-79',
                   'age_80-84','age_85-89',
                   'age_90-94','age_95-99',
                   'age_100+']
demographic_norm = ['total_pop_Norm','smokers_perc_Norm','density_Norm',
                    'urbanpop_Norm','hospibed_Norm','lung_Norm',
                    'femalelung_Norm','malelung_Norm']

demo_vals = {}

norm_cols1 = ['ConfirmedCases_Day','Fatalities_Day']
norm_cols2 = ['total_pop','smokers_perc','density','urbanpop',
                    'hospibed','lung','femalelung','malelung']

norm_cols = []
for col in norm_cols1:
  norm_cols.append(col+'_Norm')

for col in norm_cols2:
  norm_cols.append(col+'_Norm')

## inserting missing rows ##
start_time = time.time()
with tqdm(total = len(enrichedCountries)) as pbar:
  for country in enrichedCountries:
      country_enriched = df_enriched[df_enriched['Country_Province']==country]
      
      country_min_date = country_enriched['Date_datetime'].min()
      country_max_date = country_enriched['Date_datetime'].max()
      train_max_date = df_train[df_train['Country_Province']==country]['Date_datetime'].max()
      
      latest_row = country_enriched[country_enriched['Date_datetime']==country_max_date]
    
      for date in pd.date_range(start=country_max_date+datetime.timedelta(days=1),
                                end=train_max_date):
          mask = (df_train['Date_datetime'] == date) & (df_train['Country_Province'] == country)   
          edit_row = latest_row.copy()
          edit_row['Date_datetime'] = date
          for cols in (output_cols):
              edit_row[cols] = df_train.loc[mask,cols].values
          df_enriched = df_enriched.append(edit_row)
      
      pbar.update(1)

df_enriched = df_enriched.reset_index(drop=True)
print('Time spent for inserting missing rows is {} minutes'.format(round((time.time()-start_time)/60,1)))

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

## normalizing data ##
df_enriched = df_enriched[df_enriched['ConfirmedCases']>0]
props = {}

for col in (norm_cols1 + norm_cols2):
  props[col+'_Norm'] = [df_enriched[col].min(),
                df_enriched[col].max()-df_enriched[col].min()]

  df_enriched[col+'_Norm'] = ((df_enriched[col] - 
                               props[col+'_Norm'][0])*1.0)/props[col+'_Norm'][1]
    
df_enriched.tail()

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
          
          for temp_col in temporal_cols:
              try:
                  day_arr = day_arr + [df_enriched.loc[mask_org, temp_col].values[0]]
              except:
                  day_arr = day_arr + [0]
          
          temp_arr = temp_arr + [day_arr]
      
      # print(temp_arr)
      temp_final = temp_final + [temp_arr]

      pbar.update(1)

df_enriched['Temporal_cols'] = temp_final
print('Time spent for building temporal cols is {} minutes'.format(round((time.time()-start_time)/60,1)))

## making demographic cols ##

demo_final = []
for i,row in df_enriched.iterrows():
  demo_arr = []
  for col in (demographic_cols+demographic_norm):
    demo_arr = demo_arr + [row[col]]
  demo_final = demo_final + [demo_arr]

  demo_vals[row['Country_Province']] = row[demographic_cols+demographic_norm]
print(len(demo_final),len(demo_final[0]))

df_enriched['Demographic_cols'] = demo_final

df_enriched[df_enriched['Country_Province']=='India'].tail()

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

X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in training_df["Temporal_cols"].values]),(training_item_count,5,sequence_length)),(0,2,1) )).astype(np.float32)
X_demographic_train = np.asarray([np.asarray(x) for x in training_df["Demographic_cols"]]).astype(np.float32)
Y_cases_train = np.asarray([np.asarray(x) for x in training_df["ConfirmedCases_Day_Norm"]]).astype(np.float32)
Y_fatalities_train = np.asarray([np.asarray(x) for x in training_df["Fatalities_Day_Norm"]]).astype(np.float32)

X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in validation_df["Temporal_cols"]]),(validation_item_count,5,sequence_length)),(0,2,1)) ).astype(np.float32)
X_demographic_test = np.asarray([np.asarray(x) for x in validation_df["Demographic_cols"]]).astype(np.float32)
Y_cases_test = np.asarray([np.asarray(x) for x in validation_df["ConfirmedCases_Day_Norm"]]).astype(np.float32)
Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_df["Fatalities_Day_Norm"]]).astype(np.float32)

print(X_temporal_train.shape,
      X_demographic_train.shape,
      X_temporal_test.shape,
      X_demographic_test.shape)

## building the model ##
#, return_sequences=True, recurrent_dropout=0.2

temporal_input_layer = Input(shape=(days_in_sequence,5))
main_rnn_layer = layers.LSTM(32)(temporal_input_layer)

#demographic input branch
demographic_input_layer = Input(shape=(29))
demographic_dense_1 = layers.Dense(32)(demographic_input_layer)
demographic_dense_2 = layers.Dense(8)(demographic_dense_1)
demographic_dropout = layers.Dropout(0.2)(demographic_dense_2)

#merge temporal and demographic ##
merge = layers.Concatenate(axis=-1)([main_rnn_layer,demographic_dropout])

#cases output branch
# rnn_c = layers.LSTM(16)(main_rnn_layer)
# merge_c = layers.Concatenate(axis=-1)([main_rnn_layer,demographic_dropout])
# dense_c_1 = layers.Dense(64)(merge_c)
# dropout_c_1 = layers.Dropout(0.3)(dense_c_1)
dense_c_2 = layers.Dense(16)(merge)
dropout_c_2 = layers.Dropout(0.3)(dense_c_2)
cases = layers.Dense(1, activation='relu',name="cases")(dropout_c_2)

#fatality output branch
# rnn_f = layers.LSTM(16)(main_rnn_layer)
# merge_f = layers.Concatenate(axis=-1)([main_rnn_layer,demographic_dropout])
# dense_f_1 = layers.Dense(64)(merge_f)
# dropout_f_1 = layers.Dropout(0.3)(dense_f_1)
dense_f_2 = layers.Dense(16)(merge)
dropout_f_2 = layers.Dropout(0.3)(dense_f_2)
fatalities = layers.Dense(1, activation='relu', name="fatalities")(dropout_f_2)

model = Model([temporal_input_layer,demographic_input_layer], [cases,fatalities])
model.summary()

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, 
                                   factor=0.6),
            ModelCheckpoint(filepath='best_model.h5', 
                        monitor='val_loss', save_best_only=True)]

opti = optimizers.Adam(lr=1e-3)
model.compile(loss=[tf.keras.losses.MeanSquaredError(),
                    tf.keras.losses.MeanSquaredError()], 
              optimizer=opti)

## training model ##

history = model.fit(x=[X_temporal_train,X_demographic_train], 
                    y=[Y_cases_train, Y_fatalities_train], 
                    epochs = 50, 
                    batch_size = 16, 
                    validation_data=([X_temporal_test,X_demographic_test],  
                                      [Y_cases_test, Y_fatalities_test]),
                    callbacks = callbacks)

## making test data ##

test_df = df_test[df_test['Country_Province'].isin(enrichedCountries)].copy()
embed_cols = ['quarantine','restrictions','schools'] + output_cols + day_cols
test_df = test_df.join(pd.DataFrame(columns = embed_cols))

test_overlap_mask = (test_df['Date_datetime'] <= df_enriched_copy['Date_datetime'].max())
train_overlap_mask = (df_enriched_copy['Date_datetime'] >= test_df['Date_datetime'].min())
# print((test_df.loc[test_overlap_mask]))
# print((df_enriched_copy.loc[train_overlap_mask]))
test_df.loc[test_overlap_mask, embed_cols] = df_enriched_copy.loc[train_overlap_mask, embed_cols].values

pred_dt_range = pd.date_range(start = df_enriched_copy['Date_datetime'].max() + pd.Timedelta(days=1), 
                              end = test_df['Date_datetime'].max(), freq = '1D')

for col in norm_cols1:
  test_df[col+'_Norm'] = ((test_df[col] - props[col+'_Norm'][0])*1.0)/props[col+'_Norm'][1]

model.load_weights("best_model.h5")

df_enriched_copy[df_enriched_copy['Country_Province']=='India'].tail()

for country in enrichedCountries:
    demo_input = np.array([demo_vals[country]]).astype(np.float32)
    # print(demo_input.shape)
    for date in pred_dt_range:
      mask = (test_df['Date_datetime'] == date) & (test_df['Country_Province'] == country)

      temp_input = []
      for lag in lags:
          mask_org = (test_df['Date_datetime'] == (date - pd.Timedelta(days=lag))) & (test_df['Country_Province'] == country)
          day_input = []

          for col in temporal_cols:
            try:
                day_input.append(test_df.loc[mask_org, col].values[0])
            except:
                day_input.append(0)

          temp_input = temp_input + [day_input]

      temp_input = np.array([temp_input])
      # print(temp_input.shape)

      # print('predict shape = '+str(temp_input))
      # print('predict shape = '+str(demo_input))
      result = model.predict([temp_input,demo_input])

      result[0] = np.nan_to_num(result[0])
      result[1] = np.nan_to_num(result[1])

      confday = result[0][0][0]*props['ConfirmedCases_Day_Norm'][1] + props['ConfirmedCases_Day_Norm'][0] 
      fatday = result[1][0][0]*props['Fatalities_Day_Norm'][1] + props['Fatalities_Day_Norm'][0]

      mask_prev = (test_df['Date_datetime'] == (date - pd.Timedelta(days=1))) & (test_df['Country_Province'] == country)
      test_df.loc[mask,'ConfirmedCases'] = test_df.loc[mask_prev,'ConfirmedCases'].values[0] + confday
      test_df.loc[mask,'Fatalities'] = test_df.loc[mask_prev,'Fatalities'].values[0] + fatday

      test_df.loc[mask,'ConfirmedCases_Day'] = confday
      test_df.loc[mask,'Fatalities_Day'] = fatday
      test_df.loc[mask,'schools'] = latest_row['schools'].values[0]
      test_df.loc[mask,'restrictions'] = latest_row['restrictions'].values[0]
      test_df.loc[mask,'quarantine'] = latest_row['quarantine'].values[0]

      print(country,date,
            result[0][0][0],result[1][0][0],
            confday,fatday,
            test_df.loc[mask,'ConfirmedCases'].values,
            test_df.loc[mask,'Fatalities'].values)
        
## for countries that are not in enriched list ##

train_df_2 = df_train[df_train['Country_Province'].isin(lockdownCountries)].copy()
test_df_2 = df_test[df_test['Country_Province'].isin(lockdownCountries)].copy()
test_df_2 = test_df_2.join(pd.DataFrame(columns = output_cols))

test_overlap_mask = (test_df_2['Date_datetime'] <= train_df_2['Date_datetime'].max())
train_overlap_mask = (train_df_2['Date_datetime'] >= test_df_2['Date_datetime'].min())
test_df_2.loc[test_overlap_mask, output_cols] = train_df_2.loc[train_overlap_mask, output_cols].values

pred_dt_range = pd.date_range(start = train_df_2['Date_datetime'].max() + pd.Timedelta(days=1), 
                              end = test_df_2['Date_datetime'].max(), freq = '1D')

for country in lockdownCountries:

    print(country)
    
    country_train = train_df_2[(train_df_2['Country_Province']==country) &
                            (train_df_2['ConfirmedCases']>0)]
    country_test = test_df_2[test_df_2['Country_Province']==country]
    
    country_min_date = country_train['Date_datetime'].min()

    country_train['Days since start'] = country_train['Date_datetime']-country_min_date
    country_train['Days since lockdown'] = country_train['Date_datetime'] - lockdown_dates[country]
    country_train['Days since lockdown'] = country_train['Days since lockdown'].apply(
        lambda x:x if(x>datetime.timedelta(minutes=0)) else datetime.timedelta(days=0))
    
    country_test['Days since start'] = country_test['Date_datetime']-country_min_date
    country_test['Days since lockdown'] = country_test['Date_datetime'] - lockdown_dates[country]
    country_test['Days since lockdown'] = country_test['Days since lockdown'].apply(
        lambda x:x if(x>datetime.timedelta(minutes=0)) else datetime.timedelta(days=0))

    ## normalising data ##
    math_cols = ['Days since start','Days since lockdown','ConfirmedCases','Fatalities']
    props = {}
    for col in math_cols:
        minmax = []
        minmax.append(country_train[col].min())
        minmax.append(country_train[col].max() - minmax[0])
        props[col] = minmax
        
        country_train[col] = ((country_train[col] - minmax[0])*1.0)/(minmax[1]*1.0)
        country_train[col] = country_train[col].fillna(0)
        
    country_test['Days since start'] = ((country_test['Days since start'] 
                                         - props['Days since start'][0])*1.0)/(props['Days since start'][1]*1.0)

    country_test['Days since lockdown'] = ((country_test['Days since lockdown'] 
                                         - props['Days since lockdown'][0])*1.0)/(props['Days since lockdown'][1]*1.0)
    country_test = country_test.fillna(0.0)
    
    country_train = country_train.replace([np.inf,-np.inf],0.0)
    country_test = country_test.replace([np.inf,-np.inf],0.0)
    
    ## predicting confirmed cases ##

    train_X = ((country_train[['Days since start','Days since lockdown']]).to_numpy()).reshape((-1,2))
    train_y_conf = country_train.ConfirmedCases.to_numpy()
    
    test_X = ((country_test[['Days since start','Days since lockdown']]).to_numpy()).reshape((-1,2))

    clf_conf = LinearRegression()
    clf_conf.fit(train_X,train_y_conf)

    poly_pred_init_conf = clf_conf.predict(test_X)

    poly_pred_conf = np.array([x if x>0 else 0 for x in poly_pred_init_conf])
    poly_pred_conf = np.nan_to_num(poly_pred_conf)

    ## predicting fatalities ##

    train_X = ((country_train[['Days since start','Days since lockdown']]).to_numpy()).reshape((-1,2))
    train_y_fat = country_train.Fatalities.to_numpy()
    
    test_X = ((country_test[['Days since start','Days since lockdown']]).to_numpy()).reshape((-1,2))
    
    clf_fat = LinearRegression()
    clf_fat.fit(train_X,train_y_fat)


    poly_pred_init_fat = clf_fat.predict(test_X)

    poly_pred_fat = np.array([x if x>0 else 0 for x in poly_pred_init_fat])
    poly_pred_fat = np.nan_to_num(poly_pred_fat)
    
    mask = test_df_2['Country_Province'] == country
    test_df_2.loc[mask,'ConfirmedCases'] = poly_pred_conf
    test_df_2.loc[mask,'ConfirmedCases'] = (test_df_2[mask]['ConfirmedCases']*
                                          props['ConfirmedCases'][1]) + props['ConfirmedCases'][0]

    test_df_2.loc[mask,'Fatalities'] = poly_pred_fat
    test_df_2.loc[mask,'Fatalities'] = (test_df_2[mask]['Fatalities']*
                                          props['Fatalities'][1]) + props['Fatalities'][0]
    
test_df = test_df[['ForecastId','ConfirmedCases','Fatalities']]
test_df_2 = test_df_2[['ForecastId','ConfirmedCases','Fatalities']]

sub = pd.concat([test_df,test_df_2])
sub = sub.sort_values('ForecastId')
sub.to_csv('submission.csv',index=False)