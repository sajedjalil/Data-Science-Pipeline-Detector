import pandas as pd
import numpy as np
import random
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

#Compute RMSLE
def produce_rmsle(test_df,targets):
    rmsle = {}
    print('|'+'|'.rjust(25)+'|'.rjust(10))
    print('|'+''.join(['-' for i in range(24)])+'|'+''.join(['-' for i in range(9)])+'|')
    for t in targets:
        rmsle[t] = np.sqrt(np.sum(np.square(np.log(test_df['estimated_'+t].values+1)-np.log(test_df[t].values+1)))/len(test_df))
        print('|'+(t+'|').rjust(25)+('{0:.4f}|'.format(rmsle[t]).rjust(10)))
    print('|'+('averaged|').rjust(25)+('{0:.4f}|'.format(np.mean([rmsle[t] for t in rmsle])).rjust(10)))

#Initialise the random seeds
random.seed(0)
np.random.seed(0)
    
#Read train and test
train_df = pd.read_csv('../input/tabular-playground-series-jul-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-jul-2021/test.csv')

#Generate datetime features
train_df['hour_of_day'] = pd.to_datetime(train_df['date_time']).dt.hour
train_df['day_of_month'] = pd.to_datetime(train_df['date_time']).dt.day
train_df['month'] = pd.to_datetime(train_df['date_time']).dt.month
train_df['day_of_week'] = pd.to_datetime(train_df['date_time']).dt.dayofweek
test_df['hour_of_day'] = pd.to_datetime(test_df['date_time']).dt.hour
test_df['day_of_month'] = pd.to_datetime(test_df['date_time']).dt.day
test_df['month'] = pd.to_datetime(test_df['date_time']).dt.month
test_df['day_of_week'] = pd.to_datetime(test_df['date_time']).dt.dayofweek

#Generate sensors down feature
train_df['sensors_down'] = False
train_df.loc[train_df['target_benzene'] == 0.1,'sensors_down'] = True

#User Nov-Dec 2010 for validation
#Use Mar-Aug 2010 for training
#Leave Aug-Oct 2010 for scale learning
valid_df = train_df.loc[train_df['month']>=11]
no_train_df = train_df.loc[(train_df['month']>8) & (train_df['month']<11)]
train_df = train_df.loc[train_df['month']<=8][:-1]

#Define features to use
feature_list = ['hour_of_day','day_of_week','day_of_month','deg_C','relative_humidity','absolute_humidity','sensor_1','sensor_2','sensor_3','sensor_4','sensor_5']         

#Estimate when sensors are messed up with a binary classifier
lgb_sensor = LGBMRegressor(objective='binary')
lgb_sensor.fit(train_df[feature_list].values,train_df['sensors_down'].values)
valid_df['estimated_sensors_down'] = lgb_sensor.predict(valid_df[feature_list].values) >= 0.5
test_df['estimated_sensors_down'] = lgb_sensor.predict(test_df[feature_list].values) >= 0.5

#Train the predictors for the 3 targets
xgb = {}
lgb = {}
for t in ['target_carbon_monoxide','target_benzene','target_nitrogen_oxides']:
    lgb[t] = LGBMRegressor()
    lgb[t].fit(train_df[feature_list].values,train_df[t].values)
    xgb[t] = XGBRegressor(objective="reg:squarederror")
    xgb[t].fit(train_df[feature_list].values,train_df[t].values)

#Infer the 3 target values
for t in ['target_carbon_monoxide','target_benzene','target_nitrogen_oxides']:
    valid_df['estimated_'+t] = (lgb[t].predict(valid_df[feature_list].values) + xgb[t].predict(valid_df[feature_list].values)) / 2
    test_df['estimated_'+t] = (lgb[t].predict(test_df[feature_list].values) + xgb[t].predict(test_df[feature_list].values)) / 2
    
print('RMSLE results for cross-validation set with initial LigtGBM classifier')
produce_rmsle(valid_df,['target_carbon_monoxide','target_benzene','target_nitrogen_oxides'])

#Correct the difference in scale for NOx since September using a simple scale factor
factor = np.mean(no_train_df['target_nitrogen_oxides'])/np.mean(train_df['target_nitrogen_oxides'])
valid_df['estimated_target_nitrogen_oxides'] = factor * valid_df['estimated_target_nitrogen_oxides']
test_df['estimated_target_nitrogen_oxides'] = factor * test_df['estimated_target_nitrogen_oxides']

print('RMSLE results for cross-validation set rescaling new values from September 2010')
produce_rmsle(valid_df,['target_carbon_monoxide','target_benzene','target_nitrogen_oxides'])

#When sensors are erroneous values of CO and NOx are reverted to the means of the nearest estimations
for t in ['target_carbon_monoxide','target_nitrogen_oxides']:
     for i in np.unique(train_df['day_of_week'].values):
        for j in np.unique(train_df['hour_of_day'].values):
            mean_hour_day = np.mean(valid_df.loc[(~valid_df['estimated_sensors_down']) & (valid_df['day_of_week']==i) & (valid_df['hour_of_day']==j)]['estimated_'+t].values)
            valid_df.loc[(valid_df['estimated_sensors_down']) & (valid_df['day_of_week']==i) & (valid_df['hour_of_day']==j),'estimated_'+t] = mean_hour_day
            mean_hour_day = np.mean(test_df.loc[(~test_df['estimated_sensors_down']) & (test_df['day_of_week']==i) & (test_df['hour_of_day']==j)]['estimated_'+t].values)
            test_df.loc[(test_df['estimated_sensors_down']) & (test_df['day_of_week']==i) & (test_df['hour_of_day']==j),'estimated_'+t] = mean_hour_day
            
print('RMSLE results for cross-validation correcting for sensor errors')
produce_rmsle(valid_df,['target_carbon_monoxide','target_benzene','target_nitrogen_oxides'])
    
#Generate final CSV output with test set 
pd.DataFrame({**{'date_time':test_df['date_time'].values},**{t:test_df['estimated_'+t].values for t in lgb}}).set_index('date_time').to_csv('submission.csv')