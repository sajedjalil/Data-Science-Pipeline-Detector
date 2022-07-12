# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import time
import datetime
from sklearn.linear_model import LinearRegression 
#import xgboost as xgb #XGBoost classifier
from sklearn.model_selection import train_test_split, cross_val_score
from math import sin, cos, sqrt, atan2, radians
from sklearn import metrics #evaluating models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def read_file_to_df(fileName,rows):
    # Set columns to most suitable type to optimize for memory usage
    traintypes = {'fare_amount': 'float32',
                  'pickup_datetime': 'str', 
                  'pickup_longitude': 'float32',
                  'pickup_latitude': 'float32',
                  'dropoff_longitude': 'float32',
                  'dropoff_latitude': 'float32',
                  'passenger_count': 'uint8'}
    
    cols = list(traintypes.keys())
    

    if (rows> 0):
        df = pd.read_csv(fileName, usecols=cols, dtype=traintypes,nrows=rows)
    else:
        df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
    df.head()
    print(df.shape)
    return df



def read_test_file_to_df(fileName):
    df = pd.read_csv(fileName)
    return df



def drop_rows_with_nan(df):
    print("Before dropna")
    df.dropna(how = 'any', axis = 'rows',inplace=True)
    print(df.shape)

def data_cleanup(tmp_df):
    print("Before clearing outliers")
    tmp_df = tmp_df[tmp_df['fare_amount'] > 0]
    tmp_df = tmp_df[tmp_df['pickup_longitude'] < -72]
    tmp_df = tmp_df[(tmp_df['pickup_latitude'] > 40) & (tmp_df['pickup_latitude'] < 44)]
    tmp_df = tmp_df[tmp_df['dropoff_longitude'] < -72]
    tmp_df = tmp_df[(tmp_df['dropoff_latitude'] > 40) & (tmp_df['dropoff_latitude'] < 44)]
    tmp_df = tmp_df[(tmp_df['passenger_count'] > 0) & (tmp_df['passenger_count'] < 10)]
    print(tmp_df.shape)

def reformat_pickup_datetime(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 13)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],utc=True,format='%Y-%m-%d %H')

def add_day_of_week_feature(tmp_df):
    tmp_df['dayOfWeek'] = tmp_df['pickup_datetime'].dt.dayofweek.astype('uint8')


def add_time_of_day_feature(tmp_df):
    tmp_df['timeOfDay'] = tmp_df['pickup_datetime'].dt.hour.astype('uint8')

def add_month_feature(tmp_df):
    tmp_df['month'] = tmp_df['pickup_datetime'].dt.month.astype('uint8')

def add_week_of_year_feature(tmp_df):
    tmp_df['weekOfYear'] = tmp_df['pickup_datetime'].dt.weekofyear.astype('uint8')



def distance_between_two_points(row):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(row['pickup_latitude'])
    lon1 = radians(row['pickup_longitude'])
    lat2 = radians(row['dropoff_latitude'])
    lon2 = radians(row['dropoff_longitude'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return "{0:.3f}".format(distance)

#calculate distance between two cordinates 
def add_distance_feature(tmp_df):
    tmp_df['distance'] = tmp_df.apply(distance_between_two_points,axis=1).apply(pd.to_numeric).astype('float32')
    #tmp_df['distance'] = tmp_df.apply(distance_between_two_points,axis=1)




def filter_based_on_distance(df,distance):
    df = df[(df['distance'] < 100)]    
    

def data_one_hot_encoding(tmp_df):
    print("Before one hot encoding")
    
    tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['timeOfDay'], prefix='timeOfDay')],axis=1)
    tmp_df.drop(['timeOfDay'],axis=1, inplace=True)
    
    tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['dayOfWeek'], prefix='dayOfWeek')],axis=1)
    tmp_df.drop(['dayOfWeek'],axis=1, inplace=True)
    
    tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['month'], prefix='month')],axis=1)
    tmp_df.drop(['month'],axis=1, inplace=True)
    
    #tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['weekOfYear'], prefix='weekOfYear')],axis=1)
    print(tmp_df.columns)


def prepare_data_split(tmp_df):
    #tmp_X = tmp_df.drop(['pickup_datetime','fare_amount','key'],axis=1)
    tmp_X = tmp_df.drop(['pickup_datetime','fare_amount'],axis=1)
    tmp_y = tmp_df['fare_amount']
    print("Before train test split")
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(tmp_X,tmp_y, test_size=0.05)
    print("df.shape:" + str(tmp_df.shape))
    print("tmp_X.shape:" + str(tmp_X.shape))
    print("tmp_y.shape:" + str(tmp_y.shape))
    return tmp_X,tmp_y,tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test




def fit_linear_model(X_train,X_test,y_train,y_test):
    print("Before applying regression model")
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    print(lm.score(X_train,y_train))
    print(lm.score(X_test,y_test))
    return lm    



def get_rmse(model,data,output):
    y_pred = model.predict(data)
    rmse = np.sqrt(metrics.mean_squared_error(y_pred, output))
    #print("rmse:" + str(rmse))
    return rmse


def fit_random_forest_model(X_train,X_test,y_train,y_test):
    rfr = RandomForestRegressor(n_estimators=30,min_samples_split=2,criterion="mse", n_jobs=3,random_state=1)
    rfr.fit(X_train,y_train)
    print("RFR Score on train:" + str(rfr.score(X_train,y_train)))
    print("RFR score on test:" + str(rfr.score(X_test,y_test)))
    return rfr


def fit_random_forest_model_grid_search(X_train,X_test,y_train,y_test):
    rfr = RandomForestRegressor(criterion="mse", n_jobs=3,random_state=1)
    parameters = {'n_estimators':[45], 'max_features':['auto'],'min_samples_split':[3]}
    gsc = GridSearchCV(rfr, parameters)
    
    gsc.fit(X_train,y_train)
    print("RFR Score on train:" + str(gsc.score(X_train,y_train)))
    print("RFR score on test:" + str(gsc.score(X_test,y_test)))
    return gsc

    
def output_submission(model,df_test,test_X):
    test_pred = model.predict(test_X)
    print(type(test_pred))
    print("Shape of test_pred:" + str(test_pred.shape))
    test_pred = np.round(test_pred,2)
    df
    print(df_test.shape)
    print(test_pred.shape)
    # Write the predictions to a CSV file which we can submit to the competition.
    submission = pd.DataFrame(
        {'key': df_test.key, 'fare_amount': test_pred},
        columns = ['key', 'fare_amount'])
    submission.to_csv('submission.csv', index = False)

#def main():
df = read_file_to_df("../input/train.csv",1_000_000)
drop_rows_with_nan(df)
data_cleanup(df)
print("Null before feature addition")
print(df.isnull().sum())
print("Before adding new features")
#Add new features using existing features

reformat_pickup_datetime(df)
add_day_of_week_feature(df)
add_time_of_day_feature(df)
add_month_feature(df)
add_distance_feature(df)
print("Null after feature addition")
filter_based_on_distance(df,100)
data_one_hot_encoding(df)    
print("Before dropping na")

drop_rows_with_nan(df)

X,y,X_train,X_test,y_train,y_test = prepare_data_split(df)


gsc= fit_random_forest_model(X_train,X_test,y_train,y_test)
rfrmse = get_rmse(gsc,X,y)
print("rfrmse:" + str(rfrmse))

df_test = read_test_file_to_df("../input/test.csv")
reformat_pickup_datetime(df_test)
add_day_of_week_feature(df_test)
add_time_of_day_feature(df_test)
add_month_feature(df_test)
add_distance_feature(df_test)
data_one_hot_encoding(df_test)    
test_X = df_test.drop(['pickup_datetime','key'],axis=1)
output_submission(gsc,df_test,test_X)
del df
del X,y,X_train,X_test,y_train,y_test