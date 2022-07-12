import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import vincenty
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

path='../input/train.csv'
data=pd.read_csv(path)
df=data

#Calculating Vincenty distance between pickup and dropoff
df['distance'] = df.apply(lambda x: vincenty((x[6], x[5]), (x[8], x[7])).miles, axis = 1)

#Split Train/Validation data
train_data, validation_data = train_test_split(df,train_size=0.8)

#Baseline model: average trip duration
train_data['baseline']=train_data.trip_duration.mean()
rmse_baseline = np.sqrt(mean_squared_error(np.array(train_data.trip_duration), np.array(train_data.baseline)))
print('RMSE for baseline model: ', rmse_baseline)                                      

#Basic regression model
train_data=np.array(train_data)
validation_data=np.array(validation_data)

X = train_data[:,11].reshape(-1,1)
y = train_data[:, 10]
lm = LinearRegression()
lm.fit(X, y)

X_validation = validation_data[:,11].reshape(-1,1)
y_validation = validation_data[:, 10]
y_pred = lm.predict(X_validation)

rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
print('RMSE basic regression (x=distance): ', rmse)
print('RMSE - difference from baseline ', rmse_baseline-rmse)