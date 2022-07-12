import numpy as np
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
# Machine Learning
from xgboost import XGBRegressor
from math import sqrt
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("../input/train.csv",nrows= 1000000)
test_df = pd.read_csv("../input/test.csv")

# The Haversine formula lets us calculate the distance on a sphere using
# Latitude and Longitude
def haversine(lon1, lat1, lon2, lat2):
    lat1 = np.radians(lat1)
    lat2= np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat=(lat2-lat1).abs()
    dlon=(lon2-lon1).abs()
    R = 6371 #radius of Earth
    a = (np.sin(dlat/2.0))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def distance_travelled(df):
    df['Distance_Travelled'] = haversine(df.dropoff_longitude,df.dropoff_latitude,df.pickup_longitude,df.pickup_latitude)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df.pickup_datetime.dt.hour
    df['day'] = df.pickup_datetime.dt.day
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['year'] = df.pickup_datetime.dt.year

distance_travelled(train_df)
distance_travelled(test_df)

train_df.dropna(how = 'any', axis = 'rows', inplace=True)
train_df.isnull().sum()

features = train_df.drop(['fare_amount','pickup_datetime'],axis=1)

label = train_df.pop('fare_amount')
test_df=test_df.drop(['key', 'pickup_datetime'], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size=0.05, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

xgb_regressor = XGBRegressor(n_estimators=150)
xgb_regressor.fit(train_features, train_labels)

prediction = xgb_regressor.predict(test_df)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = prediction
submission.to_csv('fare_pred.csv', index=False)