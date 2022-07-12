## IMPORTS ##
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timezone
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from math import sin, cos, sqrt, atan2, radians
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
## ##

## distance km
def latlong_km(lat1, lon1, lat2, lon2):   

    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def data_cleanse(d):
    ## drop the id, its not needed
    d.drop(['id', "store_and_fwd_flag"], axis=1, inplace=True)
    if set(['dropoff_datetime']).issubset(d.columns):
        ## drop the dropoff datetime since its not a feature in the test set
        d.drop(['dropoff_datetime'], axis=1, inplace=True)
    ## convert store flag to ints    
    # d["store_and_fwd_flag"].replace(["Y", "N"], [1,0], inplace=True)
    ## convert date_time to datetime
    d['pickup_datetime'] = pd.to_datetime(d['pickup_datetime'])
    ## get day of week, hour of day, min of hour, and month from date time and create new columns
    d["pickup_day_of_week"] = d["pickup_datetime"].dt.strftime('%u').astype(int)
    d["pickup_hour_of_day"] = d["pickup_datetime"].dt.strftime('%H').astype(int)
    d["pickup_min_of_hour"] = d["pickup_datetime"].dt.strftime('%M').astype(int)
    d["pickup_month"] = d["pickup_datetime"].dt.strftime('%m').astype(int)

    d['km'] = d.apply(lambda row: latlong_km(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
    d.drop(['pickup_datetime','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'], axis=1, inplace=True)
    return d

#load the datasets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# clean the datasets
train_data = data_cleanse(train_data)
test_data = data_cleanse(test_data)

#print train head
print(train_data.head())

train_data_arr = train_data.as_matrix() ## turn pd into array
train_data_arr = train_data_arr[:50] ## only get first 50 to prevent overloading on kaggle. comment this out if using
train_X = train_data_arr[:,[0,1,2,4,5,6,7,8]]
train_y = train_data_arr[:,[3]]
train_y = train_y.reshape(-1, 1)


test_data_arr = test_data.as_matrix()
print(test_data_arr)
test_X = test_data_arr[:,[0,1,2,3,4,5,6,7]]

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2)

### edit these features to do feature selection
# rfc = DecisionTreeRegressor(max_features="auto")
# param_grid = { 
#     # "loss": ["deviance", "exponential"],
#     # "n_estimators": [100,50,200],
#     # "max_depth": range(3,10),
#     # "min_samples_split": [3],
#     # "min_samples_leaf": [8],
#     #"max_features": ["auto"],
#     # "max_leaf_nodes": [47]
#     #"criterion": ["mse", "mae"]
#     }
    
# rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
# rfc.fit(X_train, y_train.ravel())   

# print('Best score:', rfc.best_score_)
# print('max_features:',rfc.best_estimator_.max_features)
# print('criterion:',rfc.best_estimator_.criterion)
# print('Estimators:',rfc.best_estimator_.n_estimators)
# print('loss:',rfc.best_estimator_.loss)
#etc....


model = DecisionTreeRegressor(max_features="auto")
model.fit(X_train,y_train.ravel())
print(model.score(X_test,y_test.ravel()))
p = model.predict(test_X)
p = np.round(p,0)
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.savefig("test.png")

## load test file
# match predictions
test_data = pd.read_csv('../input/test.csv')
test_data['trip_duration'] = ""
for index, row in test_data.iterrows():
    survived = round(p[index])
    test_data.set_value(index,"trip_duration",survived)
test_data.drop(['pickup_datetime', 'vendor_id' ,'passenger_count' ,'pickup_longitude' ,'pickup_latitude' ,'dropoff_longitude' ,'dropoff_latitude' ,'store_and_fwd_flag'], axis=1, inplace=True)
print(test_data.head())
test_data.to_csv('predictions.csv', index = False)