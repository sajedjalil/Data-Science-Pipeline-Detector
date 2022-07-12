import pandas as pd
import numpy as np

# training file is large (55 million rows), let's just train on 2 million rows
train = pd.read_csv('../input/train.csv',nrows = 2_000_000)
test = pd.read_csv('../input/test.csv')

train.isnull().sum()

test.isnull().sum()

# drop rows if there are null values in columns
train = train[pd.notnull(train['dropoff_longitude'])]
train = train[pd.notnull(train['dropoff_latitude'])]

# check for null values
train.isnull().sum()

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

# add new column to dataframe with distance in miles
train['distance_miles'] = distance(train.pickup_latitude, train.pickup_longitude, train.dropoff_latitude, train.dropoff_longitude)

# drop row if distance equals zero
train = train[(train[['distance_miles']] != 0).all(axis=1)]

X = train.iloc[:, [7,8]].values
y = train.iloc[:, 1].values

# add new column to dataframe with distance in miles
test['distance_miles'] = distance(test.pickup_latitude, test.pickup_longitude, test.dropoff_latitude, test.dropoff_longitude)

X_test = test.iloc[:, [6,7]].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)



# Predicting the Test set results
y_pred = regressor.predict(X_test)

# define test ID 
test_ID = test['key']

# create submission
sub = pd.DataFrame()
sub['key'] = test_ID
sub['fare_amount'] = y_pred
sub.to_csv('submission.csv',index=False)