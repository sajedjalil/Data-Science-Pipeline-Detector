import kagglegym
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

train = observation.train
train_median = train.median()

train.fillna(train_median, inplace = True)
#scaler = StandardScaler()
#data = scaler.fit_transform(observation.train.iloc[:, 2:-1])
#train = train.query('y > -0.086093 & y < 0.093497')


#low_y_cut = -0.086093
#high_y_cut = 0.093497

## get from rf.feature_importances_
topfeats = ['timestamp', 'technical_20', 'technical_30', 'technical_40', 'technical_35', 'fundamental_30', 'fundamental_18', 'fundamental_44', 'derived_3', 'fundamental_53', 'fundamental_58']
rf = RandomForestRegressor(max_depth=3, n_jobs = -1, n_estimators=30)
rf.fit(train[topfeats], train.y.values)

# lr = LinearRegression()
# lr.fit(data, observation.train.y)

#ridge = Ridge()
#ridge.fit(data, observation.train.y)

while True:
    
    observation.features.fillna(train_median, inplace = True)
    # target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    #y_pred = rf.predict(observation.features[topfeats]).clip(low_y_cut, high_y_cut)
    y_pred = rf.predict(observation.features[topfeats])
    observation.target.y = y_pred
    target = observation.target
    
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break