import kagglegym
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
median_values = train.median(axis=0)
train.fillna(median_values, inplace=True)

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

m1 = LinearRegression(n_jobs=-1)
m2 = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=12, verbose=0)
m3 = Ridge()
train = train.loc[y_is_within_cut]

l = ['technical_30', 'technical_20']
m1.fit(train[l].values, train.y)
m2.fit(train[l].values, train.y)
m3.fit(train[l].values, train.y)

ymedian_dict = dict(train.groupby(["id"])["y"].median())


def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.94 * y + 0.06 * ymedian_dict[id] if id in ymedian_dict else y


while True:
    observation.features.fillna(median_values, inplace=True)
    observation.target['y'] = m1.predict(observation.features[l].values).clip(low_y_cut,high_y_cut)*0.3+m2.predict(observation.features[l].values).clip(low_y_cut,high_y_cut)*0.3+m3.predict(observation.features[l].values).clip(low_y_cut, high_y_cut)*0.4
    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(observation.target)
    if done:
        break
    
print(info)