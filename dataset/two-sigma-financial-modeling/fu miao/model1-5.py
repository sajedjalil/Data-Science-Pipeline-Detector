import kagglegym
import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
# median_values = train.median(axis=0)
train.fillna(mean_values, inplace=True)

# cols_to_use = ['technical_20']
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']


# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

model = Ridge()
model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target])

# ymean_dict = dict(train.groupby(["id"])["y"].mean())
ymedian_dict = dict(train.groupby(["id"])["y"].median())
yvol_dict = dict(train.groupby(["id"])["y"].std())


def get_weighted_y(series):
    id, y = series["id"], series["y"]
    # return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y
    return 0.9 * y + 0.1 * ymedian_dict[id]*(1/100/yvol_dict[id]) if id in ymedian_dict else y



while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[cols_to_use].values)
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    
    ## weighted y using average value
    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)
    
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)