import kagglegym
import numpy as np
#import pandas as p
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']


# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# Pipeline with polynomial features of degree 3 and Ridge regression with normalization  
model = make_pipeline(PolynomialFeatures(3), Ridge(random_state=42, normalize=True))

model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target])

ymedian_dict = dict(train.groupby(["id"])["y"].median())


def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y


while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[cols_to_use].values)
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    
    ## weighted y using average value
    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)