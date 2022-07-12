import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
# The "environment" is our interface for code competitions
env = kagglegym.make()
target = 'y'
# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))
# Get the train dataframe
train = observation.train
col = ['technical_20']

mean = train.mean(axis=0)
train=train.fillna(mean)

# Observed with histograns:
low_y_cut = -0.083
high_y_cut = 0.09

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

model = lm.Ridge(alpha=0.9, random_state=0)
model.fit(np.array(train.loc[y_is_within_cut, col].values).reshape(-1,1), train.loc[y_is_within_cut, target])


while True:
    observation.features.fillna(mean, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    target = observation.target

    # The "target" dataframe is a template for what we need to predict:
    print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break