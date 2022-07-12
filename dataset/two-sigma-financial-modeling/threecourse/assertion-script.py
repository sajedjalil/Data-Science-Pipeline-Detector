import kagglegym
import numpy as np
import pandas as pd

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

train = observation.train
# timestamp = train["timestamp"].max()
train_columns = [c for c in train.columns if c != "y"]
timestamp = -1
while True:
    features = observation.features
    target = observation.target
    
    previous_timestamp = timestamp
    timestamp = observation.features["timestamp"][0]
    
    assert(timestamp > previous_timestamp)
    assert(features["timestamp"].min() == features["timestamp"].max())
    assert(len(features["id"].unique()) == len(features))

    assert((features.columns.values == train_columns).all())
    assert((features["id"].values == target["id"].values).all())

    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break