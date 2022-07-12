import kagglegym
import numpy as np
import pandas as pd

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

#Compute the means per id
y = observation.train[['id', 'timestamp', 'y']]
means = y.groupby('id')['y'].mean()


# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

while True:
    target = observation.target
    
    #replace y column of target with the means of each id
    target.drop('y', axis=1, inplace=True)
    target = target.join(means, on='id')
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print("New ids: {}".format(target['y'].isnull().sum()))

    #use 0 as the prediction for unobserved ids
    target.fillna(0.0, inplace=True)
    #mean = target['y'].mean()
    #target.fillna(mean if not np.isnan(mean) else 0.0, inplace=True)
    
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break