import kagglegym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))


observation.train.fillna(observation.train.mean(), inplace = True)
scaler = StandardScaler()
data = scaler.fit_transform(observation.train.iloc[:, 2:-1].values)

#model = LinearRegression()
#model.fit(data, observation.train.y)

model = LinearRegression()
model.fit(data, observation.train.y)

while True:
    
    observation.features.fillna(observation.features.mean(), inplace = True)
    # target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    y_pred = model.predict(scaler.transform(observation.features.iloc[:, 2:].values))
    #y_pred=0
    observation.target.y = y_pred
    target = observation.target
    
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break