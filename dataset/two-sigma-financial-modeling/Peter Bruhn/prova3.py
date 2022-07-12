import kagglegym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

env = kagglegym.make()

observation = env.reset()

cols = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
col = 'technical_30'

train = observation.train
train.fillna(train.mean(axis=0), inplace = True)

model = LinearRegression()
model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)

mean_values = train.mean(axis=0)

while True:

    observation.features.fillna(mean_values, inplace = True)
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    y_pred = model.predict(np.array(observation.features[col].values).reshape(-1,1))
    observation.target.y = y_pred
    target = observation.target
        
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break

print(info)
