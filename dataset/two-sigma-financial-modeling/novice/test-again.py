import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.decomposition import FastICA

target = 'y'
env = kagglegym.make()
observation = env.reset()

# Get the train dataframe
train = observation.train
train.fillna(0, inplace=True)
cols_to_use = ['technical_20', 'technical_30']

print("Training...")
ica = FastICA(2, random_state=1)
sources = train[cols_to_use].values
model = lm.LinearRegression()
model.fit(sources, np.log1p(train.y))

print("Predicting...")
while True:
    observation.features.fillna(0, inplace=True)
    test_x = observation.features[cols_to_use].values
    observation.target.y = np.exp(model.predict(test_x)) - 1

    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)


