import kagglegym
import numpy as np
import pandas as pd
import sys as sys
import re as re
import random as rnd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, Normalizer, Imputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Estimator used to train and predict.
model = LinearRegression()

# Pipeline to preprocess training data.
pipeline = Pipeline([ ('fill',  Imputer(strategy='mean')),
                      ('kbest', SelectKBest(f_regression, k=1)),
                    ])
 
features = ['timestamp','technical_20']     

# Setup kaggle environment and variables.
env = kagglegym.make()
observation = env.reset()
labels = observation.train.loc[:,'y']
data = observation.train.loc[:,features]

# Simplified training for this sample script.
print("Training...")
data = pipeline.fit_transform(data, labels)
model.fit(data, labels )

# Collect test data features to be able to calculate extra features (e.g. diffs),
# and predictions more efficiently.
print("Preparing test data...")
test_data = None
while True:
    data = observation.features.loc[:,features]
    timestamp = observation.features.timestamp[0]

    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    test_data = test_data.append(data) if test_data is not None else data
    observation, reward, done, info = env.step(observation.target)
    if done:
        break

# Generate more features, calculate diffs, and whatnot...
# ...
# ...

# Run pipeline and predict in one shot.
print("Predicting...")
test_data.loc[:,'y'] = model.predict(pipeline.transform(test_data))

# Submit predictions made above.
observation = env.reset()  # <--- !!!
print("Submitting results...")    
rewards = np.array([])
while True:
    target = observation.target
    timestamp = observation.features.timestamp[0]
    
    # Get precalculated results from test_data.
    target.loc[:,'y'] = test_data.loc[test_data.timestamp==timestamp,'y'] 
    observation, reward, done, info = env.step(target)
    rewards = np.append(rewards, reward)
    
    # Log rewards every now and then.
    if timestamp % 100 == 0:
        print("Timestamp #{}, average reward: {:.4f} +/- {:.4f}".format(timestamp, np.mean(rewards), np.std(rewards)))
        rewards = np.array([])
    
    if done:
        print("Info: {}".format(info))    
        break