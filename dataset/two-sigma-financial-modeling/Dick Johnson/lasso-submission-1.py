import kagglegym
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Lasso


# The "environmento" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

data = observation.train


target = data['y']
data.drop('y', axis=1, inplace=True)

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data = imp.fit_transform(data)

data = preprocessing.normalize(data, norm='l1')


clf = Lasso(alpha=0.1, normalize=True)
clf.fit(data, target)


while True:
    feats = observation.features
    feats = feats.drop('timestamp', axis=1, inplace=False)
    feats['timestamp'] = observation.features['timestamp']
    
    feats = imp.fit_transform(feats)
    feats = preprocessing.normalize(feats, norm='l1')
    target = observation.target
    
    target['y'] = clf.predict(feats)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break