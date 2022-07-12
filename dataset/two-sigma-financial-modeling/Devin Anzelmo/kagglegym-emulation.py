#NOTE r scoring is still not matching kaggles implementation, bug fix needed for this code
#Otherwise seems to reproduce the behavior of the kagglegym api
# This is a modification of the code posted by Frans Slothouber on the forum. see link
# https://www.kaggle.com/c/two-sigma-financial-modeling/discussion/26044#148202


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r

class Observation(object):
    def __init__(self, train, target, features):
        self.train = train
        self.target = target
        self.features = features


class Environment(object):
    def __init__(self):
        with pd.HDFStore("../input/train.h5", "r") as hfdata:
            self.timestamp = 0
            fullset = hfdata.get("train")
            self.unique_timestamp = fullset["timestamp"].unique()
            # Get a list of unique timestamps
            # use the first half for training and
            # the second half for the test set
            n = len(self.unique_timestamp)
            i = int(n/2)
            timesplit = self.unique_timestamp[i]
            self.n = n
            self.unique_idx = i
            self.train = fullset[fullset.timestamp < timesplit]
            self.test = fullset[fullset.timestamp >= timesplit]

            self.y_test_full = self.test['y'] # Just in case the full labels are needed later
            self.temp_test_y = None
            self.total_reward = 0

    def reset(self):
        timesplit = self.unique_timestamp[self.unique_idx]
        self.unique_idx += 1
        subset = self.test[self.test.timestamp == timesplit]

        # reset index to conform to how kagglegym works
        target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
        self.temp_test_y = target['y']

        target.loc[:, 'y'] = 0.0 # set the prediction column to zero

        # changed bounds to 0:110 from 1:111 to mimic the behavior of api for feature
        features = subset.iloc[:, :110].reset_index(drop=True)

        observation = Observation(self.train, target, features)
        return observation

    def step(self, target):
        if self.unique_idx == self.n:
            done = True
            observation = None
            reward = self.total_reward/self.y_test_full.shape[0]
            info = {'public_score': reward}
        else:

            reward = r_score(target.loc[:, 'y'], self.temp_test_y)
            print(reward)
            print(target.loc[:, 'y'])
            print(self.temp_test_y)

            # keep track of total reward
            self.total_reward += reward

            done = False
            info = {}

            timesplit = self.unique_timestamp[self.unique_idx]
            self.unique_idx += 1
            subset = self.test[self.test.timestamp == timesplit]

            # reset index to conform to how kagglegym works
            target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
            self.temp_test_y = target['y']

            # set the prediction column to zero
            target.loc[:, 'y'] = 0.0

            # column bound change on the subset
            # reset index to conform to how kagglegym works
            features = subset.iloc[:, 0:110].reset_index(drop=True)

            observation = Observation(self.train, target, features)

        return observation, reward, done, info

    def __str__(self):
        return "Environment()"


def make():
    return Environment()