import kagglegym
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Lars

env = kagglegym.make()
observation = env.reset()

t0 = time.time()

# first get train labels
y_train = observation.train['y']

# clip
clip_r = .5
y_max = y_train.max()
y_min = y_train.min()
low_y_cut = y_min * clip_r
high_y_cut = y_max * clip_r
y_is_above_cut = (y_train > high_y_cut)
y_is_below_cut = (y_train < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# select features
col = [u'fundamental_53', u'fundamental_56', u'technical_20', u'technical_30']
X_train = observation.train[col]

# fill na with col median
col_median = X_train.median(axis=0)
X_train.fillna(col_median, inplace=True)

# sigmoid transformation for existing
X_train = X_train.apply(expit)

print('training on features: ', X_train.columns)

model = Lars()
model.fit(X_train[y_is_within_cut], 
          y_train[y_is_within_cut])

t1 = time.time()
print ('training took {0:.2f} mins'.format((t1-t0)/60.))


while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # select feature
    X_test = observation.features[col]
    
    # fill na
    X_test.fillna(col_median, inplace=True)
    
    # sigmoid transformation for existing
    X_test = X_test.apply(expit)
    
    observation.target.y = model.predict(X_test).clip(low_y_cut, high_y_cut)

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
    
t2 = time.time()
print('predicting took {0:.2f} mins'.format((t2-t1)/60.))