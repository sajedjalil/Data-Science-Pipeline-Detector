# In this script I want to demonstrate how I computed the mean of every ID separately.
# The script finished after ~58 minutes on submit and the score was 0.0095693.
# This is a little bit better than the public LinearRegression solution.
# This technique could not increase the score of the public RidgeRegression solution.

import kagglegym
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

data = pd.read_hdf("../input/train.h5")
# fill N/A by mean of every ID
data = pd.concat([data.id,data.groupby('id').transform(lambda x: x.fillna(x.mean()))],axis=1)


# fill remaining N/As (shouldn't be here) with global mean
train_means = data.mean()
data.fillna(train_means, inplace=True)

y = data['y']
X = data.drop('y', axis=1)
### use clipping as shown in public script 
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

cols = ['technical_20', 'fundamental_53']
cols_with_id = ['id', 'technical_20', 'fundamental_53']

# our working dataframe for the loop
X_t = X[cols_with_id]

# train regressor
r = LinearRegression()
r.fit(X.loc[y_is_within_cut, cols], y.loc[y_is_within_cut])

while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    
    # only work on the required columns
    obs = observation.features[cols_with_id]
    
    # add the new data to the already observed data
    X_t = pd.concat([X_t, obs]).reset_index(drop=True)
    
    # only operate on the IDs of the current observation
    obs_ids = obs.id.unique()
    
    # fill N/As by using existing info about ID
    a = X_t.values
    ID = a[:,0].astype(int)
    valid_mask = np.in1d(ID, obs_ids)
    for i in range(1,X_t.shape[1]):
        nan_mask = np.isnan(a[:,i])
        m1 = ~nan_mask & valid_mask
        m2 = nan_mask & valid_mask
        valid_ids = np.intersect1d(ID[m1],ID[m2])
        valid_mask_column = np.in1d(ID,valid_ids)
        m2 = nan_mask & valid_mask_column
        s1 = np.bincount(ID[m1],a[m1,i])/np.bincount(ID[m1]) 
        X_t.iloc[m2,i] = s1[ID[m2],None]
        
    # there could still be some N/As left
    # these N/As are from new IDs that weren't observed before
    # in this case we use the current overall mean
    X_t_means = X_t.mean()
    X_t.fillna(X_t_means, inplace=True)
    
    # predict only for the new rows
    p = r.predict(X_t[cols].tail(len(obs.index))).clip(low_y_cut, high_y_cut)

    target['y'] = p
    
    if timestamp % 10 == 0:
        print("Timestamp #{} Size of data {}".format(timestamp, len(X_t.index)))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break