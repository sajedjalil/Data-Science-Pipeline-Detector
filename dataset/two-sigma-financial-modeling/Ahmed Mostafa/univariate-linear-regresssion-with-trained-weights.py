import kagglegym
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

env = kagglegym.make()
ob = env.reset()
train = ob.train
train.columns
def rename_columns(data_df):
    data_df.columns = [x[0]+x[(x.index('_')+1):] if '_' in x else x for x in data_df.columns]

rename_columns(train)

#low_y_cut = -0.086093  # y.min(): -0.0860941
#high_y_cut = 0.093497  # y.max():  0.0934978
#train.query('@low_y_cut <= y <= @high_y_cut', inplace=True)

average_values = train.median(axis=0)

cols_2 = ['t20']
m2 = LinearRegression(fit_intercept=True, normalize=True)
m2.coef_ = np.array([-0.07811666701836258997])
m2.intercept_ = 0.00028078416368078080

while True:
    target = ob.target
    timestamp = ob.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
    features = ob.features
    rename_columns(features)
    
    features.fillna(features.median(axis=0), inplace=True)
    features.fillna(average_values, inplace=True)
    
    y2 = m2.predict(X=features[cols_2])
    target.y=y2
    #target.y = y2.clip(low_y_cut, high_y_cut)
    
    ob, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break