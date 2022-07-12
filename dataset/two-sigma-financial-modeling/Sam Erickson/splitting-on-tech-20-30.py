import kagglegym
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

pd.options.mode.chained_assignment = None  # default='warn'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

train = observation.train
train['tech_20_30'] = train['technical_20']-train['technical_30']

# columns to use for x in model
cols = ['technical_20','technical_30','fundamental_11']

# model for the 50% of tech_20_30 vals greater than 0
train_upper_50 = train[train['tech_20_30']>0]
upper_mean = train_upper_50.mean()
train_upper_50.fillna(upper_mean,inplace=True)

upper_model = LinearRegression()
train_x = np.array(train_upper_50[cols].values)
upper_model.fit(train_x,train_upper_50['y'].values)

# model for the 50% of tech_20_30 vals <= 0. The 'cut' variables are for removing
# outliers.
low_y_cut = -0.086093
high_y_cut = 0.093497

train_lower_50 = train[(train['tech_20_30']<=0)  & (train['y']>low_y_cut) & (train['y']<high_y_cut)]

lower_mean = train_lower_50.mean()
train_lower_50.fillna(lower_mean,inplace=True)
lower_model = Ridge(alpha=0.01)
train_x = np.array(train_lower_50[cols].values)
lower_model.fit(train_x,train_lower_50['y'].values)

while True:
    
    test = observation.features
    test['tech_20_30'] = test['technical_20']-test['technical_30']
    test['y'] = test['id']
    
    upper_test = test[test['tech_20_30']>0]
    upper_test.fillna(upper_mean,inplace=True)
    test_x = np.array(upper_test[cols].values)
    upper_test.y = upper_model.predict(test_x)
    locs = upper_test.index.values
    test.loc[locs,['y']] = upper_test.loc[locs,['y']]
    
    lower_test = test[(test['tech_20_30']<=0) | (np.isnan(test['tech_20_30']))]
    lower_test.fillna(lower_mean,inplace=True)
    test_x = np.array(lower_test[cols].values)
    lower_test.y = lower_model.predict(test_x)
    locs = lower_test.index.values
    test.loc[locs,['y']] = lower_test.loc[locs,['y']]
    
    target = observation.target
    target.y = test.y
    
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break