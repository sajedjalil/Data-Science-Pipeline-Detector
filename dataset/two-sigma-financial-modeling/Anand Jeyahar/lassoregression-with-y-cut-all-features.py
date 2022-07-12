import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

cols_to_use = ['technical_30', 'technical_20', 'technical_13', 'technical_44', 'y']
# Get the train dataframe
train = observation.train[cols_to_use]

mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

#cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)




# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

model = lm.Lasso()
model.fit(train[y_is_within_cut].as_matrix(), train.loc[y_is_within_cut, 'y'])

model1 = lm.Lasso()
model1.fit(train[y_is_above_cut].as_matrix(), train.loc[y_is_above_cut, 'y'])

model2 = lm.Lasso()
model2.fit(train[y_is_below_cut].as_matrix(), train.loc[y_is_below_cut, 'y'])

num_records = len(train)
weights = [len(train[y_is_below_cut])/num_records, len(train[y_is_within_cut])/num_records, len(train[y_is_above_cut])/num_records]


while True:
    observation.features.fillna(mean_values, inplace=True)
    print(weights)
    #observation.features.fillna(np.mean, inplace=True)
    test_x = observation.features.as_matrix()
    
    observation.target.y = weights[0]*model1.predict(test_x) + weights[1] * model.predict(test_x) + weights[2] * model2.predict(test_x)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
    
info