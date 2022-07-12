import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

cols_to_use = ['technical_20']

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

models_dict = {}
for col in cols_to_use:
    model = lm.LinearRegression()
    model.fit(np.array(train.loc[y_is_within_cut, col].values).reshape(-1,1), train.loc[y_is_within_cut, target])
    models_dict[col] = model

avg = observation.train[['id', 'y']].groupby(['id']).mean()
#avg['id_avg'] = avg.index
avg.reset_index(inplace=True) # index => id
avg.columns = ['id','mean_y']
print(avg) # index id, column y
print(len(avg))
avg_all = observation.train['y'].mean()
print(avg_all)

print(observation.target)

col = 'technical_20'
model = models_dict[col]

while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape((-1,1))
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    observation.target = observation.target.merge(avg,how="left",on="id")
    #print(set(observation.target.id)-set(avg.id))
    observation.target.mean_y.fillna(avg_all, inplace=True)
    observation.target["y"]=observation.target["y"]*0.95 + observation.target["mean_y"]*0.05
    observation.target.drop("mean_y",1) 
    print(observation.target.isnull().sum())
    if observation.target.isnull().sum().sum()>0:
        print(observation.target)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)

