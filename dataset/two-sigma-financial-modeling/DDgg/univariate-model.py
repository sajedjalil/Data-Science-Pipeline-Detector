import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

## Best predictors for y
## Value        F-Value
## technical_27	    20.7233678
## technical_20	    18.4858663
## technical_19	    16.6676897
## fundamental_51	16.0119455
## fundamental_2	14.7703224
## fundamental_18	14.7703224
## fundamental_11	14.7703224
## technical_30	    12.6149954
## technical_2	    12.4383649

# based on absolute x-y correlation top 42
cols_to_use = ['technical_30','fundamental_50','fundamental_39','fundamental_23','fundamental_8','fundamental_54','fundamental_44','fundamental_14','fundamental_43','fundamental_37','technical_21','fundamental_35','fundamental_13','fundamental_55','fundamental_28','derived_0','fundamental_15','fundamental_31','technical_7','fundamental_51','fundamental_46','technical_42','fundamental_29','fundamental_25','derived_4','technical_27', 'technical_20', 'technical_19', 'fundamental_51','fundamental_2','fundamental_18','fundamental_11']


models_dict = {}
for col in cols_to_use:
    model = lm.LinearRegression()
    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)
    models_dict[col] = model

col =  ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
model = models_dict[col]
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
info


