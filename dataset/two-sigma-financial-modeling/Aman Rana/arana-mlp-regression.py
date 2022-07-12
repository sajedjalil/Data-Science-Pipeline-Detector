import kagglegym
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

names = np.array(observation.train.columns)
names_fea = np.array(observation.features.columns)

# Varbose parameter
_verbose = True

train = observation.train
# tr = train.dropna()
avg = np.mean(train)
train.fillna(avg, inplace=True)

# selected_features = ['technical_30', 'technical_20', 'fundamental_11', 'technical_27']
# print('Number of selected features: {}'.format(len(selected_features)))

if _verbose:
    print('Starting training...')

## Training
model = Ridge()
model.fit(np.array(train[names[2:-1]]), np.array(train['y']))

if _verbose:
    print('Model trained')
    print('Starting prediction')

## Predicting
while True:
    try:
        features = observation.features
        # fe = features.dropna()
        avg2 = np.mean(features)
        features.fillna(avg2, inplace=True)
        target = observation.target
    
        target['y'] = model.predict(np.array(features[names[2:-1]]))
        
        timestamp = observation.features["timestamp"][0]
        if timestamp % 100 == 0:
            print("Timestamp #{}".format(timestamp))

        # We perform a "step" by making our prediction and getting back an updated "observation":
        observation, reward, done, info = env.step(target)
    
    except:
        continue
    
    
    if done:
        print("Public score: {}".format(info["public_score"]))
        break




