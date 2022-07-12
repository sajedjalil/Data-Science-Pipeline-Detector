# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
import copy
import matplotlib.pyplot as plt
import kagglegym

# Create environment
env = kagglegym.make()
# Get first observation
observation = env.reset()
pos_of_selected_fatures = np.array([8, 26, 37, 67])
names = np.array(observation.train.columns)
cols_to_use = ['technical_30', 'technical_33', 'technical_41', 'technical_20',
       'technical_24', 'timestamp', 'technical_3', 'technical_1',
       'technical_5', 'technical_31', 'technical_25', 'technical_44',
       'technical_28', 'technical_13']
selected_features = cols_to_use[:4]#names[pos_of_selected_fatures]
print('Number of selected features: {}'.format(len(selected_features)))

avg = np.mean(observation.train)
observation.train.fillna(avg, inplace=True)
print("Done filling empty")

#model = ExtraTreesRegressor(n_estimators=25, max_depth=4, n_jobs=-1)
model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
model.fit(np.array(observation.train[selected_features]), np.array(observation.train['y']))
print(model.coef_, model.alpha_)

targets = dict()
features = dict()
first = True
features = dict()
def run_episode(env, model, selected_features):
    observation = env.reset()
    totalreward = 0
    counter = 0
    info = None
    while True:
        # env.render()
        # action = 0 if np.matmul(parameters,observation) < 0 else 1
        
        timestamp = observation.features["timestamp"][0]
        if timestamp not in features:
            features[timestamp] = observation.features.fillna(avg, inplace=True)
        else:
            observation.features = features[timestamp]
        #features = observation.features[selected_features].fillna(avg, inplace=True)
        observation.target['y'] = model.predict(observation.features[selected_features])#np.matmul(features, parameters)
        target = observation.target

        if timestamp % 100 == 0:
            print("Timestamp #{}".format(timestamp))
        
        observation, reward, done, info = env.step(target)            
        totalreward += reward
        counter += 1
        if done:
            break
    return totalreward, info

def train(env, model, selected_features):

    episodes_per_update = 2
    noise_scaling = 0.0001
    bestreward, bestinfo = run_episode(env, model, selected_features)
    print('baseline reward: %f, baseline score: %f' % (bestreward, bestinfo['public_score']))
    counter = 0

    for i in range(100):
        counter += 1
        newmodel = copy.copy(model)
        newmodel.coef_ = model.coef_ + (np.random.rand(len(model.coef_)) * 2 - 1)*noise_scaling
        # print newparams
        # reward = 0
        # for _ in xrange(episodes_per_update):
        #     run = run_episode(env,newparams)
        #     reward += run
        reward, info = run_episode(env,newmodel, selected_features)
        print("%d: reward %f best %f score %f" % (i, reward, bestreward, info['public_score']))
        if info['public_score'] > bestinfo['public_score']:
            # print "update"
            bestinfo = info
            model = newmodel
            if reward == 200:
                break
        #print("Episode", i, "done,", "reward:", reward)

#baseline reward: -117.535478, baseline score: 0.018820
train(env, model, selected_features)

