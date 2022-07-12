import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import math
from scipy.stats import ttest_ind

def compute_increment_physics(
    reward_hist, increment_hist, verbose,
    len_history_fill, rand_width,
    increment_mean_shift_margin, p_thresh, diff_multiplier
):
    if len(increment_hist)<15:
        verbose = False
    if len(reward_hist)==0:
        return 0
    if len(reward_hist)<2*len_history_fill:
        return np.random.normal(0, rand_width)
    else:
        last_x_increments = np.array(increment_hist)[-len_history_fill:]
        last_x_rewards = np.array(reward_hist)[-len_history_fill:]
        last_x_rewards = (
                last_x_rewards - np.min(last_x_rewards)
            )/(np.max(last_x_rewards) - np.min(last_x_rewards))
        total_mass = np.sum(last_x_rewards)
        
        last_x_massed_increments = []
        for k in range(len(last_x_increments)):
            last_x_massed_increments.append(last_x_increments[k]*\
                (last_x_rewards[k]/total_mass)*len(last_x_increments))
        last_x_massed_increments = np.array(last_x_massed_increments)
        
        center_of_mass = np.mean(last_x_massed_increments)
        center_of_geometry = np.mean(last_x_increments)
        
        
        t_stat, p_value = ttest_ind(last_x_massed_increments, last_x_increments)
        if verbose:
            print ('-'*20)
            print ('center_of_mass = %f (*pow(10, 6))'%(center_of_mass*pow(10, 6)))
            print ('center_of_geometry = %f (*pow(10, 6))'%(center_of_geometry*pow(10, 6)))
            print ('p_value = %f'%(p_value))
            print (last_x_increments)
            print (last_x_massed_increments)
        
        reweighter = diff_multiplier*(center_of_mass - center_of_geometry)
        if abs(reweighter) > increment_mean_shift_margin:
            reweighter = increment_mean_shift_margin*np.sign(reweighter)
        next_increment = reweighter + center_of_geometry
        
        next_increment = np.random.normal(
            next_increment, rand_width
        )
        
        if p_value > p_thresh or np.isnan(p_value):
            next_increment = np.random.normal(
                0, rand_width
            )
        if verbose:
            print (
                'increment = %f (*pow(10, 6))'%(
                    next_increment*pow(10, 6)
                )
            )
        return next_increment

env = kagglegym.make()
observation = env.reset()
excl_cols = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
all_feature_cols = [c for c in observation.train.columns if c not in excl_cols]


train = pd.read_hdf('../input/train.h5')
train = train[all_feature_cols]
d_mean= train.median(axis=0)

train = observation.train[all_feature_cols]
n_nulls = train.isnull().sum(axis=1)

for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['null_counter'] = n_nulls

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)  #308537
model1 = rfr.fit(train, observation.train['y'])


low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (observation.train.y > high_y_cut)
y_is_below_cut = (observation.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

train_linear_models = train[y_is_within_cut]
y_linear_models = observation.train.y[y_is_within_cut]

cols_to_use = ['technical_20']

model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(train_linear_models[cols_to_use]), y_linear_models)
train = []

ymedian_dict = dict(observation.train.groupby(["id"])["y"].median())


verbose = False
np.random.seed(0)

reward_hist = []
increment_hist = []
original_predicted_hist = []
final_predicted_hist = []
increment = 0

i = 0
while True:
    test_all_features = observation.features[all_feature_cols]
    null_counter = test_all_features.isnull().sum(axis=1)
    for c in test_all_features.columns:
        test_all_features[c + '_nan_'] = pd.isnull(test_all_features[c])
    test_all_features = test_all_features.fillna(d_mean)
    test_all_features['znull'] = null_counter
 
    observation.features.fillna(d_mean, inplace=True)
    test_linear_model = np.array(observation.features[cols_to_use])

    pred = observation.target
    pred['y'] = 0.65*model1.predict(test_all_features).clip(low_y_cut, high_y_cut) + 0.35*model2.predict(test_linear_model).clip(low_y_cut, high_y_cut)
    pred['y'] = pred['y'] + increment
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymedian_dict[r['id']] if r['id'] in ymedian_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.8f')) for x in pred['y']]
    
    
    observation, reward, done, info = env.step(pred[['id','y']])
    
    reward_hist.append(reward)
    increment_hist.append(increment)
    
    increment = compute_increment_physics(reward_hist, increment_hist, verbose = verbose, 
        len_history_fill = 6, rand_width = 3*pow(10,-6), 
        increment_mean_shift_margin = 60*pow(10,-6), p_thresh = 1.0, diff_multiplier = 6
    )
    
    if i % 100 == 0:
        print(reward)
    i += 1
    if done:
        print(info["public_score"])
        break
