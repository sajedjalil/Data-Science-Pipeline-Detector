import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))
  
print("Preparing data for model...")
df = observation.train

low_y_cut = -0.086092
high_y_cut = 0.093496

fundamental = ['fundamental_61', 'fundamental_54', 'fundamental_46', 'fundamental_32', 'fundamental_35', 'fundamental_28', 'fundamental_26', 'fundamental_25', 'fundamental_14', 'fundamental_1']
technical = ['technical_42', 'technical_37', 'technical_18', 'technical_10']
derived = ['derived_1', 'derived_3']
drop_list2 = ['technical_28', 'technical_43', 'technical_5', 'technical_7', 'technical_9', 'technical_32', 'technical_12', 'fundamental_63', 'fundamental_30', 'fundamental_33', 'fundamental_34', 'fundamental_12', 'fundamental_16', 'fundamental_9', 'fundamental_6']
others = ['timestamp']

drop_list = fundamental + technical + derived + drop_list2 + others

df.drop(drop_list, axis=1, inplace=True)

#df = df.sample(frac=0.01)
y_is_within_cut = ((df['y'] > low_y_cut) & (df['y'] < high_y_cut))
X = df.loc[y_is_within_cut, df.columns[:-1]]
y = df.loc[y_is_within_cut, 'y'].values.reshape(-1, 1)
xgdmat = xgb.DMatrix(X, y) 
print("Data for model: X={}, y={}".format(X.shape, y.shape))

param = {'colsample_bytree': 0.96687561114739584,
 'gamma': 38.508207482845037,
 'learning_rate': 0.30660942775795547,
 'max_delta_step': 90,
 'max_depth': 9,
 'min_child_weight': 10.520743291766465,
 'n_estimators': 24,
 'reg_alpha': 1.6370990999682116,
 'reg_lambda': 60.70480564627433,
 'subsample': 0.60593731012742447}
 
print("Fitting...")
model = xgb.train(param, xgdmat, num_boost_round = 30)
print("Fitting done")




# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

while True:
    temp_observation_features = observation.features.drop(drop_list, axis=1)
    dmat_test = xgb.DMatrix(temp_observation_features) 
    observation.target.y = model.predict(dmat_test).clip(low_y_cut, high_y_cut)
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break