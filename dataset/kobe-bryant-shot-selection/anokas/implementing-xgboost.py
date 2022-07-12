import numpy as np
import pandas as pd
import xgboost as xgb

data=pd.read_csv("../input/data.csv")


data_x=pd.get_dummies(data.action_type,prefix="action_type")
cols=["combined_shot_type","game_event_id","period","playoffs",
      "shot_type","shot_zone_area","shot_zone_basic","shot_zone_range",
      "matchup","opponent","game_date","shot_distance","minutes_remaining","seconds_remaining","loc_x","loc_y"]
for col in cols:
    data_x=pd.concat([data_x,pd.get_dummies(data[col],prefix=col),],axis=1)

x_train=data_x[-pd.isnull(data.shot_made_flag)]
x_test=data_x[pd.isnull(data.shot_made_flag)]
y_train=data.shot_made_flag[-pd.isnull(data.shot_made_flag)].values

print(y_train)

d_train = xgb.DMatrix(x_train, label=y_train)

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['max_depth'] = 6
params['silent'] = 1


# Optimisation
i = 5
while i<9:
    print(i)
    params['max_depth'] = i
    cvp = xgb.cv(params, d_train, num_boost_round=100000, early_stopping_rounds=10, metrics=['logloss'], verbose_eval=10, stratified=True)
    i = i+1