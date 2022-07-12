import numpy as np 
import pandas as pd 

import scipy as sp
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data=pd.read_csv("../input/data.csv")
data.lon.unique().shape

data_x=pd.get_dummies(data.action_type,prefix="action_type")
cols=["combined_shot_type","game_event_id","period","playoffs",
      "shot_type","shot_zone_area","shot_zone_basic","shot_zone_range",
      "matchup","opponent","game_date","shot_distance","minutes_remaining","seconds_remaining",
      "loc_x","loc_y"]
for col in cols:
    data_x=pd.concat([data_x,pd.get_dummies(data[col],prefix=col),],axis=1)
train_x=data_x[-pd.isnull(data.shot_made_flag)]
test_x=data_x[pd.isnull(data.shot_made_flag)]
train_y=data.shot_made_flag[-pd.isnull(data.shot_made_flag)]

clf = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=550,
                     subsample=0.5, colsample_bytree=0.5, seed=0)
clf.fit(train_x, train_y)
y_pred = clf.predict(train_x)
print("Number of mislabeled points out of a total %d points : %d"  % (train_x.shape[0],(train_y != y_pred).sum()))

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    print(ll)
    return ll
    
logloss(train_y,clf.predict_proba(train_x)[:,1])

test_y=clf.predict_proba(test_x)[:,1]
test_id=data[pd.isnull(data.shot_made_flag)]["shot_id"]
submission=pd.DataFrame({"shot_id":test_id,"shot_made_flag":test_y})
submission.to_csv("submissson_1.csv",index=False)
