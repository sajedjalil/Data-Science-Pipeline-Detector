import pandas as pd
import time
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

params = {  
    "n_estimators": [800,1000],
    "max_depth": [3,5,7],
    "learning_rate": [0.05,0.1,0.4],
    "colsample_bytree": [0.4],
    "subsample": [0.4],
    "gamma": [1,2,3,4,5,6,7,10],
    'reg_alpha': [5],
    "min_child_weight": [5,7],
}

train=pd.read_csv('../input/train_V2.csv')
test=pd.read_csv('../input/test_V2.csv')
sub=pd.read_csv('../input/sample_submission_V2.csv')

#Just for faster running
train=train.ix[0:10000]
t=time.localtime()
print(time.asctime(t))
#Show the columns of Data
#print(train.columns)
train=train.dropna(axis=0)
test=test.fillna(0)

mathtype2num={'squad-fpp':0,'duo':1,'solo-fpp':2,'squad':3,'duo-fpp':4}
train['matchType']=train['matchType'].map(mathtype2num)
mathtype2num={'squad-fpp':0,'duo':1,'solo-fpp':2,'squad':3,'duo-fpp':4}
test['matchType']=test['matchType'].map(mathtype2num)

col=['assists', 'boosts','damageDealt', 'DBNOs', 'headshotKills', 'heals','matchType','killPlace','killPoints', 'kills', 'killStreaks', 'matchDuration', 'maxPlace', 'numGroups', 'rankPoints', 'revives','rideDistance', 'roadKills', 'swimDistance', 'teamKills','vehicleDestroys', 'walkDistance', 'weaponsAcquired','winPoints']
train2=train[col]
test2=test[col]

xgbmodel = XGBRegressor(nthreads=-1)  
RSCxgb = RandomizedSearchCV(xgbmodel, params, n_jobs=2,cv=10)

model=RSCxgb.fit(train2,train['winPlacePerc']) 

t=time.localtime()
print(time.asctime(t))

result=model.predict(test2)
sub['winPlacePerc']=result
sub.to_csv('submission.csv',index=False)

