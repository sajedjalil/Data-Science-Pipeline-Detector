import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp
import time

from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier

def logloss(label,pred):
    epsilon = 1e-15;
    pred = sp.maximum(epsilon,pred);
    #print pred;
    pred = sp.minimum(1-epsilon,pred);
    #print pred;
    loss = sum(label*sp.log(pred)+sp.subtract(1,label)*sp.log(sp.subtract(1,pred)));
    loss = loss * -1.0/len(label)
    return loss

filename= "../input/data.csv"
data = pd.read_csv(filename)

data['dist'] = np.sqrt(data['loc_x']**2+data['loc_y']**2)
loc_x_zero = (data['loc_x'] == 0);
data['angle'] = np.array([0]*len(data))
data['angle'][~loc_x_zero] = np.arctan(data['loc_y'][~loc_x_zero]/data['loc_x'][~loc_x_zero])
data['angle'][loc_x_zero] = np.pi/2
data['remaining_time'] = data['minutes_remaining']*60+data['seconds_remaining']
data['season'] = data['season'].apply(lambda row:int(row.split('-')[1]))

test_id = data[pd.isnull(data['shot_made_flag'])]['shot_id']

drops = ['shot_zone_area','shot_zone_basic','shot_zone_range','shot_id','team_id','team_name','matchup','game_date']
for drop in drops:
    data = data.drop(drop,1)

#category_val = ['action_type','combined_shot_type','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range','opponent']
category_val = ['action_type','combined_shot_type','shot_type','period','season','opponent']

for var in category_val:
    data = pd.concat([data,pd.get_dummies(data[var],prefix=var)],1);
    data = data.drop(var,1);

train = data[pd.notnull(data['shot_made_flag'])]
test = data[pd.isnull(data['shot_made_flag'])]

train_label = train['shot_made_flag']
train = train.drop('shot_made_flag',1);
test = test.drop('shot_made_flag',1);

print ('find the best n for RandomForest')
min_score = 100000
best_n = 0
"""scores_n = []
#range_n = np.logspace(0,2,num = 3).astype(int)
range_n = [500]
print (range_n)
for n in range_n:
    print ('the number of trees:{0}'.format(n))
    t1 = time.time();
    score = 0;
    model = RandomForestClassifier(n_estimators = n)
    #print model
    for train_k,test_k in KFold(len(train),n_folds=10,shuffle=True):
        #print type(train_k)
        #print type(test_k)
        #print train.iloc[test_k]
        #print train.iloc[train_k].head()
        model.fit(train.iloc[train_k],train_label.iloc[train_k])
        
        pred = model.predict(train.iloc[test_k])
        score += logloss(train_label.iloc[test_k].values,pred)/10;
    scores_n.append(score);
    if score < min_score:
        min_score = score;
        best_n = n;
    t2 = time.time()
    print ('Done processing {0} trees({1:.3f} sec)'.format(n,t2-t1))

print (best_n,min_score)

print ('find the best max_depth for RandomForest')
min_score = 100000
best_m = 0
scores_m = []
#range_m = np.logspace(0,2,num = 3).astype(int)
range_m = [10]
print (range_m)
for m in range_m:
    print ('the number of trees:{0}'.format(m))
    t1 = time.time();
    score = 0;
    model = RandomForestClassifier(max_depth = m,n_estimators = best_n)
    for train_k,test_k in KFold(len(train),n_folds=10,shuffle=True):
        model.fit(train.iloc[train_k],train_label.iloc[train_k])
        pred = model.predict(train.iloc[test_k])
        score += logloss(train_label.iloc[test_k],pred)/10;
    scores_m.append(score);
    if score < min_score:
        min_score = score;
        best_m = m;
    t2 = time.time()
    print('Done processing {0} trees({1:.3f} sec)'.format(m,t2-t1))

print (best_m,min_score)
"""
best_n = 500
best_m = 10
model = GradientBoostingClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train, train_label)
pred = model.predict_proba(test)


final = pd.DataFrame(columns=['shot_id','shot_made_flag'])
final['shot_id'] = test_id
final['shot_made_flag'] = pred[:,1]
print (final.head())
final.to_csv('submission.csv',index=False);

