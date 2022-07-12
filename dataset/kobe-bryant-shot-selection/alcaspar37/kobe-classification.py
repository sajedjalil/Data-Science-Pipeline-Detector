# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/data.csv')
train.dropna(inplace=True)
#print(train.corr())
train.drop(['loc_x','loc_y','lat','period','shot_zone_range','matchup','season','team_name','game_date','shot_type'],axis=1,inplace=True)

shots=train['combined_shot_type'].unique()
areas=train['shot_zone_area'].unique()
actions=train['action_type'].unique()
zones=train['shot_zone_basic'].unique()
opps=train['opponent'].unique()
train['combined_shot_type'].replace(to_replace=shots,value=list(range(len(shots))),inplace=True)
train['shot_zone_area'].replace(to_replace=areas,value=list(range(len(areas))),inplace=True)
train['action_type'].replace(to_replace=actions,value=list(range(len(actions))),inplace=True)
train['shot_zone_basic'].replace(to_replace=zones,value=list(range(len(zones))),inplace=True)
train['opponent'].replace(to_replace=opps,value=list(range(len(opps))),inplace=True)

#print(train.describe())
y_train=train['shot_made_flag']
x_train=train.drop(['shot_made_flag'],axis=1)
print(x_train.describe())
print(y_train.describe())
#print(x_train[x_train.isin(['IND'])])
#corrMatrix=train.corr()

model=RandomForestClassifier()
model.fit(x_train,y_train)
score=cross_val_score(model,x_train,y_train)
print(score)

model=ExtraTreesClassifier()
model.fit(x_train,y_train)
score=cross_val_score(model,x_train,y_train)
print(score)
#remove nan examples, look at variable distribution
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.