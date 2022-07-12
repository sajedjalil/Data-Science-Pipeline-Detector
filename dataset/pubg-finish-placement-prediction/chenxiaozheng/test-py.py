# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import gc

import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
# Input data file

def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)


def dataeng(train):
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    match = train.groupby('matchId')
    train['killPlacePerc'] = match['kills'].rank(pct=True).values
    train['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
    train['killsNorm'] = train['kills'] * ((100 - train['playersJoined']) / 100 + 1)
    train['damageDealtNorm'] = train['damageDealt'] * ((100 - train['playersJoined']) / 100 + 1)
    train['maxPlaceNorm'] = train['maxPlace'] * ((100 - train['playersJoined']) / 100 + 1)
    train['matchDurationNorm'] = train['matchDuration'] * ((100 - train['playersJoined']) / 100 + 1)
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
    train['healthItems'] = train['heals'] + train['boosts']
    train['headshotKillRate'] = train['headshotKills'] / train['kills']
    train['killPlaceOverMaxPlace'] = train['killPlace'] / train['maxPlace']
    train['killsOverWalkDistance'] = train['kills'] / train['walkDistance']
    train['killStreakRate'] = train['killStreaks']/train['kills']
    train['killMinute'] = train['kills'] / train['matchDuration']
    train['damageDealtMinute'] = train['damageDealt'] / train['matchDuration']
    train['participateKills'] = train['kills'] + train['assists'] + train['DBNOs']
    train['vehicleDestroysMinute'] = train['vehicleDestroys'] / train['matchDuration']
    train['killsMiter'] = train['roadKills'] / train['rideDistance']
    fillInf(train, 0)
    # train['killStreakrate'] = train['killStreaks'] / train['killsNorm']
    return train


debug = False
if debug == True:
    train = pd.read_csv('../input/train_V2.csv')
    test = pd.read_csv('../input/test_V2.csv')
else:
    train = pd.read_csv('../input/train_V2.csv')
    test = pd.read_csv('../input/test_V2.csv')

test_id = test["Id"]
train=dataeng(train)
test=dataeng(test)
train.drop(2744604, inplace=True)   # 2744604 is NaN
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
train.drop(["matchType", "killsWithoutMoving", "rankPoints"], axis =1, inplace=True)
test.drop(["matchType", "rankPoints"], axis =1, inplace=True)
train.dropna(inplace = True)
y = train["winPlacePerc"]
X = train.drop(["Id", "groupId", "matchId", "winPlacePerc","damageDealt","kills","maxPlace","matchDuration", 'headshotKills', 'roadKills', 'vehicleDestroys', 'killPoints', 'winPoints'], axis = 1)
test = test.drop(["Id", "groupId", "matchId","damageDealt","kills","maxPlace","matchDuration", 'headshotKills', 'roadKills', 'vehicleDestroys', 'killPoints', 'winPoints'], axis = 1)
print('data prepared')
gbd = GradientBoostingRegressor(learning_rate=0.1, random_state=10)
gbd.fit(X, y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.1,random_state=1)
# X_second_train, X_second_test,Y_second_train, Y_second_test = train_test_split(X_train, Y_train, test_size=0.2,random_state=1)
# # X_three_train, X_three_test, Y_three_train, Y_three_test = train_test_split()

# param_test1 = {'n_estimators':range(20,81,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=300,min_samples_leaf=20,max_depth=8, subsample=0.8,random_state=10), param_grid = param_test1, scoring='neg_mean_squared_error',iid=False,cv=5)
# gsearch1.fit(X_test, Y_test)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
# gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=300,min_samples_leaf=20,max_depth=8, subsample=0.8,random_state=10), param_grid = param_test1, scoring='neg_mean_squared_error',iid=False,cv=5)
# gsearch2.fit(X_second_train, Y_second_train)
# print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
# gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=300,min_samples_leaf=20,max_depth=8, subsample=0.8,random_state=10), param_grid = param_test1, scoring='neg_mean_squared_error',iid=False,cv=5)
# gsearch3.fit(X_second_test, Y_second_test)
# print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
# model = xgboost.XGBRegressor(booster='gblinear',learning_rate=0.01,n_estimator=100,max_depth=20,min_child_weight=2, gamma=0, subsample=0.8,colsample_bytree=0.8,verbose=10,seed=27,n_job=-1)
# linerReg = LinearRegression()
# alphas_to_test = np.linspace(0.0001, 500)
# rcv = RidgeCV(alphas=alphas_to_test, store_cv_values=True)
# rcv = RidgeCV(alphas=np.array([.9,1.0, 5.0, 10.0, 30.0, 50.0, 90.0]))
# rcv = Ridge(alpha=132.65)
# rcv.fit(X, y)
# smallest_idx = rcv.cv_values_.mean(axis=0).argmin()
# alphas_to_test[smallest_idx]
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=1)
# linerReg.fit(X, y)
# print(rcv.alpha_)
# print (linerReg.coef_)
# print(linerReg.intercept_)
print('fitted')
# xgboost.plot_importance(model)
# pred  = rcv.predict(test)
pred = gbd.predict(test)
from sklearn import metrics

# print ("MSE:",metrics.mean_squared_error(Y_test, pred))
data = pd.read_csv("../input/test_V2.csv")

result = pd.DataFrame({'Id':data['Id'].as_matrix(), 'winPlacePerc':pred.astype(np.float64)})
print('finish')
result.to_csv("sample_submission_V2.csv", index=False)