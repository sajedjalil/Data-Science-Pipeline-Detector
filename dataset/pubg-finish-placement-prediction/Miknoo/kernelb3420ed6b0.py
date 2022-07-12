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

selected_feature=['assists', 'boosts', 'damageDealt', 'DBNOs','headshotKills', 'heals', 'killPlace', 'killPoints', 'kills','killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 'roadKills', 'swimDistance', 'teamKills','vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']
result_feature=['winPlacePerc']

trainxy=pd.read_csv("../input/train.csv")
# print(trainxy.columns)
x_test=pd.read_csv("../input/test.csv")
x_test=x_test[selected_feature]
x_test=np.array(x_test)
print("x_test",x_test)
print(x_test.shape[0])
y_test=pd.read_csv("../input/sample_submission.csv")
# y_test=y_test[result_feature]
print("y_test",y_test)
x_train=trainxy[selected_feature]
x_train=np.array(x_train)
print("x_train",x_train)
print(x_train.shape[0])
y_train=trainxy[result_feature]
y_train=np.array(y_train)
y_train=y_train.reshape(-1)
print("y_train",y_train)
print(y_train.shape[0])
#以上数据导入初始化完成

from sklearn.preprocessing import StandardScaler
print("StandardScaler")
SS=StandardScaler()
x_train=SS.fit_transform(x_train)
print("x_train",x_train)
# SSS=StandardScaler()
x_test = np.array(x_test)   ########
x_test = SS.transform(x_test)
# x_test = x_test[x_test.columns[:-1]]
print("x_test",x_test)

# SSS=StandardScaler()
# y_train = np.array(y_train)
# y_train=y_train.reshape(-1,1)
# print("y_train")
# y_train = SSS.fit_transform(y_train)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
print("RandomForestRegressor fit")
y_predict = rfr.predict(x_test)
print("y_predict")
# print("random forestry score：", rfr.score(x_test, y_test))

y_test["winPlacePerc"]=y_predict
print("new  y_test")
y_test.to_csv("sample_submission.csv",index=False)
print("data go to submission.csv")