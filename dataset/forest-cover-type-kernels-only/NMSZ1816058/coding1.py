# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

 # linear alg # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#数据清洗
train = pandas.read_csv('../input/train.csv')
r1=list(range(11,15))
r2=list(range(11,51))
name = []
for label in train.head(1):
    name.append(label)
wild=name[11:15]
soil=name[15:55]
train.insert(56,'Wildness',0)
train.insert(57,'Soil_Type',0)
for a in range(4):
    train['Wildness']=train['Wildness']+(a+1)*train[wild[a]]
for b in range(40):
    train['Soil_Type']=train['Soil_Type']+(b+1)*train[soil[b]]
train.drop(train.columns[r1],axis=1,inplace=True)
train.drop(train.columns[r2],axis=1,inplace=True)
train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology
train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2
#训练开始
predictors=[]
for label in train.head(1):
    predictors.append(label)
predictors.remove('Id')
predictors.remove('Cover_Type')
model = RandomForestClassifier(n_estimators=300,oob_score=True)
train_predictors = train[predictors]
train_target = train["Cover_Type"]
train_X, test_X, train_y, test_y = train_test_split(train_predictors, train_target,
     random_state=6)
model.fit(train_X, train_y)
predictions =model.predict(test_X)
print('accuracy:',accuracy_score(test_y, predictions))
test = pandas.read_csv('../input/test.csv')
r1=list(range(11,15))
r2=list(range(11,51))
name = []
for label in test.head(1):
    name.append(label)
wild=name[11:15]
soil=name[15:55]
test.insert(55,'Wildness',0)
test.insert(56,'Soil_Type',0)
for a in range(4):
    test['Wildness']=test['Wildness']+(a+1)*test[wild[a]]
for b in range(40):
    test['Soil_Type']=test['Soil_Type']+(b+1)*test[soil[b]]
test.drop(test.columns[r1],axis=1,inplace=True)
test.drop(test.columns[r2],axis=1,inplace=True)
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2
predictors=[]
for label in test.head(1):
    predictors.append(label)
predictors.remove('Id')
predictions = model.predict(test[predictors])
sub = pandas.DataFrame({"Id": test['Id'],"Cover_Type": predictions})
sub.to_csv("submission.csv", index=False)