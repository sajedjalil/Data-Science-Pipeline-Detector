# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score #用于kFold计算分数
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm #引入svm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv("../input/train.csv", header=0)
test = pd.read_csv("../input/test.csv", header=0)
train.describe()
train.head(6)
feature = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
           'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
           'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
           'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
           'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
           'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
           'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'] #选择特征

#留出法，先分割数据
#x_train, x_test, y_train, y_test = train_test_split(train[feature],train['Cover_Type'], test_size=0.25, random_state=0);
#clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
#clf.fit(x_train, y_train)
#print(clf.score(x_test, y_test)) #留出法的成绩,0.845

clf2 = svm.SVC(gamma='scale', decision_function_shape='ovr') #7个类，对应7个分类器svm,ovr
clf2.fit(train[feature], train['Cover_Type'])
scores = cross_val_score(clf2, train[feature], train['Cover_Type'], cv=4)
print("kFold scores: ")
print("Scores: ", scores, ". Mean score is ", scores.mean())

#get the predict result
result = clf2.predict(test[feature])

sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": result})
sub.to_csv('submission.csv', index=False)