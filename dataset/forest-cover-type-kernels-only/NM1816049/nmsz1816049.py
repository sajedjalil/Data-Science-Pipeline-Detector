# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/train.csv',  encoding='gbk')
test = pd.read_csv('../input/test.csv', encoding='gbk')
train.drop(['Soil_Type15', "Soil_Type7"], inplace=True, axis=1)
test.drop(['Soil_Type15', "Soil_Type7"], inplace=True, axis=1)

# split data to x and y
train_y= train["Cover_Type"]
train.drop(["Cover_Type"], inplace=True, axis=1)
train_X = train.iloc[:, 1:]

model = RandomForestClassifier(n_estimators = 50, random_state = 53)
model = model.fit(train_X, train_y)

test_X =  test.iloc[:, 1:]

predictions = model.predict(test_X)

my_submission = pd.DataFrame({'Id':test.Id,'Cover_Type':predictions})
my_submission.to_csv('submission.csv', index = False)