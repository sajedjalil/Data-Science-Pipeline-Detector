# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from math import sqrt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

def conpute_euclid_distance(data):
    data['Euclid_Distance_To_Hydrology'] = data.apply(lambda x: abs(sqrt(x['Horizontal_Distance_To_Hydrology']**2 + x['Vertical_Distance_To_Hydrology']**2)), axis=1)
    return data

def get_features(data):
    return data.columns[1:55]\
        .append(train_data.columns[56:57])\
        .drop(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Soil_Type15', 'Soil_Type7'])
        
train_data = conpute_euclid_distance(pd.read_csv("../input/train.csv"))
features = get_features(train_data)
y = np.array(train_data['Cover_Type'])

forest = RandomForestClassifier()
params = { 'n_jobs': [1, 4], 'n_estimators': [280, 350], 'max_features': [0.5, 0.9], 'max_depth': [50, 80] }

forest = GridSearchCV(forest, param_grid = params, cv = 4)
forest.fit(train_data[features], y)

better_forest = RandomForestClassifier(n_estimators = forest.best_params_['n_estimators'], max_features = forest.best_params_['max_features'], max_depth = forest.best_params_['max_depth'], n_jobs = forest.best_params_['n_jobs'])

better_forest.fit(train_data[features], y)

real_test_data = conpute_euclid_distance(pd.read_csv("../input/test.csv"))
result_y = better_forest.predict(real_test_data[features])

result = pd.DataFrame({"Id": real_test_data.iloc[:,0].values, "Cover_Type": result_y}, columns=['Id', 'Cover_Type'])
result.to_csv("submission.csv", index=False)

# Any results you write to the current directory are saved as output.