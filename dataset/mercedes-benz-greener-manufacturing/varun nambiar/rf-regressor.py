# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn import model_selection , preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


for f in train.columns:
    if train[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))
        
for f in test.columns:
    if test[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test[f].values))
        test[f] = lbl.transform(list(test[f].values))     

id_test = test.ID
x_target = train['y']
x_train = train.drop(['y','ID'],axis=1)
x_test = test.drop(['ID'],axis=1)


from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor()
param_grid = { 
            "n_estimators"      : [300],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }

grid = model_selection.GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)

model = grid.fit(x_train, x_target)

pred = model.predict(x_test)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = pred
sub.to_csv('randomforest.csv', index=False)