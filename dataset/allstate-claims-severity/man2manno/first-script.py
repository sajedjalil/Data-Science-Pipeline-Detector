# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
target = df.loss
ID = df.id
df = df.drop('id',axis=1)


contCols = ['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14','loss']

x_cat = df.drop(contCols,axis=1)

x_nums = x_cat.apply(LabelEncoder().fit_transform)
X = pd.concat([x_nums, df[contCols[:-1]]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=11)

reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(mean_absolute_error(y_test,y_pred))
