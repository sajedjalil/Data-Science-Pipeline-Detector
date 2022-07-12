# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import mean_squared_error
from IPython.display import display # Allows the use of display() for DataFrames
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Load the wholesale customers dataset
try:
    data = pd.read_csv("../input/train.csv")
    print(data.describe())
    print("dataset has {} samples with {} features each.".format(*data.shape))
except Exception as e:
    print("Dataset could not be loaded. Is the dataset missing? ",e)
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
target = data['place_id']
target2 = data['time']
stds = data.groupby(['place_id']).std()
print(stds.head())
data = data.drop(['place_id','time','row_id'],axis=1)
# TODO: Split the data into training and testing sets using the given feature as the target
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size = 0.25, random_state = 1)

# TODO: Create a decision tree regressor and fit it to the training set
#reg = tree.DecisionTreeRegressor(random_state = 1)
#reg.fit(X_train, y_train) 
#y_pred = reg.predict(X_test)

# TODO: Report the score of the prediction using the testing set
#score = reg.score(X_test, y_test)
#print("R^2: {}".format(score))