import pandas as pd
#---------------------
# this script was forked from https://www.kaggle.com/omarito/gridsearchcv-xgbregressor-0-556-lb
# then edited to deal with categorical features using pd.get_dummies (since datacleaner is not installed).
# parameters were also adjusted to all script to run without being killed (20 min limit?)
# and a few items are printed as potential indicators of quality of fit
# No claim of LB score (not submitted)
#---------------------
# From https://github.com/rhiever/datacleaner
#from datacleaner import autoclean
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

import numpy as np


# Load the dataframes
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# Concat the data frames

# This is for autocleaner to work properly

# Since we have categorical variables we will
# need our encoder to label them correctly
# so we must use our encoder on the full
# dataset to avoid having representation
# errors.
data = train_df.append(test_df)

dummy_columns = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
data = pd.get_dummies(data, columns=dummy_columns, drop_first=True)
#data = autoclean(data)
train, test = data[0:len(train_df)], data[len(train_df):]

# Organize our data for training
X = train.drop(["y", "ID"], axis=1)
Y = train["y"]
x_test = test.drop(["y", "ID"], axis=1)
X, X_Val, Y, Y_Val = train_test_split(X, Y)

# A parameter grid for XGBoost
#params = {'min_child_weight':[3,4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
params = {'min_child_weight':[2,3,4,5],   'subsample':[i/10.0 for i in range(8,11)], 'learning_rate':[0.1, .05],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [3,4]}

# Initialize XGB and GridSearch
xgb = XGBRegressor(nthread=4) 

grid = GridSearchCV(xgb, params)
grid.fit(X, Y)

# Print the r2 score
print("R2:", r2_score(Y_Val, grid.best_estimator_.predict(X_Val))) 
print("best estimator:", grid.best_estimator_)
# Save the file
y_test = grid.best_estimator_.predict(x_test)
print("predictions mean:", np.mean(y_test))
print("predictions std:", np.std(y_test))
print("predictions min:", np.min(y_test))
print("predictions max:", np.max(y_test))
results_df = pd.DataFrame(data={'y':y_test}) 
ids = test_df["ID"]
joined = pd.DataFrame(ids).join(results_df)
#joined.to_csv("submit.csv", index=False)

# This scored 0.5563 for me on the LB
# Let me know if you have any remarks or have any ideas for improving it.