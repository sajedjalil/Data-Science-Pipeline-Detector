# This will score 0.556+ on the LB, you can further raise this score
# by using PCA and SVD as features.
import pandas as pd

# From https://github.com/rhiever/datacleaner
from datacleaner import autoclean
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
data = autoclean(data)
train, test = data[0:len(train_df)], data[len(train_df):]

# Organize our data for training
X = train.drop(["y"], axis=1)
Y = train["y"]
x_test = test.drop(["y"], axis=1)
X, X_Val, Y, Y_Val = train_test_split(X, Y)

# A parameter grid for XGBoost
params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}

# Initialize XGB and GridSearch
xgb = XGBRegressor(nthread=-1) 

grid = GridSearchCV(xgb, params)
grid.fit(X, Y)

# Print the r2 score
print(r2_score(Y_Val, grid.best_estimator_.predict(X_Val))) 

# Save the file
y_test = grid.best_estimator_.predict(x_test)
results_df = pd.DataFrame(data={'y':y_test}) 
ids = test_df["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)

# This scored 0.5563 for me on the LB
# Let me know if you have any remarks or have any ideas for improving it.