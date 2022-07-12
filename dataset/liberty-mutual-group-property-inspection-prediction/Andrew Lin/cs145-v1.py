# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np

# DictVectorizer
from sklearn.feature_extraction import DictVectorizer as DV

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

# The competition datafiles are in the directory ../input
# Read competition data files:
data = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Get a matrix of numerical data
num_cols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14', 'T2_V15']
x_num_data = data[num_cols].as_matrix()
x_num_test = test[num_cols].as_matrix()

# Scale numerical data to <0,1>
max_data = np.amax(x_num_data, 0) # row vector of max values per column
x_num_data = x_num_data / max_data

max_test = np.amax(x_num_test, 0)
x_num_test = x_num_test / max_test

# Convert Panda data frame to a list of dicts
cat_data = data.drop(num_cols + ['Id', 'Hazard'], axis = 1)
x_cat_data = cat_data.T.to_dict().values()

cat_test = test.drop(num_cols + ['Id'], axis = 1)
x_cat_test = cat_test.T.to_dict().values()

# Vectorize:
# Convert categorical data to numerical data
vectorizer = DV( sparse = False )
vec_x_cat_data = vectorizer.fit_transform( x_cat_data )
vec_x_cat_test = vectorizer.fit_transform( x_cat_test )

# Finalize x_data, y_data, and x_test
x_data = np.hstack((x_num_data, vec_x_cat_data)) # concatenate matrices horizontally
y_data = data.Hazard

x_test = np.hstack((x_num_test, vec_x_cat_test))

# Split data to train data and validation data
eval_size = 0.1 # 10% of the data is validation data
kf = KFold(len(y_data), round(1. / eval_size))
train_indices, valid_indices = next(iter(kf))
x_train, y_train = x_data[train_indices], y_data[train_indices]
x_valid, y_valid = x_data[valid_indices], y_data[valid_indices]

#####################################################################
# Modify the two lines below (# 1, # 2) to run different algorithms
#
# Regression Algorithms to consider:
# - Random Forest
# - Linear Regression
# - SVR (Support Vector Regression)
# - Lasso
# - Ridge
# - GBM (Gradient Boosting Machine)
# - *** List more here if you guys know any ***
#
#####################################################################

#from sklearn.ensemble import GradientBoostingRegressor # 1
#model = GradientBoostingRegressor() # 2

#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 200, max_depth = 50)

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# model = RandomForestRegressor(
#         	n_estimators        = 400,
#         	max_depth           = 40,
#         	max_features        = 0.8,
#         	min_samples_leaf    = 30,
# 			random_state 		= 50)

model = DecisionTreeRegressor(
        	max_depth           = 100,
        	max_features        = 1,
        	min_samples_leaf    = 100,
			random_state 		= 50)

model.fit(x_train, y_train)
#model.fit(x_data, y_data)
y_test = model.predict(x_test)
# for i in range(len(y_test)):
#     y_test[i] = round(y_test[i])

#####################################################################


# Uncomment the two lines below to roughly check the accuracy of 
# your model on the validation data 
#y_pred = forest.predict(x_valid)
#print(accuracy_score(y_valid, y_pred))

# Create a data frame with two columns: Id & Hazard
Id = np.array(test["Id"]).astype(int)
my_solution = pd.DataFrame(y_test, Id, columns = ["Hazard"])

# Make sure your solution has the required number of submission samples
print(my_solution.shape)

# Write my_solution to a csv file
# Check the Input section to the right of the Kaggle Kernel text editor
# to preview and submit the csv file
my_solution.to_csv("my_solution.csv", index_label = ["Id"])