import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # import the random forest model
from sklearn import  preprocessing # used for label encoding and imputing NaNs

import datetime as dt # we will need this to convert the date to a number of days since some point

from sklearn.tree import export_graphviz

df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
df_macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

df_train.head()

# Create a vector containing the id's for our predictions
id_test = df_test.id

#Create a vector of the target variables in the training set
# Transform target variable so that loss function is correct (ie we use RMSE on transormed to get RMLSE)
# ylog1p_train will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
ylog1p_train = np.log1p(df_train['price_doc'].values)
df_train = df_train.drop(["price_doc"], axis=1)

# Create joint train and test set to make data wrangling quicker and consistent on train and test
df_train["trainOrTest"] = "train"
df_test["trainOrTest"] = "test"
df_all = pd.concat([df_train, df_test])

# Removing the id (could it be a useful source of leakage?)
df_all = df_all.drop("id", axis=1)

# Convert the date into a number (of days since some point)
fromDate = min(df_all['timestamp'])
df_all['timedelta'] = (df_all['timestamp'] - fromDate).dt.days.astype(int)
print(df_all[['timestamp', 'timedelta']].head())
df_all.drop('timestamp', axis = 1, inplace = True)

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))
        

# Create a list of columns that have missing values and an index (True / False)
df_missing = df_all.isnull().sum(axis = 0).reset_index()
df_missing.columns = ['column_name', 'missing_count']
idx_ = df_missing['missing_count'] > 0
df_missing = df_missing.ix[idx_]
cols_missing = df_missing.column_name.values
idx_cols_missing = df_all.columns.isin(cols_missing)

# Instantiate an imputer
imputer = preprocessing.Imputer(missing_values='NaN', strategy = 'median', axis = 0)

# Fit the imputer using all of our data (but not any dates)
imputer.fit(df_all.ix[:, idx_cols_missing])

# Apply the imputer
df_all.ix[:, idx_cols_missing] = imputer.transform(df_all.ix[:, idx_cols_missing])

# See the results - note how all missing are replaced with the mode
df_all.head()

# Prepare separate train and test datasets
idx_train = df_all['trainOrTest'] == 1
idx_test = df_all['trainOrTest'] == 0

x_train = df_all[idx_train]
x_test = df_all[idx_test]

# Step 1: Instantiate a decision tree regressor
# Choose a depth for the tree - something 3, 4 or 5 - not too large
Model = DecisionTreeRegressor(max_depth = 5)

# Step 2: Train the tree
# The .fit method takes two main arguments, the features (in our case x_train) and 
# the target variable (in our case ylog1p_train)
# Fill them in below and submit the code to train the tree
Model.fit(X =x_train, y =ylog1p_train)

# Step 3: Make predictions 
# The predict method takes one main argument - the examples for which
# we want to predict the target variable.  Here we will use the training data 
# itself i.e. x_train.  Fill this in below
ylog_pred = Model.predict(X = x_train)

# Check the training error
# Is the training error a reasonable estiamte of how this tree will perform on unseen data?
np.sqrt(np.mean((ylog_pred - ylog1p_train)**2))

# Step 1: Instantiate a random forest regressor
Model = RandomForestRegressor(n_estimators = 100, 
                              random_state = 2017, 
                              oob_score = True, 
                              max_features = 20,
                              min_samples_leaf = 8)
                              

# Step 2: Train the forest
# Again fill in X and y below with x_train and ylog1p_train
Model.fit(X = x_train, y = ylog1p_train)

# Step 3: Make predictions 
# Create predictions for the examples in x_train
ylog_pred = Model.predict(X = x_train)

# Check the training error
np.sqrt(np.mean((ylog_pred - ylog1p_train)**2)) # about 0.37 (if you use 100 trees)

np.sqrt(np.mean((Model.oob_prediction_ - ylog1p_train)**2)) # 0.47 slightly better than a simple tree.

# Create the predictions

ylog_pred = Model.predict(x_test)
y_pred = (np.exp(ylog_pred) - 1) * .969

output = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

output.to_csv('RandomForest_2.csv', index=False)

