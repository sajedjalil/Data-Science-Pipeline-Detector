# Load packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


# Load input files
features = pd.read_csv("../input/properties_2016.csv")
labels = pd.read_csv("../input/train_2016_v2.csv")

# Convert float64 data types to float32 to reduce memory
for c, dtype in zip(features.columns, features.dtypes):
	if dtype == np.float64:
		features[c] = features[c].astype(np.float32)
		
# Drop rows where all features are NaNs
features.dropna(axis = 'index', how = 'all')

# Join features to labels on ParcelID
df = labels.merge(features, how= 'left', on="parcelid")

#################################################################################
# 
# Engineered Features
#
#################################################################################
from datetime import date

# Convert date to month and year
df["transactiondate"] = pd.to_datetime(df["transactiondate"], format = '%Y-%m-%d')
df["month"] = df["transactiondate"].dt.month
df["year"] = df["transactiondate"].dt.year

# Convert date to continuous variable (No. months since Jan 2016)
df["time"] = (df["year"]-2016)*12 + df["month"]

# Pick features based on EDA
chosen_features = [#"basementsqft",
                  #"bathroomcnt",
                  #"bedroomcnt",
                  "lotsizesquarefeet", 
                  "calculatedfinishedsquarefeet",
                  #"fireplacecnt", 
                  #"garagecarcnt", 
                  #"garagetotalsqft",
                  #"poolcnt", 
                  #"poolsizesum", 
                  #"unitcnt", 
                  #"numberofstories", 
                  "structuretaxvaluedollarcnt",
                  "landtaxvaluedollarcnt", 
                  "taxamount", 
                  "latitude",
                  #"longitude"
                  "time"
                  ]
                  
# convert chosen features to NumPy array
features_np = df.as_matrix(columns=chosen_features)

# convert labels to NumPy array
labels_np = df.as_matrix(columns=["logerror"])

# Delete dataframe to reduce memory
del df

# Convert NA values to 0
features_np = np.nan_to_num(features_np)

# Split data into training and testing set
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_np, labels_np, test_size=0.30, random_state=42)

###############################################################################
### 
### XG Boost
###
###############################################################################
import xgboost as xgb

# ?
d_train = xgb.DMatrix(features_train, label=labels_train)
d_test = xgb.DMatrix(features_test, label=labels_test)

# Parameters
labels_mean = np.mean(labels_train)
params = {
    'eta': 0.04,
    'max_depth': 5,
    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.9,   
    'alpha': 0.4, 
    'base_score': labels_mean,
    'silent': 0
}

# Create Classifier
watchlist = [(d_train, 'train'), (d_test, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

# Create test set from output file
output = pd.read_csv("../input/sample_submission.csv")
features["ParcelId"] = features["parcelid"]
df_test = output.merge(features, how= 'left', on="ParcelId")

# Convert chosen features to array
features_np = df_test.as_matrix(columns=chosen_features)

# Convert NA values to 0
features_np = np.nan_to_num(features_np)

# Predict values for output file ParcelIDs
d_test = xgb.DMatrix(features_np)
pred = clf.predict(d_test)
for c in output.columns[output.columns != 'ParcelId']:
    output[c] = pred

# Write Ouput to CSV
output.to_csv("submission.csv", index=False, float_format='%.4g')

