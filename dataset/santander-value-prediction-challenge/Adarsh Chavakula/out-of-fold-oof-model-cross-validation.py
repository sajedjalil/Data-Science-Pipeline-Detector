"""
Use Scikit Learn's cross_val_predict to do a Out-of-Fold Cross validation as opposed 
to averaging out the scores on each fold.
This **usually** tends to be more stable/reliable compared to within fold average.

This script works for all Scikit Learn models as well as the Scikit Learn APIs of
XGBoost, LightGBM and Keras.
"""

import numpy as np 
import pandas as pd 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

# Read Data
print("Reading Dataset...")
train = pd.read_csv("../input/train.csv")
target = np.array(train["target"])
target_log = np.log1p(target) # Log transform target as the evaluation metric uses it
xtrain = np.array(train.iloc[:,2:])
print("Shape of training data: {}".format(np.shape(xtrain)))

# Define Model 
xgb_model = XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=70,
                         min_child_weight=100, subsample=1.0, 
                         colsample_bytree=0.8, colsample_bylevel=0.8,
                         random_state=42, n_jobs=4)

# Make OOF predictions using 5 folds
print("Cross Validating...")
oof_preds_log = cross_val_predict(xgb_model, xtrain, target_log, cv=5, 
                                  n_jobs=1, method="predict")
                                  
# Calculate RMSLE (RMSE of Log(1+y))
cv_rmsle = np.sqrt(mean_squared_error(target_log, oof_preds_log))
print("\nOOF RMSLE Score: {:.4f}".format(cv_rmsle))