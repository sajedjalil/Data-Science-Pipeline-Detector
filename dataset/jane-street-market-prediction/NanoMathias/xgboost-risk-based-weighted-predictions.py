# Adjusted predictions based on classifier certainty
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import janestreet

# Load data
train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
print(f'Done loading data. Train shape is {train.shape}')

# For training only look at data that has weight
train = train[train.weight != 0]

# Settings
NAN_VALUE = -999
FEATURES = [c for c in train.columns if 'feature' in c]
TARGET = 'resp'
MAX_WEIGHT = train.weight.max()
SAMPLE_WEIGHTS = (train['resp'] * train['weight']).abs() + 1

# Split into X and y
X = train.loc[:, FEATURES].fillna(NAN_VALUE)
y = (train.loc[:, TARGET] > 0).astype(int)

# Clear memory
del train
gc.collect()

# Train model
# Parameters from: https://www.kaggle.com/hamditarek/market-prediction-xgboost-with-gpu-fit-in-1min
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=11,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    missing=NAN_VALUE,
    random_state=2020,
    tree_method='gpu_hist'
)
model.fit(X, y, sample_weight=SAMPLE_WEIGHTS)
print('Finished training model')

# Clear memory
del X, y
gc.collect()

# Create submission
env = janestreet.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:    
    test_weight = test_df.iloc[0].weight
    if test_weight > 0:
        sample_prediction_df.action = model.predict(test_df.loc[:, FEATURES].fillna(NAN_VALUE))[0]
    else:
        sample_prediction_df.action = 0
    env.predict(sample_prediction_df)