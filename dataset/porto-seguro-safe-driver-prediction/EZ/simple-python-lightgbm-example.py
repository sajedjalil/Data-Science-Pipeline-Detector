# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


#
# Prepare the data
#

train = pd.read_csv('../input/train.csv')

# get the labels
y = train.target.values
train.drop(['id', 'target'], inplace=True, axis=1)

x = train.values

#
# Create training and validation sets
#
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#
# Create the LightGBM data containers
#
categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]
train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
test_data = lightgbm.Dataset(x_test, label=y_test)


#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)
#
# Create a submission
#

submission = pd.read_csv('../input/test.csv')
ids = submission['id'].values
submission.drop('id', inplace=True, axis=1)


x = submission.values
y = model.predict(x)

output = pd.DataFrame({'id': ids, 'target': y})
output.to_csv("submission.csv", index=False)
