import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print('Importing data...')
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
lgbm_submission = pd.read_csv('../input/sample_submission.csv')

#Separate target variable
y = data['target']
y = np.log(y)
del data['target']

#Delete ID
del data['ID']
del test['ID']

#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'regression',
          'nthread': 5,
          'num_leaves': 32,
          'learning_rate': 0.005,
          'subsample': 0.8,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 1,
          'metric' : 'rmse'
          }

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test)
predictions_lgbm_prob = np.exp(predictions_lgbm_prob)
lgbm_submission.target = predictions_lgbm_prob
lgbm_submission.to_csv('lgbm_submission.csv', index=False)