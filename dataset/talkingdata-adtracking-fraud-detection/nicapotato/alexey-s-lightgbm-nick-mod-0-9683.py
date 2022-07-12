# THANK YOU AND ACKNOLEDGEMENTS:
# This kernel develops further the ideas suggested in:
#   *  "lgbm starter - early stopping 0.9539" by Aloisio Dourado, https://www.kaggle.com/aloisiodn/lgbm-starter-early-stopping-0-9539/code
#   * "LightGBM (Fixing unbalanced data)" by Pranav Pandya, https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787?scriptVersionId=2777211
#   * "LightGBM with count features" by Ravi Teja Gutta, https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# I would like to extend my gratitude to these individuals for sharing their work.

# WHAT IS NEW IN THIS VERSION? 
# In addition to some cosmetic changes to the code/LightGBM parameters, I am adding the 'ip' feature to and 
# removing the 'day' feature from the training set, and using the last chunk of the training data to build the model.

# What new is NICKS VERSION?
#1 Added Day of Week Time Variable, A IP Count Variable, Feature Importance
#2 Increased validation set to 15%
#3 Imbalanced parameter for lgbm, lower learning rate
#4 gzip submission

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc

path = '../input/' 
path_train = path + 'train.csv'
path_test = path + 'test.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
skip = range(1, 140000000)
print("Loading Data")
train = pd.read_csv(path_train, skiprows=skip, dtype=dtypes,
        header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)
test = pd.read_csv(path_test, dtype=dtypes, header=0,
        usecols=test_cols,parse_dates=["click_time"])#.sample(1000)

len_train = len(train)
print('The initial size of the train set is', len_train)
print('Binding the training and test set together...')
train=train.append(test)

del test
gc.collect()

print("Creating new time features: 'hour' and 'day'...")
train['hour'] = train["click_time"].dt.hour.astype('uint8')
train['day'] = train["click_time"].dt.day.astype('uint8')

#######
# Added
n_chans = train[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip'], how='left')
#######

gc.collect()

print("Creating new count features: 'n_channels', 'ip_app_count', 'ip_app_os_count'...")

print('Computing the number of channels associated with ')
print('a given IP address within each hour...')
n_chans = train[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})

print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','day','hour'], how='left')
del n_chans
gc.collect()

print('Computing the number of channels associated with ')
print('a given IP address and app...')
n_chans = train[['ip','app', 'channel']].groupby(by=['ip', 
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
          
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','app'], how='left')
del n_chans
gc.collect()

print('Computing the number of channels associated with ')
print('a given IP address, app, and os...')
n_chans = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
          
print('Merging the channels data with the main data set...')       
train = train.merge(n_chans, on=['ip','app', 'os'], how='left')
del n_chans
gc.collect()

print("Adjusting the data types of the new count features... ")
train.info()
train['n_channels'] = train['n_channels'].astype('uint16')
train['ip_app_count'] = train['ip_app_count'].astype('uint16')
train['ip_app_os_count'] = train['ip_app_os_count'].astype('uint16')

test = train[len_train:]
print('The size of the test set is ', len(test))

r = 0.15 # the fraction of the train data to be used for validation
val = train[(len_train-round(r*len_train)):len_train]
print('The size of the validation set is ', len(val))

train = train[:(len_train-round(r*len_train))]
print('The size of the train set is ', len(train))

target = 'is_attributed'
train[target] = train[target].astype('uint8')
train.info()

predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels',
              'ip_count','ip_app_count', 'ip_app_os_count']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']
gc.collect()

print("Preparing the datasets for training...")

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 255,  
    'max_depth': 8,  
    'min_child_samples': 100,  
    'max_bin': 100,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
   # 'nthread': 8,
    'verbose': 0,
    'is_unbalance': True
    #'scale_pos_weight':99 
    }
    
dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
                      
evals_results = {}

print("Training the model...")

lgb_model = lgb.train(params, 
                 dtrain, 
                 valid_sets=[dtrain, dvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 num_boost_round=500,
                 early_stopping_rounds=30,
                 verbose_eval=True, 
                 feval=None)

del train
del val
gc.collect()

# Nick's Feature Importance Plot
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')

# Feature names:
print('Feature names:', lgb_model.feature_name())
# Feature importances:
print('Feature importances:', list(lgb_model.feature_importance()))

feature_imp = pd.DataFrame(lgb_model.feature_name(),list(lgb_model.feature_importance()))

print("Preparing data for submission...")

submit = pd.read_csv(path_test, dtype='int', usecols=['click_id'])

print("Predicting the submission data...")

submit['is_attributed'] = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)

print("Writing the submission data into a csv file...")

submit.to_csv("submission.csv",index=False)

print("All done...")