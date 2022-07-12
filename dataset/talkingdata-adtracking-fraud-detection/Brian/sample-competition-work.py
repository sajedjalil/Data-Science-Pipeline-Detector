# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Step 1: Read data

# iteration 1

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }


train = pd.read_csv('../input/train.csv',dtype=dtypes, parse_dates=["click_time"], nrows=1000)
test = pd.read_csv('../input/test.csv',dtype=dtypes,parse_dates=["click_time"])
sub = pd.read_csv('../input/sample_submission.csv')


#Step 2: Stats summary
print(train.head())
print(test.head())
print(sub.head())

print(train.describe(include = 'all'))

#Step 3: Data cleaning

train.dtypes
train["is_attributed_cat"] = train["is_attributed"].astype('category')



'''
train=train.append(test)

del test
gc.collect()

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

'''



#Step 4: EDA with graphs

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(train[col].unique()) for col in cols]

sns.set(font_scale=1.2)

ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature (from 1,000 samples)')

for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 10,uniq,ha="center") 



#Step 5: Feature Engineering

train['click_id'] = test['click_id'].max()
test['attributed_time'] = train['attributed_time'][1]
test['is_attributed'] = train['is_attributed'][1]

train.columns
test.columns

cols = train.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols
train_mod = train[cols] 
train_mod.columns

full=train_mod.append(test)

print("Creating new time features: 'hour' and 'day'...")
full['hour'] = full["click_time"].dt.hour.astype('uint8')
full['day'] = full["click_time"].dt.day.astype('uint8')

full.head()

train_new = full[0:500]
test_new = full[1000:]
val_new = full[500:1000]


train.head()
test.head()

#Step 6: Model specification


import lightgbm as lgb


predictors = ['ip', 'app','device', 'os', 'channel', 'hour']
target = 'is_attributed'


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 255,  
    'max_depth': 9,  
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

dtrain = lgb.Dataset(train_new[predictors].values, label=train_new[target].values,
                      feature_name=predictors
                      )
dvalid = lgb.Dataset(val_new[predictors].values, label=val_new[target].values,
                      feature_name=predictors
                      )
                      
evals_results = {}



#Step 7: model fitting

print("Training the model...")

lgb_model = lgb.train(params, 
                 dtrain, 
                 valid_sets=[dtrain, dvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 num_boost_round=1000,
                 early_stopping_rounds=30,
                 verbose_eval=50, 
                 feval=None)



#Step 8: Cross validation

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc





#Step 9: model optimization techniques


# Nick's Feature Importance Plot
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')


print('Feature names:', lgb_model.feature_name())  # Feature names:

print('Feature importances:', list(lgb_model.feature_importance())) # Feature importances:



#Step 10: make submission

print("Predicting the submission data...")

sub['is_attributed'] = lgb_model.predict(test_new[predictors], num_iteration=lgb_model.best_iteration)

print("Writing the submission data into a csv file...")

sub.to_csv("submission.csv",index=False)







#Reference/Acknowledgement:
# https://pandas.pydata.org/pandas-docs/stable/categorical.html
# https://www.datacamp.com/community/tutorials/python-data-type-conversion
# https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns
# https://www.kaggle.com/nicapotato/alexey-s-lightgbm-nick-mod-0-9683
# https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns










