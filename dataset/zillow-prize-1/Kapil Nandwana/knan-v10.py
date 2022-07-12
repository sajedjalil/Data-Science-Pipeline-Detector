import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

print('Loading data ...')

train = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','storytypeid','basementsqft','yardbuildingsqft26','fireplaceflag','architecturalstyletypeid','typeconstructiontypeid','finishedsquarefeet13','buildingclasstypeid','decktypeid','finishedsquarefeet6','poolsizesum','pooltypeid2','pooltypeid10','taxdelinquencyflag','taxdelinquencyyear','hashottuborspa','yardbuildingsqft17'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.01 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l2'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] =128       # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf


# Set params
# Scores ~0.784 (without tuning and early stopping)    
#params = {'boosting_type': 'gbdt',
#          'max_depth' : -1,
#          'objective': 'binary', 
#          'nthread': 5, 
#          'silent': True,
#          'num_leaves': 64, 
#          'learning_rate': 0.05, 
#         'max_bin': 512, 
#         'subsample_for_bin': 200,
#          'subsample': 1, 
#          'subsample_freq': 1, 
#          'colsample_bytree': 0.8, 
#          'reg_alpha': 5, 
#          'reg_lambda': 10,
#          'min_split_gain': 0.5, 
#          'min_child_weight': 1, 
#          'min_child_samples': 5, 
#          'scale_pos_weight': 1,
#          'num_class' : 1 }
#          'metric' : 'binary_error'
#          }

# Create parameters to search
#UseLater  gridParams = {
#UseLater      'learning_rate': [0.01,0.02]
#    'n_estimators': [8,24,48],
#    'num_leaves': [6,12,20,30],
#    'boosting_type' : ['gbdt'],
#    'objective' : ['binary'],
#    'seed' : [500],
#    'colsample_bytree' : [0.7,0.8],
#    'subsample' : [0.7,1],
#    'reg_alpha' : [0,1,2],
#    'reg_lambda' : [0,1,2],
#UseLater }

# Create classifier to use. Note that parameters have to be input manually
# not as a dict!
#UseLater  mdl = lgb.LGBMClassifier(boosting_type= 'gbdt'
#UseLater          objective = 'regression', 
#UseLater          nthread = 5, 
#UseLater          silent = True
#          max_depth = params['max_depth'],
#          max_bin = params['max_bin'], 
#          subsample_for_bin = params['subsample_for_bin'],
#          subsample = params['subsample'], 
#          subsample_freq = params['subsample_freq'], 
#          min_split_gain = params['min_split_gain'], 
#          min_child_weight = params['min_child_weight'], 
#          min_child_samples = params['min_child_samples'], 
#          scale_pos_weight = params['scale_pos_weight']
 #UseLater          )

# To view the default model params:
#UseLater  mdl.get_params().keys()

# Create the grid
#UseLater  grid = GridSearchCV(mdl, gridParams)
# Run the grid
#UseLater  grid.fit(x_train, y_train)

# Print the best parameters found
# print(grid.best_params_)
# print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
#params['colsample_bytree'] = grid.best_params_['colsample_bytree']
#UseLater  params['learning_rate'] = grid.best_params_['learning_rate'] 
# params['max_bin'] = grid.best_params_['max_bin']
##params['num_leaves'] = grid.best_params_['num_leaves']
#params['reg_alpha'] = grid.best_params_['reg_alpha']
#params['reg_lambda'] = grid.best_params_['reg_lambda']
#params['subsample'] = grid.best_params_['subsample']
## params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']



# other scikit-learn modules
estimator = lgb.LGBMRegressor()

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'num_leaves': [31,128],
    'n_estimators': [20, 40]
}

clf = GridSearchCV(estimator, param_grid)

clf.fit(x_train, y_train)

print('Best parameters found by grid search are:', clf.best_params_)

print('Calculate feature importances...')
# feature importances
#print('Feature importances:', list(clf.feature_importances_))


watchlist = [d_valid]
#Use Later clf = lgb.train(params, d_train, 490, watchlist)

# Plot importance
#Use Later lgb.plot_importance(clf)
#Use Later plt.show()

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()

print("Prepare for the prediction ...")
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
#Use Later clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)

p_test = 0.98*p_test + 0.02*0.011

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_starter_2.csv', index=False, float_format='%.4f')