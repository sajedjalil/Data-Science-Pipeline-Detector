# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# LightGBM regression example
# https://www.kaggle.com/tobikaggle/humble-lightgbm-starter/
# __author__ = "DDgg"
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
# -----------------------------------------------------------------------------
import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Display PCA, ICA, t-SNE add to see skewness
# -----------------------------------------------------------------------------

# data imnport 
# fork of forks from https://www.kaggle.com/jaybob20/starter-xgboost
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#pca_3D_plot(test)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] =  lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
        
#------------------------------------------------------------------------------
# pca transform forked from https://www.kaggle.com/jaybob20/starter-xgboost/code
# pca and Ica transform from https://www.kaggle.com/uluumy/mercedez-baseline-2
# OLD The rmse of prediction is: 7.3375551233

##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA

## optimize and display number of components // whole dataset
#  5 comp:  The r2 is:  0.691174519763 the rmse is: 7.04535736811
# 10 comp:  The r2 is:  0.709094189016 the rmse is: 6.83789870052
# 20 comp:  The r2 is:  0.726726138723 the rmse is: 6.6274355197
# 30 comp:  The r2 is:  0.742673315642 the rmse is: 6.4311535142
# 40 comp:  The r2 is:  0.747617698519 the rmse is: 6.36906839436
# 60 comp:  The r2 is:  0.988973825782 the rmse is: 1.33124775841
# 80 comp:  The r2 is:  0.990269624615 the rmse is: 1.25057937524
#    20L    The r2 is:  0.993219353078 the rmse is: 1.04395560348
#    40L    The r2 is:  0.998705771167 the rmse is: 0.456091522319
# 100 comp: The r2 is:  0.991084646309 the rmse is: 1.19705954246
# 160 comp: The r2 is:  0.992118848356 the rmse is: 1.12548913381
# 200 comp: The r2 is:  0.993216736475 the rmse is: 1.04415701154
# 300 comp: The r2 is: 
# 377 comp  The r2 is:  0.99903249422 the rmse is: 0.394342194068
#----------------------------------------------------------------
## 80/20 split with random_state=123

#  2 comp:  The r2 is:  0.638091262363 the rmse is: 7.35414169879
#  5 comp:  The r2 is:  0.639438224827 the rmse is: 7.34044351212
#  6 comp:  The r2 is:  0.648019302812 the rmse is: 7.2525692268
#  7 comp:  The r2 is:  0.638189248199 the rmse is: 7.35314607417
# 10 comp:  The r2 is:  0.637207039653 the rmse is: 7.36312011156
# 20 comp:  The r2 is:  0.633268557399 the rmse is: 7.4029792608
# 30 comp:  The r2 is:  0.632985260523 the rmse is: 7.40583807761
# 40 comp:  The r2 is:  0.632551562409 the rmse is: 7.4102124928
# 80 comp:  The r2 is:  0.629313920054 the rmse is: 7.44278713255
# 100 comp: The r2 is:  0.628919943592 the rmse is: 7.44674129266
# 200 comp: The r2 is:  0.623010420635 the rmse is: 7.50580249174
# 300 comp: The r2 is:  0.620508076124 the rmse is: 7.53067193139 
# 377 comp: The r2 is:  0.627785004593 the rmse is: 7.45812043387
n_comp = 6

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
  
    
# remove  duplicates - needs to be applied to test too
# train = train.T.drop_duplicates().T
# test = test.T.drop_duplicates().T

    
y_train = train["y"]
y_mean = np.mean(y_train)
train.drop('y', axis=1, inplace=True)

#------------------------------------
# split into training and validation set
# the data has a number of outliers, so the validation size needs
# to be large enough plus cross-validation is needed

# RND= 123   The r2 is:  0.648019302812 the rmse is: 7.2525692268
# RND= 63466 The r2 is:  0.702588909905 the rmse is: 6.24188256127

X_train, X_valid, y_train, y_valid = train_test_split(
       train, y_train, test_size=0.2, random_state=9127)

# bad bad
# X_train = train
# X_valid = train
# y_train = y_train
# y_valid = y_train

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# to record eval results for plotting
evals_result = {} 

# The r2 is:  0.648019302812 the rmse is: 7.2525692268
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 5,
    'learning_rate': 0.06,
    'max_depth': 4,
    'subsample': 0.95,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.85,
    'bagging_freq': 4,
    'min_data_in_leaf':4,
    'min_sum_hessian_in_leaf': 0.8,
    'verbose':10
}

print('Start training...')

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=8000, # 200
                valid_sets=[lgb_train, lgb_valid],
                evals_result=evals_result,
                verbose_eval=10,
                early_stopping_rounds=50) # 50

#print('\nSave model...')
# save model to file
#gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

print('\nFeature names:', gbm.feature_name())

print('\nCalculate feature importances...')

# feature importances
print('Feature importances:', list(gbm.feature_importance()))

# -------------------------------------------------------
print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='l2')
plt.show()

print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()
# -------------------------------------------------------
# eval r2-score 
from sklearn.metrics import r2_score
r2 = r2_score(y_valid, y_pred)

# eval rmse (lower is better)
print('\nThe r2 is: ',r2, 'the rmse is:', mean_squared_error(y_valid, y_pred) ** 0.5)

# -------------------------------------------------------
print('\nPredicting test set...')
y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

# y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'], 'y': y_pred})
output.to_csv('submit-lightgbm-ICA-PCA.csv', index=False)

# -----------------------------------------------------------------------------
print("Finished.")
# -----------------------------------------------------------------------------

#==============================================================================
# # Grid search example // uncomment block if needed
# from sklearn.model_selection import GridSearchCV
# estimator = lgb.LGBMRegressor()
# 
# print("\n-----------------------------------------------------------------------------")
# #==============================================================================
#==============================================================================
# print("Now doing grid search.")
# 
# # get possible parameters
# estimator.get_params().keys()
# 
# # fill parameters ad libitum
# param_grid = {
#     'num_leaves': [2, 5, 10, 20],    
#     'learning_rate': [0.06],
#     'n_estimators': [100],
# #     'colsample_bytree' :[],
# #     'min_split_gain' :[],
# #     'subsample_for_bin' :[],
#      'max_depth' :[1,2,3,4,5,10],
# #      'subsample' :[], 
# #     'reg_alpha' :[], 
# #     'max_drop' :[], 
# #     'gaussian_eta' :[], 
# #     'drop_rate' :[], 
# #     'silent' :[], 
# #     'boosting_type' :[], 
# #     'min_child_weight' :[], 
# #     'skip_drop' :[], 
# #     'learning_rate' :[], 
# #     'fair_c' :[], 
# #     'seed' :[], 
# #     'poisson_max_delta_step' :[], 
# #     'subsample_freq' :[], 
# #     'max_bin' :[], 
# #     'n_estimators' :[], 
# #     'nthread' :[], 
# #     'min_child_samples' :[], 
# #     'huber_delta' :[], 
# #     'use_missing' :[], 
# #     'uniform_drop' :[], 
# #     'reg_lambda' :[], 
# #     'xgboost_dart_mode' :[], 
# #     'objective'
# }
# 
# 
# gbm = GridSearchCV(estimator, param_grid)
# 
# gbm.fit(X_train, y_train)
# 
# # list them
# print('Best parameters found by grid search are:', gbm.best_params_)
# print("finished grid search")
#==============================================================================
# -----------------------------------------------------------------------------