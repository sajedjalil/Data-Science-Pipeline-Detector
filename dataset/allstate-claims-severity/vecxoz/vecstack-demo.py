#-------------------------------------------------------------------------------
# Below we have REDIFINITION of package https://github.com/vecxoz/vecstack
# Actual script begins at line 100
# If we have package we could just do: from vecstack import stacking
#-------------------------------------------------------------------------------

import numpy as np
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def transformer(y, func=None):
    if func is None:
        return y
    else:
        return func(y)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression=True,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
    # Print type of task
    if regression and verbose > 0:
        print('task:   [regression]')
    elif not regression and verbose > 0:
        print('task:   [classification]')

    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = accuracy_score
        
    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)
        
    # Split indices to get folds (stratified can be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(len(y_train), n_folds, shuffle = shuffle, random_state = random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], len(kf)))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, transformer(y_tr, func = transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(model.predict(X_te), func = transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(model.predict(X_test), func = transform_pred)
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return (S_train, S_test)

#-------------------------------------------------------------------------------
# END of REDIFINITION
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# SCRIPT begins
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load data
dir_path = '../input/'
train_df = pd.read_csv(dir_path + 'train.csv', sep = ',', header = 0, low_memory = False)
test_df = pd.read_csv(dir_path + 'test.csv', sep = ',', header = 0)
subm_df = pd.read_csv(dir_path + 'sample_submission.csv', sep = ',', header = 0)

# Encode categorical
r, c = train_df.shape
test_df.loc[:, 'loss'] = 0
z_df = pd.concat([train_df, test_df])
obj_cols = train_df.dtypes[train_df.dtypes == 'object'].index.tolist()
for col in obj_cols:
    z_df.loc[:, col] = pd.factorize(z_df[col], sort = True)[0]
        
train_df = z_df[:r]
test_df = z_df[r:]

# Get numpy arrays
y_col = 'loss'
X_cols = train_df.columns.tolist()
X_cols.remove(y_col)
X_cols.remove('id')

X_train = train_df[X_cols].values
y_train = train_df[y_col].values
X_test = test_df[X_cols].values

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# Categorical features in sparse form
ohe = OneHotEncoder(sparse = True)
Z_sparse = ohe.fit_transform(z_df[obj_cols])

X_train_sparse = Z_sparse[:r]
X_test_sparse = Z_sparse[r:]

print('X_train_sparse.shape:', X_train_sparse.shape)
print('X_test_sparse.shape:', X_test_sparse.shape)

#-------------------------------------------------------------------------------
print('\nStarting stacking on dense data and log transformed target\n')

# Initialize 1-st level models
models_1 = [
    XGBRegressor(seed = 0, nthread = -1, colsample_bytree = 0.69, subsample = 0.69, learning_rate = 0.1, 
        max_depth = 6, min_child_weight = 1, n_estimators = 100),
    ExtraTreesRegressor(random_state = 0, n_jobs = -1, n_estimators = 100, 
        max_features = 0.49, max_depth = 8, min_samples_leaf = 2),
    RandomForestRegressor(random_state = 0, n_jobs = -1, n_estimators = 100, 
        max_features = 0.22, max_depth = 8, min_samples_leaf = 2),
    ]

# Get stacking features
S_train_1, S_test_1 = stacking(models_1, X_train, y_train, X_test, 
    n_folds = 4, shuffle = True, transform_target = np.log, 
    transform_pred = np.exp, verbose = 2)
#-------------------------------------------------------------------------------
print('\nStarting stacking on sparse data and log transformed target\n')

# Initialize 1-st level models
models_2 = [
    Ridge(alpha = 200),
    SGDRegressor(random_state = 0, alpha = 0.01, power_t = 0.40),
    ]

# Get stacking features
S_train_2, S_test_2 = stacking(models_2, X_train_sparse, y_train, 
    X_test_sparse, n_folds = 4, shuffle = True, 
    transform_target = np.log, transform_pred = np.exp, verbose = 2)
#-------------------------------------------------------------------------------
print('\nStarting stacking on sparse data and original target\n')

models_3 = [
    LinearSVR(random_state = 0, C = 10),
    ]
    
# Get stacking features
S_train_3, S_test_3 = stacking(models_3, X_train_sparse, y_train, 
    X_test_sparse, n_folds = 4, shuffle = True, 
    transform_target = None, transform_pred = None, verbose = 2)
#-------------------------------------------------------------------------------

print('\nFitting 2-nd level model\n')

# 2-nd level model
model = XGBRegressor(seed = 0, colsample_bytree = 0.79, subsample = 0.59, 
    learning_rate = 0.1, max_depth = 4, min_child_weight = 1, n_estimators = 300)
model = model.fit(np.c_[S_train_1, S_train_2, S_train_3], np.log(y_train))
subm_df.iloc[:, 1] = np.exp(model.predict(np.c_[S_test_1, S_test_2, S_test_3]))
subm_df.to_csv('s.csv', sep = ',', index = False)

print('Done')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------















