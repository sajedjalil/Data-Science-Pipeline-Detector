import numpy as np
from scipy import sparse, stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import time

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.svm import NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

N_estimators = 5000
N_iterations = 50000
N_iterations_few = 1000
Early_stopping_rounds = 500

print('1. Loading data')
X_tr = pd.read_csv('../input/lanl-feature-transform/train_features.csv', dtype=np.float64)
Y_tr = pd.read_csv('../input/lanl-ttf-error/time_prediction.csv', dtype=np.float64, usecols=[0])
Z_tr = pd.read_csv('../input/lanl-ttf-error/time_prediction.csv', dtype=np.float64, usecols=[1])
print(X_tr.shape)
print(Y_tr.shape)
print(Z_tr.shape)

from sklearn.model_selection import LeaveOneGroupOut
group_kfold = LeaveOneGroupOut()
groups = np.unique(Z_tr)
n_fold = 9 #groups.shape[0]-1
folds = KFold(n_splits=n_fold, shuffle=False, random_state=17)
print(n_fold)

index_condition = (Y_tr.values > 0.275)
X_tr = X_tr[index_condition]
Y_tr = Y_tr[index_condition]
Z_tr = Z_tr[index_condition]
XY_tr = pd.concat([X_tr,Y_tr], axis=1)
print(X_tr.shape)
print(Y_tr.shape)
print(Z_tr.shape)

good_columns = ['denoise_abs_num_peaks_R_mean', 'ifreq_abs_median_mean', 'num_crossing_mean_mean']
#good_columns = ['rW_num_crossing_mean_mean', 'num_peaks_R_mean', 'denoise_abs_num_peaks_R_mean', 'hT_abs_time_rev_asym_stat_1_mean', 'hT_env_q05_mean', 'ifreq_abs_median_mean', 'rW_ifreq_mean_mean', 'rW_ifreq_skew_mean', 'rW_time_rev_asym_stat_5T_mean']
#good_columns = [column for column in X_tr.columns]# if abs(stats.pearsonr(X_tr[column], Z_tr.values.ravel())[0]) > 0.05]
X_tr = X_tr[good_columns]
print(X_tr.shape)

scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=[str(col) + '_scaled' for col in X_tr.columns], index=X_tr.index)
#X_train_all = pd.concat([X_tr,X_train_scaled], axis=1)

#x,y = sparse.coo_matrix(X_train_all.isnull()).nonzero()
#print('X_train_all no data indices:')
#print(list(zip(x,y)))

X_test = pd.read_csv('../input/lanl-feature-transform/test_features.csv', index_col=[0])
#Y_test = pd.read_csv('../input/lanl-selection/submission_median.csv', index_col=[0])
#XY_test = pd.concat([X_test,Y_test], axis=1)
X_test = X_test[good_columns]
print(X_test.shape)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=[str(col) + '_scaled' for col in X_test.columns], index=X_test.index)
#X_test_all = pd.concat([X_test,X_test_scaled], axis=1)

#x,y = sparse.coo_matrix(X_test_all.isnull()).nonzero()
#print('X_test_all no data indices:')
#print(list(zip(x,y)))

print('2. Computing')
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=np.divide(Y_tr,Z_tr), z=Z_tr, params=None, folds=folds, model_type='lgb', model_name='lgb', model=None):
    print(model_name)
    pred_valid = np.zeros(len(X))
    pred_train = np.zeros((len(X), n_fold))
    pred_test = np.zeros((len(X_test), n_fold))
    scores = np.zeros(n_fold)
    fold_splitter = folds.split(X)#, y, z.values.ravel())
    for fold_n, (train_index, valid_index) in enumerate(fold_splitter):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        z_valid = z.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=False, early_stopping_rounds=Early_stopping_rounds)
            y_pred_valid = model.predict(X_valid)
            y_pred_train = model.predict(X, num_iteration=model.best_iteration_)
            y_pred_test = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)
            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=N_iterations, evals=watchlist, early_stopping_rounds=Early_stopping_rounds, verbose_eval=False, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred_train = model.predict(xgb.DMatrix(X, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred_test = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train.values.reshape(-1,))
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            y_pred_train = model.predict(X).reshape(-1,)
            y_pred_test = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
            y_pred_valid = model.predict(X_valid)
            y_pred_train = model.predict(X)
            y_pred_test = model.predict(X_test)

        pred_valid[valid_index] = y_pred_valid.reshape(-1,)
        pred_train[:, fold_n] = y_pred_train
        pred_test[:, fold_n] = y_pred_test
        scores[fold_n] = mean_absolute_error(y_valid, y_pred_valid)
        print(f'Fold {fold_n}. MAE: {scores[fold_n]:.4f}.')

    print('CV scores mean: {0:.4f}, std: {1:.4f}.'.format(scores.mean(), scores.std()))
    print('CV score: {0:.4f}.'.format(mean_absolute_error(pred_valid, y)))

    pd.DataFrame(pred_valid).to_csv('train_validation_predictions_{0}.csv'.format(model_name), index=False)
    pd.DataFrame(pred_train).to_csv('train_predictions_{0}.csv'.format(model_name), index=False)
    pd.DataFrame(pred_test).to_csv('test_predictions_{0}.csv'.format(model_name), index=False)
    pd.DataFrame(scores).to_csv('test_scores_{0}.csv'.format(model_name), index=False)

model = NuSVR(gamma='auto', kernel='rbf', nu=0.75, C=1.0, tol=0.01, max_iter=N_iterations)
train_model(params=None, model_type='sklearn', model_name='svr', model=model)

xgb_params = {
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': True,
    'nthread': 4
}
train_model(params=xgb_params, model_type='xgb', model_name='xgb')

params = {
    "num_leaves": 54,
    "min_data_in_leaf": 79,
    "max_depth": -1,
    "boosting": "gbdt",
    "bagging_freq": 4,
    "bagging_fraction": 0.8126672064208567,
    "reg_alpha": 0.1302650970728192,
    "reg_lambda": 0.3603427518866501,
    "learning_rate": 0.02,
    "objective": "regression",
    "metric": "mae",
    "bagging_seed": 17,
    "verbosity": -1,
    "n_estimators": N_estimators,
    "n_jobs": -1
}
train_model(params=params, model_type='lgb', model_name='lgb')

params = {
    'loss_function': 'MAE',
    'iterations': N_iterations_few,
    'eval_metric': 'MAE', 
    'learning_rate': 0.1,
    'depth': 5,
    'random_seed': 17
}
train_model(params=params, model_type='cat', model_name='cat')

#model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.005)
#train_model(params=None, model_type='sklearn', model_name='kr', model=model)

#model = GradientBoostingRegressor(loss='huber', n_estimators=N_estimators, learning_rate=0.05)
#train_model(params=None, model_type='sklearn', model_name='gb', model=model)

#model = KNeighborsRegressor(n_neighbors=20, weights='distance', p=1)
#train_model(params=None, model_type='sklearn', model_name='knb', model=model)

#model = MLPRegressor(max_iter=N_iterations, activation ='identity', hidden_layer_sizes=(16,8), random_state=17, n_iter_no_change=Early_stopping_rounds, alpha=0.02, learning_rate_init=0.001)
#train_model(params=None, model_type='sklearn', model_name='mlp', model=model)