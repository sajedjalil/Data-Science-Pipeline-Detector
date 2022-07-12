# https://gplearn.readthedocs.io/en/stable/examples.html#example-1-symbolic-regressor
import numpy as np
from scipy import sparse, stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import time
import os
import pickle

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

def _less(x1, x2):
    return 1*(x1 < x2)

def _x10(x1):
    return 10*x1

def _x01(x1):
    return 0.1*x1

tanh = make_function(function=np.tanh, name='tanh', arity=1)
sqr = make_function(function=np.square, name='sqr', arity=1)
x10 = make_function(function=_x10, name='x10', arity=1)
x01 = make_function(function=_x01, name='x01', arity=1)
less = make_function(function=_less, name='less', arity=2)
function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'inv', 'min', 'max', 'cos', 'sin', 'tan', tanh, less, sqr, x10, x01)

print('1. Loading data')
X_tr = pd.read_csv('../input/lanl-feature-transform/train_features.csv', dtype=np.float64)
Y_tr = pd.read_csv('../input/lanl-ttf-error/time_prediction.csv', dtype=np.float64, usecols=[0])
Z_tr = pd.read_csv('../input/lanl-ttf-error/time_prediction.csv', dtype=np.float64, usecols=[1])
print(X_tr.shape)
print(Y_tr.shape)
print(Z_tr.shape)

R_tr = np.divide(Y_tr.values,Z_tr.values)
index_condition = (Y_tr.values > 0.275)
X_tr = X_tr[index_condition]
Y_tr = Y_tr[index_condition]
Z_tr = Z_tr[index_condition]
R_tr = R_tr[index_condition]
XY_tr = pd.concat([X_tr,Y_tr], axis=1)
print(X_tr.shape)
print(Y_tr.shape)
print(Z_tr.shape)

good_columns = [column for column in X_tr.columns if abs(stats.pearsonr(X_tr[column], Y_tr.values.ravel())[0]) > 0.05]
#good_columns = ['ifreq_abs_median_mean', 'ifreq_abs_q95_mean', 'denoise_abs_median_mean', 'denoise_abs_num_peaks_R_mean']
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
def train_model(X=X_train_scaled, X_test=X_test_scaled, Y=(R_tr * 100).ravel(), model_name='gpl'):
    print(model_name)
    if os.path.isfile('../input/lanl-gp-regression/gp_model.pkl'):
        with open('../input/lanl-gp-regression/gp_model.pkl', 'rb') as f:
            model = pickle.load(f)
        model.set_params(generations=300, warm_start=True)
    #else:
    model = SymbolicRegressor(population_size=20000, generations=2, random_state=17, verbose=1,
                   p_crossover=0.1, p_subtree_mutation=0.2, p_hoist_mutation=0.1, p_point_mutation=0.4,
                   parsimony_coefficient=0.0002, stopping_criteria=0.00001, max_samples=0.9,
                   function_set=function_set, tournament_size = 25, n_jobs=-1,
                   init_depth=(5, 10), init_method='full', const_range=(-10.,10.))
    model.fit(X, Y)
    print(model._program)
    pred_train = model.predict(X)
    pred_test = model.predict(X_test)
    print('CV score: {0:.4f}.'.format(mean_absolute_error(pred_train, Y)))

    pd.DataFrame(pred_train).to_csv('train_predictions_{0}.csv'.format(model_name), index=False)
    pd.DataFrame(pred_test, columns=['time_to_failure'], index=X_test.index).to_csv('test_predictions_{0}.csv'.format(model_name), index=True)
    
    with open('gp_model.pkl', 'wb') as f:
        pickle.dump(model, f)

train_model(model_name='gpl')
