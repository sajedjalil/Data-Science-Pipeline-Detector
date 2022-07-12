import numpy as np
import pandas as pd
import random

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

# Load data
data_path = '../input/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')

# Remove non-onformative columns
cols_to_remove = []
for c in test.columns:
    if len(train[c].unique()) == 1:
        cols_to_remove.append(c)
print('Columns to remove: ' + str(cols_to_remove))
train = train.drop(cols_to_remove, axis=1)
test = test.drop(cols_to_remove, axis=1)

# Add some categorical features
train['X0_0'] = train['X0'].apply(lambda x: x[0])
train['X0_1'] = train['X0'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

test['X0_0'] = test['X0'].apply(lambda x: x[0])
test['X0_1'] = test['X0'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

train['X2_0'] = train['X2'].apply(lambda x: x[0])
train['X2_1'] = train['X2'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

test['X2_0'] = test['X2'].apply(lambda x: x[0])
test['X2_1'] = test['X2'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

# Process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# Add decomposed components: PCA / ICA etc.
n_comp = 12

# PCA
pca = PCA(n_components=n_comp, random_state=random_seed)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=random_seed)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:, i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:, i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    
# Prepare data
X = np.array(train.drop(['y'], axis=1))
y = train.y.values

y_mean = np.mean(y)

X_test = np.array(test)
ids_test = test.ID.values

print('X.shape = ' + str(X.shape) + ', y.shape = ' + str(y.shape))
print('X_test.shape = ' + str(X.shape))

params = {}
params['n_trees'] = 500
params['objective'] = 'reg:linear'
params['eta'] = 0.005
params['max_depth'] = 4
params['subsample'] = 0.95
params['base_score'] = y_mean
params['silent'] = 1

xgb_r2_buf = []
test_preds_buf = []
d_test = xgb.DMatrix(X_test)

cv = ShuffleSplit(n_splits=15, test_size=0.19, random_state=random_seed)
fold_i = 0
for train_index, test_index in cv.split(X):
    print('Fold #' + str(fold_i))
    x_train, x_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)   

    print('XGB: Evaluating model')
    eval_set = [(x_train, y_train), (x_valid, y_valid)]
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, \
        feval=xgb_r2_score, maximize=True, verbose_eval=100)
        
    p = model.predict(d_valid)
    r2 = r2_score(y_valid, p)
    xgb_r2_buf.append(r2)
    print('R2 = ' + str(r2))

    test_preds_buf.append(model.predict(d_test))

    fold_i += 1

print('XGB Mean R2 = ' + str(np.mean(xgb_r2_buf)) + ' +/- ' + str(np.std(xgb_r2_buf)))

print('XGB: Train on full dataset and predicting on test')
d_train = xgb.DMatrix(X, label=y)
watchlist = [(d_train, 'train')]
model = xgb.train(params, d_train, 700, watchlist, feval=xgb_r2_score, \
    maximize=True, verbose_eval=100)

p_test = model.predict(d_test)

test_preds_buf = np.array(test_preds_buf).T
test_preds_buf = np.concatenate((test_preds_buf, p_test.reshape((len(p_test),1))), axis=1)

subm = pd.DataFrame()
subm['ID'] = ids_test
subm['y'] = np.mean(test_preds_buf, axis=1)
subm.to_csv('submission.csv', index=False)
