# some of the code here comes from https://www.kaggle.com/frednavruzov/baselines-to-start-with-lb-0-56

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import r2_score
import xgboost as xgb

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


print('Started')

# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

# Add decomposed components: PCA / ICA
from sklearn.decomposition import PCA, FastICA
n_comp = 10

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
    
y_train = train["y"]
y_mean = np.mean(y_train)


# Preparing xgb model
def runXGB(train_X, train_y, test_X, test_y=None, seed_val=157, num_rounds=1000):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.05
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'rmse'
    param['min_child_weight'] = 3
    param['subsample'] = 0.95
    param['colsample_bytree'] = 0.95
    param['base_score'] = y_mean
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, model.best_ntree_limit)
    return pred_test_y, model
    
    
# prepare train and test data
X_train = train.drop('y', axis=1).values
X_test = test.values

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(X_train.shape[0])):
        dev_X, val_X = X_train[dev_index,:], X_train[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(r2_score(val_y, preds))
        print(cv_scores)
        break

# run xgb and make predictions
n_rounds = model.best_ntree_limit
print('Geting predictions...')
y_pred, model = runXGB(X_train, y_train, X_test, num_rounds=n_rounds)

print('Writing submission...')
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost-seed157-n{}-pca-ica_v4.csv'.format(n_rounds), index=False)
print('Done')








