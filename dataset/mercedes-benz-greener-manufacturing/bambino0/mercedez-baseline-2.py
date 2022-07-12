import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder


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

train["x"] = pd.Series([1]*len(train.index))
row_select = (train["X314"] == 0) & (train["X315"] == 0) & (train["X47"] == 0)
train.loc[row_select,'x'] = 0 

train["n"] = pd.Series([1]*len(train.index))
row_select = (train["X314"] == 0) & (train["X118"] == 0)
train.loc[row_select,'n'] = 0 


test["x"] = pd.Series([1]*len(test.index))
row_select = (test["X314"] == 0) & (test["X315"] == 0) & (test["X47"] == 0)
test.loc[row_select,'x'] = 0 

test["n"] = pd.Series([1]*len(test.index))
row_select = (test["X314"] == 0) & (test["X118"] == 0)
test.loc[row_select,'n'] = 0

train = train.drop(["X315","X118","X47"],axis = 1)
test = test.drop(["X315","X118","X47"],axis = 1)

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

from sklearn.decomposition import PCA, FastICA
##Add decomposed components: PCA / ICA etc.
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



### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}


# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=800, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score
print(r2_score(model.predict(dtrain), dtrain.get_label()))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('submission_baseLine_revised.csv', index=False)
