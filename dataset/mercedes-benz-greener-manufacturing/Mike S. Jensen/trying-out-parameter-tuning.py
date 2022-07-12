import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# process columns, apply LabelEncoder to categorical features
#for c in train.columns:
#    if train[c].dtype == 'object':
#        lbl = LabelEncoder() 
#        lbl.fit(list(train[c].values) + list(test[c].values)) 
#        train[c] = lbl.transform(list(train[c].values))
#        test[c] = lbl.transform(list(test[c].values))


# one hot instead of label encoding
alldata = train.append(test)
alldata = pd.get_dummies(alldata)

train_new = alldata[:len(train)]
test_new =  alldata[len(train):len(train) + len(test)]
test_new = test_new.drop(["y"], axis=1)

#train_new = pd.get_dummies(train);
#test_new = pd.get_dummies(test);
#not_in_test = list(set(train_new.columns).difference(test_new.columns))
#not_in_train = list(set(test_new.columns).difference(train_new.columns))

#print(not_in_test)
#print(not_in_train)


#add = pd.DataFrame(0, index=np.arange(len(test_new)), columns=not_in_test)
#test_new = test_new.join(add)
#test_new = test_new.drop('y', axis=1)

#drp = pd.DataFrame(0, index=np.arange(len(train_new)), columns=not_in_train)
#test_new = test_new.drop(drp, axis=1)

train = train_new
test = test_new

print(train.shape)
print(test.shape)

# shape        
#print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

# note to self -- this includes ID as a predictor -- maybe OK, maybe not

##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
n_comp = 50

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# PLS
pls = PLSRegression(n_components=n_comp)
(pls2_results_train, pls2_y) = pls.fit_transform(train.drop(["y"], axis=1), train["y"])
pls2_results_test  = pls.transform(test)

print(type(ica2_results_test))
print(type(pls2_results_test))
print(type(pls2_results_train))

print(ica2_results_test.shape)
print(pls2_results_train.shape)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    train['pls_' + str(i)] = pls2_results_train[:,i-1]
    test['pls_' + str(i)] = pls2_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)

print(train.head)

### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 200, 
    'eta': 0.005,
    'max_depth': 3,
    'subsample': 0.95,
    'colsample_bytree': 1.0,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    'min_child_weight': 1
}


# form DMatrices for Xgboost training
#dtrain = xgb.DMatrix(train.drop(["y", "ID"], axis=1), y_train)
#dtest = xgb.DMatrix(test.drop(["ID"], axis=1))

dtrain = xgb.DMatrix(train.drop(["y"], axis=1), y_train)
dtest = xgb.DMatrix(test)


# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1000, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)


# use GridSearchCV to perform parameter tuning on best tree  
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

param_test1 = {
 'max_depth': [ 2, 3, 4],
 'subsample': [0.95, .9, 0.8]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(**xgb_params, 
    learning_rate = xgb_params['eta']), param_grid = param_test1, cv=5)
 
#gsearch1.fit(train.drop(["y", "ID"], axis=1), y_train)
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

#for k in gsearch1.best_params_ :
#    xgb_params[k] = gsearch1.best_params_[k]

#print(xgb_params)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# check r2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score
print(r2_score(dtrain.get_label(), model.predict(dtrain)))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('slightly_tuned_xgb.csv', index=False)
