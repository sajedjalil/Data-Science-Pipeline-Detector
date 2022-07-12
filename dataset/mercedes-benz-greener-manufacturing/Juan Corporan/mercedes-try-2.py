import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

#What did they do different?
##Add decomposed components: PCA / ICA etc. 
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
n_comp = 10 #(This is one)

# tSVD, this goes unused...
tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

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
    
    #train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    #test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)



### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 750,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, 
    'silent': 0
}


model = xgb.XGBRegressor()
n_estimators = range(400,1000,10)
reg_alpha= st.expon(scale = 0.001)
reg_lambda=st.expon(scale = 1)
learning_rate = st.expon(scale = 0.01)

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,reg_alpha = reg_alpha, reg_lambda = reg_lambda)
grid_search = RandomizedSearchCV(model, param_grid,verbose = 5, scoring="r2",n_iter = 40, n_jobs=2, cv=3)
grid_search.fit(train.drop('y', axis=1), y_train)

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
"""
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1750, # increase to have better results (~700)
                   early_stopping_rounds=500,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))
"""
#Is this it?
# train model
#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=875)
#print(model.eval(dtrain))


# check f2-score (to get higher score - increase num_boost_round in previous cell)
#from sklearn.metrics import r2_score
#print(r2_score(model.predict(dtrain), dtrain.get_label()))

# make predictions and save results
print(grid_search.best_params_)
y_pred = grid_search.predict(test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('submission.csv', index=False)