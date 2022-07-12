import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn.metrics import r2_score

import numpy as np

from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

from bayes_opt import BayesianOptimization

def average_dups(x):
    # Average value of duplicates
    Y.loc[list(x.index)] = Y.loc[list(x.index)].mean()

def xgb_r2_score(preds, dtrain):
    # Courtesy of Tilii
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def train_xgb(max_depth, subsample, min_child_weight, gamma, colsample_bytree):
    # Evaluate an XGBoost model using given params
    xgb_params = {
        'n_trees': 250,
        'eta': 0.01,
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'objective': 'reg:linear',
        'base_score': np.mean(Y), # base prediction = mean(target)
        'silent': 1,
        'min_child_weight': int(min_child_weight),
        'gamma': max(gamma, 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0)
    }
    scores = xgb.cv(xgb_params, dtrain, num_boost_round=1500, early_stopping_rounds=50, verbose_eval=False, feval=xgb_r2_score, maximize=True, nfold=5)['test-r2-mean'].iloc[-1]
    return scores


# Load the dataframes
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

for c in train.columns:
    if train[c].dtype == 'object':

        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


# Organize our data for training
X = train.drop(["y"], axis=1)
Y = train["y"]
X_Test = test
# X_Test = test.drop(["ID"], axis=1)

# Handling duplicate values
# First we group the duplicates and then average them
dups = X[X.duplicated(keep=False)]
dups.groupby(dups.columns.tolist()).apply(average_dups)

# Drop duplicates keeping only 1 instance of each group
train.drop(X[X.duplicated()].index.values, axis=0, inplace=True)
X = train.drop(["y"], axis=1)
Y = train["y"]

# Fix index after dropping
X.reset_index(inplace=True, drop=True)
Y.reset_index(inplace=True, drop=True)

# Handling outliers
# Y[Y > 150] = Y.quantile(0.99)


pca = PCA(n_components=5)
ica = FastICA(n_components=5, max_iter=1000)
tsvd = TruncatedSVD(n_components=5)
gp = GaussianRandomProjection(n_components=5)
sp = SparseRandomProjection(n_components=5, dense_output=True)

x_pca = pd.DataFrame(pca.fit_transform(X))
x_ica = pd.DataFrame(ica.fit_transform(X))
x_tsvd = pd.DataFrame(tsvd.fit_transform(X))
x_gp = pd.DataFrame(gp.fit_transform(X))
x_sp = pd.DataFrame(sp.fit_transform(X))

x_pca.columns = ["pca_{}".format(i) for i in x_pca.columns]
x_ica.columns = ["ica_{}".format(i) for i in x_ica.columns]
x_tsvd.columns = ["tsvd_{}".format(i) for i in x_tsvd.columns]
x_gp.columns = ["gp_{}".format(i) for i in x_gp.columns]
x_sp.columns = ["sp_{}".format(i) for i in x_sp.columns]

X = pd.concat((X, x_pca), axis=1)
X = pd.concat((X, x_ica), axis=1)
X = pd.concat((X, x_tsvd), axis=1)
X = pd.concat((X, x_gp), axis=1)
X = pd.concat((X, x_sp), axis=1)

x_test_pca = pd.DataFrame(pca.transform(X_Test))
x_test_ica = pd.DataFrame(ica.transform(X_Test))
x_test_tsvd = pd.DataFrame(tsvd.transform(X_Test))
x_test_gp = pd.DataFrame(gp.transform(X_Test))
x_test_sp = pd.DataFrame(sp.transform(X_Test))

x_test_pca.columns = ["pca_{}".format(i) for i in x_test_pca.columns]
x_test_ica.columns = ["ica_{}".format(i) for i in x_test_ica.columns]
x_test_tsvd.columns = ["tsvd_{}".format(i) for i in x_test_tsvd.columns]
x_test_gp.columns = ["gp_{}".format(i) for i in x_test_gp.columns]
x_test_sp.columns = ["sp_{}".format(i) for i in x_test_sp.columns]


X_Test = pd.concat((X_Test, x_test_pca), axis=1)
X_Test = pd.concat((X_Test, x_test_ica), axis=1)
X_Test = pd.concat((X_Test, x_test_tsvd), axis=1)
X_Test = pd.concat((X_Test, x_test_gp), axis=1)
X_Test = pd.concat((X_Test, x_test_sp), axis=1)

dtrain = xgb.DMatrix(X, Y)
dtest = xgb.DMatrix(X_Test)

# A parameter grid for XGBoost
params = {
  'min_child_weight':(1, 20),
  'gamma':(0, 10),
  'subsample':(0.5, 1),
  'colsample_bytree':(0.1, 1),
  'max_depth': (2, 10)
}

# Initialize BO optimizer
xgb_bayesopt = BayesianOptimization(train_xgb, params)

# Maximize R2 score
xgb_bayesopt.maximize(init_points=5, n_iter=25)

# Get the best params
p = xgb_bayesopt.res['max']['max_params']

xgb_params = {
    'n_trees': 250,
    'eta': 0.01,
    'max_depth': int(p['max_depth']),
    'subsample': max(min(p['subsample'], 1), 0),
    'objective': 'reg:linear',
    'base_score': np.mean(Y), # base prediction = mean(target)
    'silent': 1,
    'min_child_weight': int(p['min_child_weight']),
    'gamma': max(p['gamma'], 0),
    'colsample_bytree': max(min(p['colsample_bytree'], 1), 0)
}

model = xgb.train(xgb_params, dtrain, num_boost_round=1500, verbose_eval=False, feval=xgb_r2_score, maximize=True)

Y_Test = model.predict(dtest)

results_df = pd.DataFrame(data={'y':Y_Test}) 
ids = test["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)
