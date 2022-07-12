'''
CREDITS:

Nick Brooks : https://www.kaggle.com/nicapotato/lgbm-cv-tuning-and-seed-diversification/notebook
the1owl : https://www.kaggle.com/the1owl/love-is-the-answer

Correlation between these two is about 0.70, so, blending them, might get better score. One scores 1.44 (LGB) other scores 1.47 (CB + decomposition features). 
'''

print('merge3')
# 1. LightGBM
import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random
random.seed(2018)
from sklearn import *
# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split

# Gradient Boosting
import lightgbm as lgb

# Specify index/ target name
id_col = "ID"
target_var = "target"

# House Keeping Parameters
Debug = False
Home = False
Build_Results_csv = False # if running for first time

results = pd.DataFrame(columns = ["Rounds","Score","STDV", "LB", "Parameters"])
print("Data Load Stage")
training = pd.read_csv('../input/train.csv', index_col = id_col)
if Debug is True : training = training.sample(100)
traindex = training.index
testing = pd.read_csv('../input/test.csv', index_col = id_col)
if Debug is True : testing = testing.sample(100)
testdex = testing.index

y = np.log1p(training[target_var])
training.drop(target_var,axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

# Modeling Datasets
test_df = df.loc[testdex,:]
vocab = df.columns

# LGBM Dataset
lgtrain = lgb.Dataset(df.loc[traindex,vocab],y ,feature_name = "auto")
print("Starting LightGBM. Train shape: {}, Test shape: {}".format(df.loc[testdex,:].shape,test_df.shape))
print("Feature Num: ",len(vocab))
del df; gc.collect();

print("Light Gradient Boosting Regressor: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 200,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing':True
}

modelstart= time.time()
# Find Optimal Parameters / Boosting Rounds
lgb_cv = lgb.cv(
    params = lgbm_params,
    train_set = lgtrain,
    num_boost_round=2500,
    stratified=False,
    nfold = 5,
    verbose_eval=50,
    seed = 23,
    early_stopping_rounds=75)

optimal_rounds = np.argmin(lgb_cv['rmse-mean'])
best_cv_score = min(lgb_cv['rmse-mean'])

print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
    optimal_rounds,best_cv_score,lgb_cv['rmse-stdv'][optimal_rounds]))

results = results.append({"Rounds": optimal_rounds,
                          "Score": best_cv_score,
                          "STDV": lgb_cv['rmse-stdv'][optimal_rounds],
                          "LB": None,
                          "Parameters": lgbm_params}, ignore_index=True)
        
learning_rates = [0.012,0.008,0.016]
for param in learning_rates:
    print("Learning Rate: ", param)
    modelstart= time.time()
    lgbm_params["learning_rate"] = param
    # Find Optimal Parameters / Boosting Rounds
    lgb_cv = lgb.cv(
        params = lgbm_params,
        train_set = lgtrain,
        num_boost_round=10000,
        stratified=False,
        nfold = 5,
        verbose_eval=200,
        seed = 23,
        early_stopping_rounds=75)

    optimal_rounds = np.argmin(lgb_cv['rmse-mean'])
    best_cv_score = min(lgb_cv['rmse-mean'])

    print("Optimal Round: {}\nOptimal Score: {} + {}".format(
        optimal_rounds,best_cv_score,lgb_cv['rmse-stdv'][optimal_rounds]))

    results = results.append({"Rounds": optimal_rounds,
                              "Score": best_cv_score,
                              "STDV": lgb_cv['rmse-stdv'][optimal_rounds],
                              "LB": None,
                              "Parameters": lgbm_params}, ignore_index=True)
    if Home is True:
        with open('results.csv', 'a') as f:
            results.to_csv(f, header=False)
        # results = pd.read_csv("results.csv")

final_model_params = results.iloc[results["Score"].idxmin(),:]["Parameters"]
optimal_rounds = results.iloc[results["Score"].idxmin(),:]["Rounds"]

allmodelstart= time.time()
# Run Model with different Seeds
multi_seed_pred = dict()
all_feature_importance_df  = pd.DataFrame()

all_seeds = [27,22,300,401,7]
for seeds_x in all_seeds:
    modelstart= time.time()
    print("Seed: ", seeds_x,)
    # Go Go Go
    final_model_params["seed"] = seeds_x
    lgb_reg = lgb.train(
        final_model_params,
        lgtrain,
        num_boost_round = optimal_rounds + 1,
        verbose_eval=200)

    # Feature Importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = vocab
    fold_importance_df["importance"] = lgb_reg.feature_importance()
    all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

    multi_seed_pred[seeds_x] =  list(lgb_reg.predict(test_df))
    del lgb_reg

print("All Model Runtime: %0.2f Minutes"%((time.time() - allmodelstart)/60))

sub_preds = pd.DataFrame.from_dict(multi_seed_pred).replace(0,0.000001)
del multi_seed_pred; gc.collect();

lgb_ans = np.expm1(sub_preds.mean(axis=1))
mean_sub = np.expm1(sub_preds.mean(axis=1).rename(target_var))
mean_sub.index = testdex

# Submit
mean_sub.to_csv('lgb.csv'
            ,index = True, header=True)
print("Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

# 2. CatBoost + decomposition features
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

print("Load data...")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))
col = [c for c in train.columns if c not in ['ID', 'target']]

scl = preprocessing.StandardScaler()
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train.target.values, test_size=0.10, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs = -1, random_state = 7)
model.fit(scl.fit_transform(x1), y1)
print(rmsle(y2, model.predict(scl.transform(x2))))

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])[:600]['feature'].values

#Added Columns from feature_selection
train = train[['ID', 'target']+list(col)]
test = test[['ID']+list(col)]
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))

PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 20            ### Number of decomposition components ###

target = np.log1p(train['target']).values
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop \
    and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

print('\nModelling...')
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_x, trn_y = train.ix[trn_idx], target[trn_idx]
    val_x, val_y = train.ix[val_idx], target[val_idx]
    cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4, l2_leaf_reg=20, bootstrap_type='Bernoulli', subsample=0.6, eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
    cb_model.fit(trn_x, trn_y, eval_set=(val_x, val_y), cat_features=[], use_best_model=True, verbose=True)
    oof_preds[val_idx] = cb_model.predict(val_x)
    sub_preds += cb_model.predict(test) / folds.n_splits
    print("Fold %2d RMSLE : %.6f" % (n_fold+1, rmsle(np.exp(val_y)-1, np.exp(oof_preds[val_idx])-1)))

print("Full RMSLE score %.6f" % rmsle(np.exp(target)-1, np.exp(oof_preds)-1)) 
subm['target'] = np.exp(sub_preds)-1
cb_ans = np.exp(sub_preds)
subm.to_csv('CB_PCA_and_stuff.csv', index=False)

print('merging..')
print(np.corrcoef([lgb_ans, cb_ans]))
ensemble_ans = lgb_ans * 0.5 + cb_ans * 0.5
subm['target'] = ensemble_ans
subm.to_csv('lgb_cb.csv', index=False)