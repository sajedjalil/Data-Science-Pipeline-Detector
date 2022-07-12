# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# import training data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# submit_exp = pd.read_csv("../input/sample_submission.csv")
# print(submit_exp.head())

# train and test
id_train = train["id"]
y_train = train["target"]
X_train = train.drop(["id","target"], axis=1)

id_test = test["id"]
X_test = test.drop(["id"], axis=1)

# print(X_train.head())
# print(X_test.head())

# -1 means nan in this case...so put nan back
X_train = X_train.replace(-1, np.NaN)
X_test = X_test.replace(-1, np.NaN)

# concatenate train and test to deal with nan together
Xmat = pd.concat([X_train, X_test])

# # visualize the number of nans in each column
# # (shamelessly adapted from:
# #https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial)
# import missingno as msno
#
# msno.matrix(df=X_train.iloc[:,:39], figsize=(20,14), color=(0.5,0,0))

# Columns with many nans itself may be meaningful
def nan2bi(x):
    if np.isnan(x):
        return 1
    else:
        return 0

Xmat = pd.concat([X_train, X_test])
cols = ["ps_reg_03","ps_car_03_cat","ps_car_05_cat"]
for c in cols:
    Xmat[c + "_isnan"] = Xmat[c].apply(nan2bi)

# For other columns replace nan with median
Xmat = Xmat.fillna(Xmat.median())

# remove other columns with nan, if any
Xmat = Xmat.dropna(axis=1)
print(Xmat.shape)

# # some of binary variables can be skewed
# bin_col = [col for col in Xmat.columns if '_bin' in col]
# counts = []
# for col in bin_col:
#     counts.append(100*(Xmat[col]==1).sum()/Xmat.shape[0])
#
# ax = sns.barplot(x=counts, y=bin_col, orient='h')
# ax.set(xlabel="% of 1 in a column")
# plt.show()

# upon visual inspection, some columns with skewed data are removed
Xmat = Xmat.drop(["ps_ind_10_bin","ps_ind_11_bin","ps_ind_13_bin"], axis=1)
# print(Xmat.shape)

# # check correlation matrix
# sns.set(style="white")
#
# # Compute the correlation matrix (let's put y_train back this time)
# Xcorrmat = Xmat.iloc[:X_train.shape[0],:]
# Xcorrmat['target'] = y_train
# corr = Xcorrmat.corr()
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(20, 12))
#
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# plt.show()

# drop all the "_calc", as there being no correlation with others
calc_col = [col for col in Xmat.columns if '_calc' in col]
Xmat = Xmat.drop(calc_col, axis=1)

# outlier deletion
Xmat = Xmat.drop(['ps_ind_12_bin','ps_ind_14','ps_car_10_cat'], axis=1)

# interaction
Xmat['interaction'] = Xmat['ps_car_13']*Xmat['ps_reg_03']

# zscoring
X_train = Xmat.iloc[:X_train.shape[0],:]
X_test = Xmat.iloc[X_train.shape[0]:,:]
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

# # vizualize
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(X_train, cmap=cmap)
# plt.show()

# # vizualize
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(X_train, cmap=cmap)
# plt.show()

# # vizualize
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(X_train, cmap=cmap)
# plt.show()

# # feature importance using random forest
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
# rf.fit(X_train, y_train)
#
# print('Training done using Random Forest')
#
# ranking = np.argsort(-rf.feature_importances_)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
# ax.set_xlabel("feature importance")
# plt.show()

# # pairplot
# Xpair = X_train.iloc[:,ranking[:10]]
# Xpair['target'] = y_train
#
# sns.pairplot(Xpair, hue='target')

# # dimensioanlity reduction and visualization
# Xdr = X_train
# Xdr['target'] = y_train
# Xdr1 = Xdr.loc[y_train==1, :]
# Xdr0 = Xdr.loc[y_train==0, :]
#
# print('rows for target 1: ' + str(Xdr1.shape[0]))
# print('rows for target 0: ' + str(Xdr0.shape[0]))
#
# # random sampling from X_train0, as the target value distribution is skewed
# # use N = 10,000 samples for now
# N = 10000
# np.random.seed(20171021)
# Xdr0 = Xdr0.iloc[np.random.choice(Xdr0.shape[0], N), :]
# Xdr1 = Xdr1.iloc[np.random.choice(Xdr1.shape[0], N), :]
#
# Xdr = pd.concat([Xdr0, Xdr1])

# # pairplot
# Xpair =pd.concat([Xdr.iloc[:,ranking[:7]], Xdr['target']], axis=1)
#
# ax = sns.pairplot(Xpair, hue='target')
# plt.show()


# Xdr = Xdr.drop(['target'], axis=1)
#
# # PCA
# from sklearn.decomposition import PCA
#
# pcamat = PCA(n_components=2).fit_transform(Xdr)
#
# plt.figure()
# plt.scatter(pcamat[:Xdr0.shape[0],0],pcamat[:Xdr0.shape[0],1], c='b', label='targ 0', alpha=0.3)
# plt.scatter(pcamat[Xdr0.shape[0]:,0],pcamat[Xdr0.shape[0]:,1],c='r', label='targ 1', alpha=0.3)
# plt.legend()
# plt.title('PC space')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.tight_layout()
# plt.show()
# print("PCA done")


# # TSNE
# from sklearn.manifold import TSNE
#
# tsnemat = TSNE(n_components=2, random_state=0).fit_transform(Xdr)
#
# plt.figure()
# plt.scatter(tsnemat[:Xdr0.shape[0],0],tsnemat[:Xdr0.shape[0],1], c='b', label='targ 0', alpha=0.3)
# plt.scatter(tsnemat[Xdr0.shape[0]:,0],tsnemat[Xdr0.shape[0]:,1],c='r', label='targ 1', alpha=0.3)
# plt.legend()
# plt.title('TSNE space')
# plt.xlabel('dim 1')
# plt.ylabel('dim 2')
# plt.tight_layout()
# plt.show()
# print("TSNE done")

# let's use Gini coefficient for this skewed dataset...
# as it quantifies "inequality" :D
def gini(y, y_pred):
    assert(len(y) == len(y_pred))
    g = np.asarray(np.c_[y, y_pred, np.arange(np.size(y))], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum()/g[:,0].sum()
    return gs/np.size(y)

def gini_normalized(a,b):
    return gini(a,b)/gini(a,a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    score = gini_normalized(labels, preds)
    return 'gini', score

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

# XGBoost
import xgboost as xgb
params_xgb = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.9, 'seed': 6,\
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

# lightGBM
import lightgbm as lgb
params_lgb = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':10, 'max_bin':10,  'objective': 'binary',
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10,  'min_data': 500}

# # logistic regression
# from sklearn.linear_model import LogisticRegression
# regr = LogisticRegression(random_state=None, max_iter=10000)

# stratified k-fold
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.metrics import f1_score

# #random forest classifier
# clf = RandomForestClassifier(n_estimators=150, max_features="sqrt")

# because the target is skewed, stratified k-fold is used
# also prediction is based on the average output from separately-trained models
# (ensembling)
nsplit = 3
skf = StratifiedKFold(n_splits=nsplit, random_state=None, shuffle=False)
# rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=None)

# pandas to numpy
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values

y_pred_xgb = np.zeros((len(id_test),nsplit))
y_pred_lgb = np.zeros((len(id_test),nsplit))
y_pred_logreg = np.zeros((len(id_test),nsplit))
for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print('kfold: ' + str(i))
        Xcv_train, Xcv_test = X_train[train_index], X_train[test_index]
        ycv_train, ycv_test = y_train[train_index], y_train[test_index]

        # XGB
        dtrain = xgb.DMatrix(Xcv_train, ycv_train)
        dtest = xgb.DMatrix(Xcv_test, ycv_test)
        watchlist = [(dtrain, 'train'),(dtest, 'test')]
        xgb_model = xgb.train(params_xgb, dtrain, 1000, watchlist, early_stopping_rounds=70,\
            feval=gini_xgb, maximize=True, verbose_eval=100)
        y_pred_xgb[:,i] = xgb_model.predict(xgb.DMatrix(X_test), \
            ntree_limit=np.round((xgb_model.best_ntree_limit+50)/2*nsplit).astype(int))

        # lightGBM
        lgb_model = lgb.train(params_lgb, lgb.Dataset(Xcv_train, label=ycv_train), 1000,
                  lgb.Dataset(Xcv_test, label=ycv_test), verbose_eval=100,
                  feval=gini_lgb, early_stopping_rounds=70)
        y_pred_lgb[:,i] = lgb_model.predict(X_test, num_iteration=np.round((lgb_model.best_iteration+50)/2*nsplit).astype(int))

# y_pred = 0.5*np.mean(y_pred_xgb, axis=1) + 0.4*np.mean(y_pred_lgb, axis=1) + 0.1*np.mean(y_pred_logreg, axis=1)
y_pred = 0.6*np.mean(y_pred_xgb, axis=1) + 0.4*np.mean(y_pred_lgb, axis=1)

print("model prediction using stratified cross-validation done")

# for submission
submission = pd.DataFrame({
    "id": id_test,
    "target": y_pred
    })
submission.to_csv('safedriver.csv', index=False)