# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time
import warnings
warnings.filterwarnings('ignore')
print('oooooooo tuning noweeeeeee')
# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestRegressor, AdaBoostClassifier, 
                              GradientBoostingRegressor, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


train_df = pd.read_csv('../input/train.csv', parse_dates=["activation_date"])
test_df = pd.read_csv('../input/test.csv', parse_dates=["activation_date"])
y_train = train_df["deal_probability"].values

train_df["price_new"] = train_df["price"].values
train_df["price_new"].fillna(np.nanmean(train_df["price"].values), inplace=True)

train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))

tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['title'].values.tolist() + test_df['title'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title'].values.tolist())

### SVD Components ###
n_comp = 3
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

train_df["description"].fillna("NA", inplace=True)
test_df["description"].fillna("NA", inplace=True)

train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(x.split()))
test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(x.split()))

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000)
full_tfidf = tfidf_vec.fit_transform(train_df['description'].values.tolist() + test_df['description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['description'].values.tolist())

### SVD Components ###
n_comp = 3
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

train_y = train_df["deal_probability"].values
y_train = train_df["deal_probability"].values
test_id = test_df["item_id"].values

# New variable on weekday #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    
cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image"]
train_X = train_df.drop(cols_to_drop + ["price_new", "deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_df["price_new"] = train_df["price"].values
train_df["price_new"].fillna(np.nanmean(train_df["price"].values), inplace=True)

train_df["image_top_1_new"] = train_df["image_top_1"].values
train_df["image_top_1_new"].fillna(np.nanmedian(train_df["image_top_1"].values), inplace=True)

test_df["price_new"] = test_df["price"].values
test_df["price_new"].fillna(np.nanmean(test_df["price"].values), inplace=True)

test_df["image_top_1_new"] = test_df["image_top_1"].values
test_df["image_top_1_new"].fillna(np.nanmedian(test_df["image_top_1"].values), inplace=True)

train_X = train_df.drop(["price", "deal_probability", "image_top_1", "image"], axis=1)
test_X = test_df.drop(["price", "image_top_1", "image"], axis=1)

cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image", 'item_id', 'user_id']
train_df.drop(cols_to_drop, axis=1, inplace = True)
test_df.drop(cols_to_drop, axis=1, inplace = True)

train_df.drop("deal_probability", axis = 1, inplace = True)
train_df.drop("price", axis = 1, inplace = True)
test_df.drop("price", axis = 1, inplace = True)

train_df.drop("image_top_1", axis = 1, inplace = True)
test_df.drop("image_top_1", axis = 1, inplace = True)

train_X = train_df
test_X = test_df
dev_X = train_X.iloc[:-200000,:]
val_X = train_X.iloc[-200000:,:]
dev_y = train_y[:-200000]
val_y = train_y[-200000:]
print(dev_X.shape, val_X.shape, test_X.shape)

# Some useful parameters which will come in handy later on
ntrain = dev_X.shape[0]
ntest = test_X.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        print("training start")
        clf.train(x_tr, y_tr)
        print("training complete")
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        print("predict complete")
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    
'''
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}
'''
# rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 25, cv = 3, verbose=2, random_state=7, n_jobs = -1)
# y_train = train_df["deal_probability"].values
x_train = dev_X.values
y_train = dev_y
x_test = test_X.values

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
print('training started')
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
best_random = rf_random.best_estimator_
rf_random.predict(x_test)

rf_feature = best_random.feature_importances(x_train,y_train)

# rf_ans = rf.predict(x_test)

rf_ans[rf_ans>1] = 1
rf_ans[rf_ans<0] = 0
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = rf_ans
sub_df.to_csv("baseline_tuned_rf.csv", index=False)