# Inspired by:
#SRK script
#https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python
#Faron script

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

SEED = 42
NFOLDS = 5
n_classes=3

data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour"])

categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
            
            
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])


train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
y_train = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

x_train = train_X.toarray()
x_test = test_X.toarray()

ntrain = x_train.shape[0]
ntest = x_test.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


## Creating Classes for stacking


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict_proba(self, x):
        proba = self.clf.predict_proba(x)
        return proba


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 30)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict_proba(self, x):
        proba = self.gbdt.predict(xgb.DMatrix(x))
        return proba


def get_oof(clf):
    oof_train = np.zeros((ntrain,n_classes))
    oof_test = np.zeros((ntest,n_classes))
    oof_test_skf = np.empty((ntest, NFOLDS*n_classes))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict_proba(x_te)
        oof_test_skf[:,3*i: 3*i+3] = clf.predict_proba(x_test)
        
    for i in range(3):
        oof_test[:,i] = (oof_test_skf[:,i]+oof_test_skf[:,i+3]+oof_test_skf[:,i+6]+oof_test_skf[:,i+9]+oof_test_skf[:,i+12])/5

    
    return oof_train, oof_test


et_params = {
    'n_jobs': 16,
    'n_estimators': 10,
    'max_features': 0.5,
    'max_depth': 12,
    #'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 10,
    'max_features': 0.2,
    'max_depth': 12,
    #'min_samples_leaf': 2,
}

xgb_params = {
    'objective': 'multi:softprob',
    'eta':0.1,
    'max_depth':6,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}


xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)


print("XG-CV: {}".format(log_loss(y_train, xg_oof_train)))
print("ET-CV: {}".format(log_loss(y_train, et_oof_train)))
print("RF-CV: {}".format(log_loss(y_train, rf_oof_train)))


x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'objective': 'multi:softprob',
    'eta':0.1,
    'max_depth':2,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=5, seed=SEED,
             early_stopping_rounds=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]


print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

out_df = pd.DataFrame(gbdt.predict(dtest))
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv('xgstacker_starter.sub.csv', index=False)