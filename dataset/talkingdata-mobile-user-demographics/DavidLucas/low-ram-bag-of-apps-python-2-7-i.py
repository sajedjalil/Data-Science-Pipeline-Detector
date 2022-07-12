# coding=utf8

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import random

random.seed(1)
# Create bag-of-apps in character string format
# first by event
# then merge to generate larger bags by device

##################
#   App Labels
##################

print("# Read App Labels")
app_lab = pd.read_csv("../input/app_labels.csv", dtype={'device_id': np.str})
app_lab = app_lab.groupby("app_id")["label_id"].apply(
    lambda x: " ".join(str(s) for s in x))

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
app_ev = app_ev.groupby("event_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_lab

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
events["app_lab"] = events["event_id"].map(app_ev)
events = events.groupby("device_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_ev

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("../input/phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")
train = pd.read_csv("../input/gender_age_train.csv",
                    dtype={'device_id': np.str})
train["app_lab"] = train["device_id"].map(events)
train = pd.merge(train, pbd, how='left',
                 on='device_id', left_index=True)

test = pd.read_csv("../input/gender_age_test.csv",
                   dtype={'device_id': np.str})
test["app_lab"] = test["device_id"].map(events)
test = pd.merge(test, pbd, how='left',
                on='device_id', left_index=True)

del pbd
del events

#train.to_csv("train_py.csv", index=False)
#test.to_csv("test_py.csv", index=False)

##################
#   Build Model
##################

# def get_hash_data(df):
#     hasher = FeatureHasher(input_type='string')
#     # hasher = DictVectorizer(sparse=False)
#     df = df[["phone_brand", "device_model", "app_id"]].apply(
#         lambda x: ",".join(str(s) for s in x), axis=1)
#     df = hasher.transform(df.apply(lambda x: x.split(",")))
#     return df


def get_hash_data(train, test):
    df = pd.concat((train, test), axis=0, ignore_index=True)
    split_len = len(train)

    # TF-IDF Feature
    tfv = TfidfVectorizer(min_df=1)
    df = df[["phone_brand", "device_model", "app_lab"]].astype(np.str).apply(
        lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
    df_tfv = tfv.fit_transform(df)

    train = df_tfv[:split_len, :]
    test = df_tfv[split_len:, :]
    return train, test

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)

device_id = test["device_id"].values
train, test = get_hash_data(train,test)

X_train, X_val, y_train, y_val = train_test_split(train, Y, train_size=.92)

##################
#     XGBoost
##################

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

del train
del X_train
del X_val


params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gbtree",
    "eval_metric": "mlogloss",
    "eta": 0.199,
    "silent": 1,
}
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 72, evals=watchlist, verbose_eval=True)

y_pre = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration)

# Write results
result = pd.DataFrame(y_pre, columns=lable_group.classes_)
result["device_id"] = device_id
result = result.set_index("device_id")
result.to_csv('test_Result_70.csv', index=True, index_label='device_id')