import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import time

SEED = 42
INPUT_PREFIX = "../input"
IEEE_PATH = f"{INPUT_PREFIX}"
TRAIN_TRANSACTION_PATH = os.path.join(f"{IEEE_PATH}", "train_transaction.csv")
TRAIN_IDENTITY_PATH = os.path.join(f"{IEEE_PATH}", "train_identity.csv")
TEST_TRANSACTION_PATH = os.path.join(f"{IEEE_PATH}", "test_transaction.csv")
TEST_IDENTITY_PATH = os.path.join(f"{IEEE_PATH}", "test_identity.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(f"{IEEE_PATH}", "sample_submission.csv")
TARGET_COLUMN = "isFraud"


class Load:
    def __init__(self):
        self

    def load(self, path, index_col=None):
        return pd.read_csv(path, index_col=index_col)

    def load_train_transaction(self):
        return self.load(TRAIN_TRANSACTION_PATH)

    def load_train_identity(self):
        return self.load(TRAIN_IDENTITY_PATH)

    def load_test_transaction(self):
        return self.load(TEST_TRANSACTION_PATH)

    def load_test_identity(self):
        return self.load(TEST_IDENTITY_PATH)
    
    def load_sample_submission(self, index_col="TransactionID"):
        return self.load(SAMPLE_SUBMISSION_PATH, index_col=index_col)


load = Load()
train_trans = load.load_train_transaction()
train_identity = load.load_train_identity()
train = pd.merge(train_trans, train_identity, how="left", on="TransactionID")
del train_trans, train_identity

test_trans = load.load_test_transaction()
test_identity = load.load_test_identity()
test = pd.merge(test_trans, test_identity, how="left", on="TransactionID")
del test_trans, test_identity

not_used_features = ['TransactionID', 'TransactionDT']

# Drop target, fill in NaNs
y_train = train[TARGET_COLUMN]
X_train = train.copy()
X_test = test.copy()
X_train.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)
del train, test

# Label Encoding
for f in X_train.columns:
    if f == TARGET_COLUMN:
        continue
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

# Split
TransactionID_split = X_train.TransactionID.quantile(q=0.7)
X_tr, X_val = X_train[X_train.TransactionID <= TransactionID_split], X_train[X_train.TransactionID > TransactionID_split]
y_tr, y_val = y_train[:len(X_tr)], y_train[len(X_tr):]
del X_train, y_train

# Sampling
print(f"original train length: {len(X_tr)}")
original_length = len(X_tr)
original_positive_rate = X_tr[TARGET_COLUMN].sum() / len(X_tr)
negative_sampling_rate = 0.1
train_positive = X_tr[X_tr[TARGET_COLUMN]==1]
train_negative = X_tr[X_tr[TARGET_COLUMN]==0].sample(n=int(original_length * (1 - original_positive_rate) * negative_sampling_rate), random_state=SEED)
X_tr = pd.concat((train_positive, train_negative))
del train_positive, train_negative
sampled_positive_rate = X_tr[TARGET_COLUMN].sum() / len(X_tr)
print(f"sampled train length: {len(X_tr)}")
print(f"original positive_rate: {original_positive_rate}")
print(f"sampled positive_rate: {sampled_positive_rate}")

y_tr = y_tr[X_tr.index]

X_tr.drop(not_used_features+[TARGET_COLUMN], axis=1, inplace=True)
X_val.drop(not_used_features+[TARGET_COLUMN], axis=1, inplace=True)
X_test.drop(not_used_features, axis=1, inplace=True)

# Modeling
t0 = time.time()
clf = xgb.XGBClassifier(n_estimators=10000,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999)
clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric="logloss", early_stopping_rounds=100, verbose=1000)
print("Modeling Runtime:", round((time.time() - t0)/60,1), "[min.]")
validation_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
print(f"validation_score is {validation_score}")

# Predict
prediction = clf.predict_proba(X_test)[:,1]
sns.distplot(prediction)

# Submit
submission = load.load_sample_submission()
submission[TARGET_COLUMN] = prediction
submission.to_csv('submission_sampling.csv')
print(submission.head(20))
