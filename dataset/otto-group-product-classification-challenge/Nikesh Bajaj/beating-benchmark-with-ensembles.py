"""
Beating benchmark with ensembles
Otto Group product classification challenge @ Kaggle

__author__ : Nikesh Bajaj
"""

import pandas as pd
import numpy as np
from time import time 
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)
    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss

# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')




# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# train, validation split



# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=.2)

# train a with ensemble
clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, verbose=True, n_jobs=-1)

t0 = time()
clf.fit(X_train, y_train)
print("Fitting Done in %0.3fs" % (time() - t0))

t0 = time()
y_tr_pred = clf.predict_proba(X_train)
print("Done in %0.3fs" % (time() - t0))

t0 = time()
y_ts_pred = clf.predict_proba(X_test)
print("Done in %0.3fs" % (time() - t0))

TrainLoss = multiclass_log_loss(y_train, y_tr_pred, eps=1e-15)
TestLoss =  multiclass_log_loss(y_test, y_ts_pred, eps=1e-15)
print('Multiclass Loss')
print('   Train', TrainLoss)
print('   Test', TestLoss)

# Train with Full Data

clf.fit(train, labels)

# predict on test set
preds = clf.predict_proba(test)

# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('MyModel.csv', index_label='id')
