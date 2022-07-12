import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.calibration import CalibratedClassifierCV

# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
labels = preprocessing.LabelEncoder().fit_transform(labels)
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# train a random forest classifier without calibration
#clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
#clf.fit(train, labels)
#preds = clf.predict_proba(test)

# train a random forest classifier with calibration
clf = ensemble.RandomForestClassifier(n_estimators=180, n_jobs=-1)
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic')
calibrated_clf.fit(train, labels)
preds = calibrated_clf.predict_proba(test)

# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('benchmark.csv', index_label='id')