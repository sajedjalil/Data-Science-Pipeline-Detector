import pandas as pd
from sklearn import linear_model
import os
from sklearn import ensemble, feature_extraction, preprocessing

from sklearn.calibration import CalibratedClassifierCV


os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print(train.head())


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')


labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

logreg = linear_model.LogisticRegression()
# logreg.fit(train, labels)
# logreg = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, max_features = 50, verbose=2)
logreg = CalibratedClassifierCV(logreg, method='isotonic', cv=5)
logreg.fit(train, labels)

preds = logreg.predict_proba(test)
# print(preds)
# print("asas")

preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('benchmark.csv', index_label='id')