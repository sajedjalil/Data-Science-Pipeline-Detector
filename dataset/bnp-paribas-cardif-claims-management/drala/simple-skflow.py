# https://github.com/tensorflow/skflow

import pandas as pd
import numpy as np
import skflow
from scipy.cluster.vq import whiten
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

np.random.seed(42)

train_data = pd.read_csv("data/train.csv")
label_data = train_data['target'].values
train_data.drop(['target'], axis=1, inplace=True)
test_data = pd.read_csv("data/test.csv")

data = pd.concat((train_data, test_data), axis=0, ignore_index=True)
data.drop(['ID','v22'], axis=1, inplace=True)
data.fillna(0, inplace=True)

catagorical_features = []
numeric_features = []

for f in data.columns:
    if data[f].dtype == 'object':
        catagorical_features.append(f)
    else:
        numeric_features.append(f)

data_num = whiten(data[numeric_features])
data_cat = pd.get_dummies(data[catagorical_features], columns=catagorical_features)

trlen = train_data.shape[0]
train = np.hstack((data_num[:trlen], data_cat[:trlen]))
test = np.hstack((data_num[trlen:], data_cat[trlen:]))
labels = label_data.astype(int)

xtrain, xtest, ytrain, ytest = train_test_split(train, labels, train_size=0.7)

model = skflow.TensorFlowDNNClassifier(
    hidden_units = [ 128, 128, 128 ], 
    learning_rate = 0.01,
    n_classes = 2, 
    batch_size = 128,
    steps = 10000
)
model.fit(xtrain, ytrain)
p = model.predict_proba(xtest)[:,1]
print("TensorFlowDNNClassifier log_loss: %0.5f" % (log_loss(ytest, p)))

model.fit(train, labels)
preds = model.predict_proba(test)[:,1]
sample = pd.read_csv("results/sample_submission.csv")
sample.PredictedProb = preds
sample.to_csv("results/simple_skflow_results.csv", index=False)
