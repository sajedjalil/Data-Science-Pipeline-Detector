import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import xgboost as xgb
from sklearn.metrics import roc_auc_score,roc_curve

train_pd = pd.read_csv('../input/train.csv')
test_pd = pd.read_csv('../input/test.csv')

Train_data = train_pd.values
X_train, y_train = Train_data[:, 1:-1], Train_data[:, -1]

Test_data = test_pd.values
X_test = Test_data[:, 1:-1]

import matplotlib.pyplot as plt

pca = PCA(n_components=40)
pca.fit(X_train)
X_train_ = pca.fit_transform(X_train)
X_test_ = pca.fit_transform(X_test)

RFC = RandomForestClassifier(n_estimators = 500)
RFC.fit(X_train_, y_train)

y_test = RFC.predict(X_test_)
test_id = test_pd['ID'].values

submission = pd.DataFrame({"ID": test_id, "TARGET":y_test})
submission.to_csv("submission_rfc.csv", index=False)

XGBc = xgb.XGBClassifier(n_estimators = 500)
XGBc.fit(X_train_, y_train)

y_test = XGBc.predict(X_test_)

submission = pd.DataFrame({"ID": test_id, "TARGET":y_test})
submission.to_csv("submission_xgbc.csv", index=False)