import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series,DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import xgboost as xgb

train_df   = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')
sns.countplot(x="TARGET", data=train_df)
for feat in train_df.columns:
    if train_df[feat].dtype == 'float64':
        train_df[feat][np.isnan(train_df[feat])] = train_df[feat].mean()
        test_df[feat][np.isnan(test_df[feat])] = test_df[feat].mean()
      
    elif train_df[feat].dtype == 'object':
        train_df[feat][train_df[feat] != train_df[feat]] = train_df[feat].value_counts().index[0]
        test_df[feat][test_df[feat] != test_df[feat]] = test_df[feat].value_counts().index[0]
for feat in train_df.columns:
    if train_df[feat].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_df[feat].values) + list(test_df[feat].values)))
        train_df[feat]   = lbl.transform(list(train_df[feat].values))
        test_df[feat]  = lbl.transform(list(test_df[feat].values))
X_train = train_df.drop(["ID","TARGET"],axis=1)
Y_train = train_df["TARGET"]
X_test  = test_df.drop("ID",axis=1).copy()
# Logistic Regression

logreg = LogisticRegression(class_weight={0:0.2, 1:0.8})

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)[:,1]

logreg.score(X_train, Y_train)
# Create submission

submission = pd.DataFrame()
submission["ID"] = test_df["ID"]
submission["TARGET"] = Y_pred

submission.to_csv('santander.csv', index=False)
