# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

#Separating target variable
train_outcome = train_df["type"]
train_df.drop(["type"], axis=1, inplace=True)

#Encoding the categorical data (color), with the complete set (train and test)
conjunto = pd.concat([train_df[["id", "color"]], test_df[["id", "color"]]])
conjunto_encoded = pd.get_dummies(conjunto, columns=["color"])
train_file = train_df.merge(conjunto_encoded, on="id", how="left")
test_file = test_df.merge(conjunto_encoded, on="id", how="left")
train_df.drop(["color"], axis=1, inplace=True)
test_df.drop(["color"], axis=1, inplace=True)

#Separating IDs
train_id = train_df[["id"]]
test_id = test_df[["id"]]
train_df.drop(["id"], axis=1, inplace=True)
test_df.drop(["id"], axis=1, inplace=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

#Training a haunted RF to get some metrics
X_train, X_val, y_train, y_val = train_test_split(train_df, train_outcome, test_size=0.2)
forest = RandomForestClassifier(n_estimators=200, n_jobs=4)
forest.fit(X_train, y_train)
y_pred_val = forest.predict(X_val)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_val, y_pred_val))
print("Accuracy: {:.1%}".format(accuracy_score(y_val, y_pred_val)))

#Training the Haunted Random Forest with the complete training set
forest = RandomForestClassifier(n_estimators=200, n_jobs=4)
forest.fit(train_df, train_outcome)
y_pred = forest.predict(test_df)

#Predicting of the type of creature haunting the forest
results = pd.read_csv("../input/sample_submission.csv")
results["type"] = y_pred

# Any results you write to the current directory are saved as output.
results.to_csv("submission.csv", index=False)
