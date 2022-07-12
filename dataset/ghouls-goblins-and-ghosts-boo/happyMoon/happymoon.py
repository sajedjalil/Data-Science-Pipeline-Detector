import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

import seaborn as sns
sns.set()

y = df["type"]
indexes_test = df_test["id"]

df = df.drop(["type","color","id"],axis=1)
df_test = df_test.drop(["color","id"],axis=1)

df = pd.get_dummies(df)
df_test = pd.get_dummies(df_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2',C=1000000)
lr.fit(X_train,y_train)
y_pred= lr.predict(X_test) 

print(classification_report(y_pred,y_test))
y_pred = lr.predict(df_test)

Y = pd.DataFrame()
Y["id"] = indexes_test
Y["type"] = y_pred
Y.to_csv("submission.csv",index=False)