# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

train=pd.read_csv("../input/train.csv")

train.drop("id",axis=1,inplace=True)

test=pd.read_csv("../input/test.csv")

y_train=train["target"]
x_train=train.drop("target",axis=1)
test_id=test["id"]
x_test=test.drop("id",axis=1)

lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     C=0.1, penalty="l1")

lr.fit(x_train,y_train)

submission=pd.DataFrame(columns=["id","target"])

submission["id"]=test_id

submission["target"]=1-lr.predict_proba(x_test)

submission.head(100)

submission.to_csv("submission.csv",index=False)