# coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
  
from sklearn.ensemble import RandomForestClassifier
 
path1 = '../input/train.csv'
path2 = '../input/test.csv'
train_df = pd.read_csv(path1)
test_df = pd.read_csv(path2)
 
 


features = [col for col in train_df.columns if col not in ['target', 'id']]
print(features)


X_train = train_df[features]
y = train_df['target']

X_test = test_df[features]
test_id = test_df['id']

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y)

y_predict = rfc.predict(X_test)

submission = pd.DataFrame(data= {'Id' : test_id, 'target': y_predict})
submission.to_csv("submission.csv", index=False)


