"""
Logistic Regression

https://www.kaggle.com/olivermeyfarth/avito-context-ad-clicks/logistic-regression-on-histctr/run/19490
"""

import sqlite3
import datetime
import zipfile
import pandas as pd
import numpy as np
from pandas.io import sql
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss

conn = sqlite3.connect('../input/database.sqlite')

# Get train data
# query1 = """
# select count(*) from trainSearchStream where ObjectType = 3;
# """
# df1 = sql.read_sql(query1, conn)
# print(df1.describe())

query = """
select Position, HistCTR, IsClick from trainSearchStream where ObjectType = 3 limit 10000000;
"""
df = sql.read_sql(query, conn)
df.to_csv("train_10000000.csv", index=False)
# print(df.describe())
# X = df[['Position', 'HistCTR']]
# y = df.IsClick


# query_t = """
# select Position, HistCTR, IsClick from trainSearchStream where ObjectType = 3 limit 150000000, 7000000;
# """
# df_t = sql.read_sql(query_t, conn)
# print("df_t",df_t.describe())
# X_t = df_t[['Position', 'HistCTR']]


# print("X shape", X.shape)
# print("y Shape", y.shape)
# print("y max", np.max(y))
# print("y median", np.median(y))
# print("y mean", np.mean(y))
# print("y min", np.min(y))

# Get test data
query_test = """
select TestID, Position, HistCTR from testSearchStream where ObjectType = 3 ;
"""
df_test = sql.read_sql(query_test, conn)
df_test.to_csv("test.csv", index=False)
# X_test = df_test[['Position', 'HistCTR']]
# print("X_test shape", X_test.shape)

# # Learn
# model = LogisticRegression(verbose=True, C=1)
# model.fit(X, y)

# pred_train = model.predict_proba(X)[:,1]
# print ("train logloss", log_loss(y, pred_train))


# pred_t = model.predict_proba(X_t)[:,1]
# print("pred_t max", np.max(pred_t))
# print("pred_t median", np.median(pred_t))
# print("pred_t mean", np.mean(pred_t))
# print("pred_t min", np.min(pred_t))
# print (log_loss(df_t.IsClick, pred_t))
# print (pred_t)

# pred = model.predict_proba(X_test)[:,1]
# print("pred max", np.max(pred))
# print("pred median", np.median(pred))
# print("pred mean", np.mean(pred))
# print("pred min", np.min(pred))

# # Output to csv
# filename = 'submission.csv'
# pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred}).to_csv(filename, index=False)

# Zip
# with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
#     z.write(filename)
