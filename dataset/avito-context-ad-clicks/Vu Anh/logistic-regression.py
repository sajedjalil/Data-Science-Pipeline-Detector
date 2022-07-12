"""
Logistic Regression

In testSearchStream
# of IsClick == 1 : 1146289
# of IsClick == 0 : 189011446

cf.
https://www.kaggle.com/olivermeyfarth/avito-context-ad-clicks/logistic-regression-on-histctr/run/19490
"""

import sqlite3
import zipfile
import pandas as pd
import numpy as np
from pandas.io import sql
from sklearn.linear_model import LogisticRegression

conn = sqlite3.connect('../input/database.sqlite')

# Get train data
query1 = """
select Position, HistCTR, IsClick from trainSearchStream where IsClick=1 limit 600000;
"""
query0 = """
select Position, HistCTR, IsClick from trainSearchStream where IsClick=0 limit 10000000;
"""
df = pd.concat([sql.read_sql(query0, conn), sql.read_sql(query1, conn)])
X = df[['Position', 'HistCTR']]
y = df.IsClick

# Get test data
query_test = """
select TestID, Position, HistCTR from testSearchStream where ObjectType = 3
"""
df_test = sql.read_sql(query_test, conn)
X_test = df_test[['Position', 'HistCTR']]

# Learn
model = LogisticRegression()
model.fit(X, y)
pred = model.predict_proba(X_test)

# Output to csv
filename = 'submission.csv'
pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)
