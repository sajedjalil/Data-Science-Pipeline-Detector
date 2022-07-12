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
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

def evaluation(label,pred_label):
    num = len(label)
    logloss = 0.0
    for i in range(num):
        p = max(min(pred_label[i][label[i]-1],1-10**(-15)),10**(-15))
        logloss += np.log(p)
    logloss = -1*logloss/num
    return logloss

conn = sqlite3.connect('../input/database.sqlite')

# Get train data
query1 = """
select Position, HistCTR, IsClick from trainSearchStream where IsClick=1 limit 6000 offset 12345;
"""
query0 = """
select Position, HistCTR, IsClick from trainSearchStream where IsClick=0 limit 1000000 offset 12345678;
"""
df = pd.concat([sql.read_sql(query0, conn), sql.read_sql(query1, conn)])
print (df)
X = df[['Position', 'HistCTR']]
y = df.IsClick

val_data = X[0:20000]
val_label = y[0:20000]
train_data = X[20000:]
train_label = y[20000:]


# Get test data
query_test = """
select TestID, Position, HistCTR from testSearchStream where ObjectType = 3
"""
df_test = sql.read_sql(query_test, conn)
X_test = df_test[['Position', 'HistCTR']]

# Learn
model =  RandomForestClassifier()
model.fit(train_data, train_label)

val_pred_label = model.predict_proba(val_data)
logloss = evaluation(val_label,val_pred_label)
print ("logloss of validation set:",logloss)


pred = model.predict_proba(X_test)

# Output to csv
filename = 'submission.csv'
pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

# Zip
# with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
#     z.write(filename)






