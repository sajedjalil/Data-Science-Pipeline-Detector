__author__ = 'alxpks'

import sqlite3
import datetime
import pandas as pd
import numpy as np
from pandas.io import sql
import sklearn
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import feature_extraction

conn = sqlite3.connect('../input/database.sqlite')

# Get train data
query_train_true = """
select 
    tss.HistCTR, tss.Position, si.IsUserLoggedOn, si.CategoryID, tss.IsClick , tss.ObjectType
from 
    trainSearchStream tss 
left join 
    SearchInfo si on tss.SearchID = si.SearchID 
where 
    IsClick = 1 and ObjectType = 3
""" # 1146289
query_train_false = """
select 
    tss.HistCTR, tss.Position, si.IsUserLoggedOn, si.CategoryID, tss.IsClick, tss.ObjectType
from 
    trainSearchStream tss 
left join 
    SearchInfo si on tss.SearchID = si.SearchID 
where 
    IsClick = 0 and ObjectType = 3
limit 
    1e7;
"""

df = pd.concat((sql.read_sql(query_train_false, conn), sql.read_sql(query_train_true, conn)))
print(df.head())
print('####')
print(df[df.IsClick==1].head())
print(df.describe())
print('####')
print(df[df.IsClick==0].head())
# X_train = df[['HistCTR', 'Position', 'IsUserLoggedOn', 'CategoryID']].values
# y_train = df.IsClick.values
# X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
# print('y_train:', y_train[:10])
# print('X_train:', X_train[:10])

# # Get test data
# query_test = """
# select 
#     tss.TestID, tss.HistCTR, tss.Position, si.IsUserLoggedOn, si.CategoryID 
# from 
#     testSearchStream tss 
# left join 
#     SearchInfo si on tss.SearchID = si.SearchID 
# where 
#     ObjectType = 3;
# """

# df_test = sql.read_sql(query_test, conn)
# X_test = df_test[['HistCTR', 'Position', 'IsUserLoggedOn', 'CategoryID']].values

# encoder =  preprocessing.OneHotEncoder(categorical_features=[1,2,3])
# X_train = encoder.fit_transform(X_train)
# X_test = encoder.transform(X_test)
# print('X_train shape:', X_train.shape)

# # Learn
# model = pipeline.Pipeline([
#     #('encoder', preprocessing.OneHotEncoder(categorical_features=[1,2,3])),
#     #('normalizer', preprocessing.Normalizer()),
#     ('clf', linear_model.LogisticRegression(C=11))])

# model.fit(X_train, y_train)
# pred = model.predict_proba(X_test)
# print(model.classes_, 'argmax:', model.classes_.argmax())

# # Output to csv
# filename = 'submission.csv'
# pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, model.classes_.argmax()]}).to_csv(filename, index=False)
