__author__ = 'alled'
import sqlite3
import zipfile
import pandas as pd
import numpy as np
from pandas.io import sql
from sklearn.linear_model import LogisticRegression
from datetime import datetime

conn = sqlite3.connect('../input/database.sqlite')
print("Connected to DB")

#2 part

query1 = """
select tss.HistCTR, tss.Position, si.IsUserLoggedOn, si.CategoryID, tss.IsClick from trainSearchStream tss left join SearchInfo si on tss.SearchID = si.SearchID where ObjectType=3 and IsClick=1 limit 20000000 offset 20000000
"""
#query1 = """
#select tss.HistCTR, tss.Position, si.IsUserLoggedOn, si.CategoryID, tss.IsClick from trainSearchStream tss where tss.IsClick=1 UNION ALL SearchInfo si on tss.SearchID = si.SearchID where ObjectType=3 limit 20000000 offset 20000000
#"""

print("first query")

#print(query1)

df = pd.read_sql(query1, conn)
print("Created dataframe")
#print(df)

# Output to csv
filename = 'query1_1.csv'
print("output")
pd.DataFrame(df).to_csv(filename, index=False)
print("finished")