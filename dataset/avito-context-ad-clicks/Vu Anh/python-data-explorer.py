import sqlite3
import zipfile
import pandas as pd
import numpy as np
from pandas.io import sql
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt


conn = sqlite3.connect('../input/database.sqlite')

# Get train data
# select count(*) from trainSearchStream;
# select count(*) from trainSearchStream where IsClick = 1;
# select count(*) from trainSearchStream where IsClick != 1;

query_train = """
select * from trainSearchStream limit 6000;
"""

df = sql.read_sql(query_train, conn)
print(df.tail())
print(df.describe())

# print('Number of Searches: 392356948')
# print('- There are 1146289 Clicks and 391210659 Ignorations')

# from pylab import *
# figure(1, figsize=(6,6))
# labels = 'clicks', 'ignorations'
# fracs = [1146289, 391210659]
# explode=(0, 0.05)
# pie(fracs, explode=explode, labels=labels,
#     autopct='%1.1f%%', shadow=True, startangle=90)
# title('Click or Ignoration', bbox={'facecolor':'0.8', 'pad':5})
# savefig('proportion.png')