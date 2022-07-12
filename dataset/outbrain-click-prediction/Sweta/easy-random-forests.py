import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

testing=False
filename = 'TREE_n10_split4_leaf2_TEST'
print ("Get tables to combine")
content = pd.read_csv('../input/promoted_content.csv')
print('Done combining')
print(content.head())
print(content.shape)

chunksize=50000

train = pd.read_csv('../input/clicks_train.csv',iterator=True,chunksize=chunksize) #Load data


for chunk in train:
  chunk=pd.merge(chunk,content,how='left',on='ad_id')
  
print(chunk.shape)