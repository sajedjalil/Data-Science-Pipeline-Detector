import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

train = pd.read_csv('../input/train.csv')
print(len(train))
pos = train[train['QuoteConversion_Flag']==1]
print(len(pos))
print(train.columns)