import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import operator

# Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
print(df_train.head())