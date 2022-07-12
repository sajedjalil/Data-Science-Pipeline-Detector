import pandas as pd
import numpy as np

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print('Size of training data: ' + str(df_train.shape))
print('Size of testing data:  ' + str(df_test.shape))

print('\nColumns:' + str(df_train.columns.values))

print(df_train.describe())

#print(df_train['place_id'])

print('\nNumber of place ids: ' + str(len(list(set(df_train['place_id'].values.tolist())))))