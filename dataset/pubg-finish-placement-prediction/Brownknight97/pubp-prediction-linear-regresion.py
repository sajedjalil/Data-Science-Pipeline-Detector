import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt # for plotting
from math import ceil, floor


df_train = pd.read_csv('../input/train_V2.csv')
df_test = pd.read_csv('../input/test_V2.csv')

df_train.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

test_ids = df_test['Id'].tolist()
df_test.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

df_train = pd.get_dummies(df_train,columns=['matchType'], drop_first=True)
df_test = pd.get_dummies(df_test,columns=['matchType'], drop_first=True)

columns_with_nan = df_train.columns[df_train.isna().any()].tolist()
for column in columns_with_nan:
    mean_value = df_train[column].mean()
    df_train[column] = df_train[column].fillna(mean_value)

columns_with_nan = df_test.columns[df_test.isna().any()].tolist()
for column in columns_with_nan:
    mean_value = df_test[column].mean()
    df_test[column] = df_test[column].fillna(mean_value)

df_train_dummy = df_train.pop('winPlacePerc')
x = df_train.loc[:,:].values
df_train['winPlacePerc'] = df_train_dummy
y = df_train.loc[:,['winPlacePerc']].values

x_test = df_test.loc[:,:].values

df_train_head = df_train.head(10)
df_test_head = df_test.head(10)

reg = LinearRegression().fit(x, y)
yPrediction = reg.predict(x_test)

index = 0
for i in yPrediction:
    if i[0] < 0:
        yPrediction[index][0] = 0
    index = index + 1

results = []
index = 0
for i in yPrediction:
    row = []
    row.append(test_ids[index])
    row.append(yPrediction[index][0])
    results.append(row)
    index = index + 1

def lastPlaceConversion(place):
    if place < 0.01:
        return 0
    elif place > 1:
        return 1
    else:
        return place

df = pd.DataFrame(results, columns=['Id', 'winPlacePerc'])
df['winPlacePerc'] = df['winPlacePerc'].apply(lastPlaceConversion)
df.reset_index(drop=True)

df.to_csv('submission.csv', index=False)
