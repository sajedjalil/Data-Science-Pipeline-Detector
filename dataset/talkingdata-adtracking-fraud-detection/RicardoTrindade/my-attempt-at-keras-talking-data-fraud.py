# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', skiprows=129903891, nrows=55000000,)
df.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
df.head()
ip_count = df.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
df = pd.merge(df, ip_count, on='ip', how='left', sort=False)
df['clicks_by_ip'] = df['clicks_by_ip'].astype('uint16')
df.drop('ip', axis=1, inplace=True)


y = df['is_attributed']
X = df.drop(['is_attributed', 'attributed_time','click_time'], axis=1)

del df

model = Sequential()
model.add(Dense(125, activation='relu', input_shape=(5,)))
model.add(Dense(75, activation='relu', input_shape=(5,)))
model.add(Dense(50, activation='relu', input_shape=(5,)))
model.add(Dense(25, activation='relu', input_shape=(5,)))
model.add(Dense(10, activation='relu', input_shape=(5,)))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
X = np.asarray(X)
y = np.asarray(y)

y = to_categorical(y, 2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
history = model.fit(X_scaled, y,
                    verbose=1,
                    batch_size=100000)


df_test = pd.read_csv('../input/test.csv')
# Count the number of clicks by ip and app
ip_count = df_test.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
df_test = pd.merge(df_test, ip_count, on='ip', how='left', sort=False)
df_test['clicks_by_ip'] = df_test['clicks_by_ip'].astype('uint16')
df_test.drop('ip', axis=1, inplace=True)

X_test = df_test.drop(['click_time', 'click_id'], axis=1)

X_test = np.asarray(X_test)
X_test_scaled = scaler.fit_transform(X_test)
predict = model.predict_proba(X_test_scaled)

df_submit = pd.DataFrame()
df_submit['click_id'] = df_test['click_id']
df_submit['is_attributed'] = predict[:,0]

df_submit[['click_id', 'is_attributed']].to_csv(
    'submission.csv.gz', index=False, compression='gzip')