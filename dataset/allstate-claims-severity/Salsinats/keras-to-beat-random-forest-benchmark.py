# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# keras imports
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df = pd.concat([df_train, df_test])

categoricals = [c for c in df.columns if c[:3] == "cat"]
continuous = [c for c in df.columns if c[:4] == "cont"]

# encode categorical features
Le = dict()
for c in categoricals:
    le = LabelEncoder()
    le.fit(df[c].values)
    Le[c] = le

    df_train[c] = le.transform(df_train[c].values)
    df_test[c] = le.transform(df_test[c].values)

# select features
X_train = df_train[continuous + categoricals].values
y_train = df_train["loss"].values

# split into train and val sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.5)

# define keras model
model = Sequential()
model.add(Dense(100, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1))

model.compile(loss="mae",
              optimizer="adam")

# parameters
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               mode="min")

kwargs = {
    "nb_epoch": 70,
    "batch_size": 1024,
    "validation_data": [X_val, y_val],
    "verbose": 1,
    "callbacks": [early_stopping]
}

# training
model.fit(X_train, y_train, **kwargs)

# prediction
X_test = df_test[continuous + categoricals].values
y_pred = model.predict(X_test)

# save prediction
now = datetime.datetime.now()
base = 'submission_'
sub_file = base + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print('Writing submission: ', sub_file)

df_pred = pd.DataFrame(columns=["id", "loss"])
df_pred["id"] = df_test.id.values
df_pred["loss"] = y_pred
df_pred.to_csv(sub_file, index=False)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.