# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
#matplotlib inline
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
id_test = test.id

mult = .971

y_train = train["price_doc"] * mult
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)


# MANAGING CATEGORICAL FEATURES

df_tot = pd.concat([x_train,x_test])
for c in df_tot.columns:
    if df_tot[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_tot[c].values)) 
        df_tot[c] = lbl.transform(list(df_tot[c].values))
x_train = df_tot.head(len(x_train))
x_test = df_tot.tail(len(x_test))


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 500  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

output.to_csv("sub.csv", index=False)