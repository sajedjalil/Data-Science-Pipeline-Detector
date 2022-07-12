# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import xgboost as xgb
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def get_timespan(df, dt, minus,period):
    print ("date range is ", (dt - timedelta(days=minus)))
    return df[pd.date_range(dt - timedelta(days=minus), periods = period)]
    

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 117477878)
)

df_pivot= df_train.pivot_table(index=['store_nbr','item_nbr'], columns='date', values='unit_sales', fill_value=0)

df_pivot_promo= df_train.pivot_table(index=['store_nbr','item_nbr'], 
                                     columns='date', values='onpromotion', fill_value=False)
                                     
df_new_train = df_pivot_promo.reset_index()

df_new_train = pd.concat([df_pivot, df_pivot_promo], axis=1)


promo_2017_test = pd.read_csv("../input/test.csv",usecols=[0,1,2,3,4],
                              dtype={'onpromotion':bool}, parse_dates=["date"])

promo_2017_test_table = promo_2017_test.pivot_table(index=['store_nbr', 'item_nbr'], columns='date', values='onpromotion', fill_value=False)

promo_2017_test_table = promo_2017_test_table.reindex(df_pivot_promo.index).fillna(False)

df_promo_new = pd.concat([df_pivot_promo, promo_2017_test_table], axis=1)



def prepare_dataset(dt, is_train=True):
    X= pd.DataFrame({
        "mean_3_2017":get_timespan(df_pivot, dt, 3,3).mean(axis=1).values,
        "mean_7_2017":get_timespan(df_pivot, dt, 7,7).mean(axis=1).values,
        "mean_14_2017":get_timespan(df_pivot, dt, 14,14).mean(axis=1).values,
        "promo_14_2017":get_timespan(df_promo_new, dt, 14,14).sum(axis=1).values
    })
    for i in range(16):
        X["promo_{}".format(i)] = df_promo_new[dt +timedelta(days=1)].values.astype(np.uint8)
    if is_train:
        y = df_pivot[pd.date_range(dt, periods = 16)].values
        #y = pd.DataFrame(y)
        #y = pd.DataFrame(y, columns=['y'])
        print(type(y))
        return X,y
    return X
    
print("Preparing the dataset")
t2017 = date(2017,6,21)
x_1, y_1 = [], []
for i in range(4):
    delta = timedelta(days=7 *i)
    print("delta is ", delta)
    X_tmp, y_tmp = prepare_dataset(t2017 + delta)
    x_1.append(X_tmp)
    y_1.append(y_tmp)
X_train = pd.concat(x_1, axis=0)
y_train = np.concatenate(y_1, axis=0)

X_val, y_val = prepare_dataset(date(2017,7,26))
X_test = prepare_dataset(date(2017,8,16), is_train=False)
features = X_train.columns[:]
print (features)

MAX_ROUNDS = 1000
val_pred = []
test_pred = []
#cate_vars = []
params = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": 0.1,
          "max_depth": 9,
          "subsample": 1.0,
          "colsample_bytree": 0.7,
          "silent": 1,
          "min_child_weight":2,
          }
X_test_new = xgb.DMatrix(X_test)


for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = xgb.DMatrix(
        X_train, label=y_train[:, i]
    )
    dval = xgb.DMatrix(
        X_val, label=y_val[:, i])
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        evals=watchlist, early_stopping_rounds=55, verbose_eval=50
    )
    val_pred.append(bst.predict(xgb.DMatrix(X_val)))
    test_pred.append(bst.predict(X_test_new))
    
print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

print("Making submission...")
y_test = np.array(test_pred).transpose()

df_preds = pd.DataFrame(
    y_test, index=df_pivot.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

df_test = promo_2017_test.set_index(['store_nbr', 'item_nbr', 'date'])

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('xgboost.csv', float_format='%.4f', index=None)