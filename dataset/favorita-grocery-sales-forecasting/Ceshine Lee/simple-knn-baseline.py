"""A simple pattern-matching approach.

Warning: It is only marginally better than the mean baseline. 
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4],
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 117477878)  # 2017-06-01
)

df_train = df_train.set_index(
    ["store_nbr", "item_nbr", "date"]).unstack(
        level=-1).fillna(0)
df_train.columns = df_train.columns.get_level_values(1)

model = KNeighborsRegressor(
    n_neighbors=100, n_jobs=-1,
    leaf_size=100,  # weights='distance'
)

X_train = np.concatenate([
    df_train[
        pd.date_range(date(2017, 6, 7) + timedelta(days=i * 7), periods=14)
    ].values for i in range(6)
], axis=0)
y_train = np.concatenate([
    df_train[
        pd.date_range(date(2017, 6, 21) + timedelta(days=i * 7), periods=16)
    ].values for i in range(6)
], axis=0)
X_test = df_train[pd.date_range("2017-08-01", periods=14)].values

print("Fitting...")
model.fit(X_train, y_train)
print("Predicting...")
y_pred = model.predict(X_test)
df_preds = pd.DataFrame(
    y_pred, index=df_train.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3],
    parse_dates=["date"] 
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

submission = df_test.join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.expm1(submission["unit_sales"])
submission.to_csv(
    'knn.csv', float_format='%.2f', index=None)
