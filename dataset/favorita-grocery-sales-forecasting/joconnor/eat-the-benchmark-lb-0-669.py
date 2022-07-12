# Forked from Paulo Pinto: https://www.kaggle.com/paulorzp/one-line-median-lb-0-777?scriptVersionId=1636973
import pandas as pd
import numpy as np
from datetime import datetime
groupby_cols = ["item_nbr","store_nbr"]
test = pd.read_csv("../input/test.csv", usecols=["id"] + groupby_cols)
train = pd.read_csv("../input/train.csv", usecols=["date", "unit_sales"] + groupby_cols)
train["date"] = pd.to_datetime(train["date"])
train = train[train["date"] > datetime(2017, 8, 1)]

# since the evaluation metric is proportional to (log(pred + 1) - log(y + 1))**2, it makes sense log transform before taking the mean.
train["unit_sales"] = np.log(train["unit_sales"].clip(lower=0) + 1)
preds = np.exp(train.groupby(groupby_cols)["unit_sales"].mean()) - 1
test = test.set_index(groupby_cols).join(preds.to_frame("unit_sales"), how="left").fillna(0)
test.to_csv("log_mean.csv", float_format="%.2f", index=None)
