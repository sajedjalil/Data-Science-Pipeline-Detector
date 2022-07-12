# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as mplt
import seaborn as sns
from sklearn.linear_model import LinearRegression


train = pd.read_csv("../input/train.csv", parse_dates=["timestamp"], date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))
test = pd.read_csv("../input/test.csv", parse_dates=["timestamp"], date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))

train_model = train[['price_doc','full_sq','life_sq','floor','max_floor','build_year','state']]


print(train_model.head(),train_model.describe())