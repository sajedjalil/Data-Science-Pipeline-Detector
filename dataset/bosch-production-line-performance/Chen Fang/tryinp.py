# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import xgboost as xgb # XGBoost is short for “Extreme Gradient Boosting”
# See here for details about xgboost: http://xgboost.readthedocs.io/en/latest/model.html
import matplotlib.pyplot as plt

import seaborn as sns
import time
import gc

STATIONS = ['S32', 'S33', 'S34']
train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)
# count missing value in each date column
date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
# create a new variable station which reads SXX in the date column name, like "L3_S37_D3949"
date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])

# filter only S32, S33, and S34
date_cols = date_cols[date_cols['station'].isin(STATIONS)]
# save the date names for S32, 33,34 to a list
date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
print(date_cols)

# Any results you write to the current directory are saved as output.

