# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# import the datasets
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

# check market data in various aspects
market_train_df.info()
market, news = market_train_df.copy(), news_train_df.copy()

market.shape

# check and fill missing values
market.isna().sum()

def fill_na(data):
    for i in data.columns:
        if data[i].dtype == "float64":
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data
    
market = fill_na(market)

market.isna().sum()