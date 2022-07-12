# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyodbc
import scipy
import seaborn as sns
import statsmodels
import matplotlib
from matplotlib import pyplot as plt
import scipy
import csv
import os
import lightgbm as lgm
import xgboost as xgb
import math
from sklearn import preprocessing
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
building_metadata_df = pd.read_csv(r"/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
train_df = pd.read_csv(r"/kaggle/input/ashrae-energy-prediction/train.csv")
weather_train_df = pd.read_csv(r"/kaggle/input/ashrae-energy-prediction/weather_train.csv")
test_df = pd.read_csv(r'/kaggle/input/ashrae-energy-prediction/test.csv')
weather_test_df = pd.read_csv(r'/kaggle/input/ashrae-energy-prediction/weather_test.csv')


building_metadata_df.to_feather('building_metadata.feather')
train_df.to_feather('train.feather')
weather_train_df.to_feather('weather_train.feather')
test_df.to_feather('test.feather')
weather_test_df.to_feather('weather_test.feather')








