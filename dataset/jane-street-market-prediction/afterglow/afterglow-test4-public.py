# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import numpy as np
import matplotlib.pyplot as plt

# import cudf

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import pandas as pd
pd.set_option('display.max_columns', 500)

import xgboost as xgb
print("XGBoost version:", xgb.__version__)

import warnings
warnings.filterwarnings("ignore")

# create the environment
import janestreet
print('Creating competition environment...', end='')
env = janestreet.make_env()
print('Finished.')

train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
features = [col for col in list(train.columns) if 'feature' in col]

print('Creating submissions file...', end='')
rcount = 0
for (test_df, prediction_df) in env.iter_test():
    X_test = test_df.loc[:, features]
    # y_preds = clf.predict(X_test)
    # prediction_df.action = y_preds
    prediction_df.action = 1
    env.predict(prediction_df)
    rcount += len(test_df.index)
print(f'Finished processing {rcount} rows.')