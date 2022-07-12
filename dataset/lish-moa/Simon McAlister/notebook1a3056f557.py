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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import multiprocessing
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss


files = ['../input/lish-moa/test_features.csv', 
         '../input/lish-moa/train_targets_scored.csv',
         '../input/lish-moa/train_features.csv',
         '../input/lish-moa/train_targets_nonscored.csv',
         '../input/lish-moa/sample_submission.csv']

with multiprocessing.Pool() as pool:
    test, train_target, train, train_nonscored, sub = pool.map(pd.read_csv, files)

for column_name in train_target.columns[1:]:
    m=1
    print(column_name,m)
    sub[column_name]=m

sub.to_csv('submission.csv', index=False)