# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# import package
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics 
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
chunksize = 100000

# define Idcol and Response Column
IDcol = "Id"
target = "Response"
# Sample the data to decide some important features
start_time = datetime.datetime.now()
date_chunks = pd.read_csv("../input/train_date.csv", index_col=0, chunksize=chunksize, dtype=np.float32)
num_chunks = pd.read_csv("../input/train_numeric.csv", index_col=0, chunksize=chunksize, dtype=np.float32)
train_value = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.001)
               for dchunk, nchunk in zip(date_chunks, num_chunks)])
end_time = datetime.datetime.now()
print (end_time - start_time)

# Any results you write to the current directory are saved as output.