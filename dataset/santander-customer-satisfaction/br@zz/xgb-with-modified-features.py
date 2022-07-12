# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Loading training and testing data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# replace all -999999 values with most common value(2)
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape

#Add features that count number of zeros in a row
X = train.iloc[:,:-1]
y = train.TARGET
X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']

#train = train.drop(['TARGET'], axis=1)
# Removing constant columns 
columnsToRemove = []
for col in train.columns:
    if train[col].std() == 0:
        columnsToRemove.append(col)


train.drop(columnsToRemove, axis=1, inplace=True)

# Remove duplicate Columns 

columnsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    for j in range(i+1, len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            columnsToRemove.append(columns[j])

train.drop(columnsToRemove, axis=1, inplace=True)

# including analysis on var38 by cast42 from the below link.
#https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0

def var36_99(var):
    if var == 99:
        return 1
    else:
        return 0
def var36_0123(var):
    if var != 99:
        return 1
    else:
        return 0
train['var36_99'] = train.var36.apply(var36_99)
train['var36_0123'] = train.var36.apply(var36_0123)


features = ['var15', 'ind_var5', 'ind_var30', 'num_var5', 'num_var30', 'num_var42', 
            'num_meses_var5_ult3', 'n0', 'logvar38', 'var38mc',
            'var36_99','var36_0123']
inputData = train[features]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(inputData, y, random_state=1301, stratify=y, test_size=0.3)

dtrain = xgb.DMatrix(X_train, label=y_train, missing=9999999999)
dtest = xgb.DMatrix(X_test, label=y_test, missing=9999999999)

param = {'bst:max_depth':3, 'bst:eta':0.02, 'silent':1, 'objective':'binary:logistic','bst:subSample':0.65 }
param['nthread'] = 4
param['eval_metric'] = 'auc'

num_round = 200

evallist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train( param, dtrain, num_round, evallist )
