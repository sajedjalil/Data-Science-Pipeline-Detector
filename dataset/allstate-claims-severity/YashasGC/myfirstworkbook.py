# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn import linear_model
from scipy.stats.stats import pearsonr   
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv', sep=',')
test=pd.read_csv('../input/test.csv',sep=',')


#train = train.sample(frac=0.8, random_state=1)
#valid = train.drop(train.index)

features = train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train[feat] = pd.factorize(train[feat], sort=True)[0]
    #valid[feat] = pd.factorize(valid[feat], sort=True)[0]


y_train=train['loss']
Xtrain=train.drop(['id','loss'], axis=1)
#X_test=valid.drop(['id','loss'], axis=1)
#y_test=valid['loss']


print(Xtrain['cont1'].describe())
print(Xtrain['cont2'].describe())
print(Xtrain['cont3'].describe())