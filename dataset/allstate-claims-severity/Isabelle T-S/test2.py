# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Importing necessary libraries
import numpy as np

import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import xgboost as xgb

# Reading and Formatting data
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
print ((trainData.shape, testData.shape))
y = trainData['loss']
trainData.drop('loss', axis =1, inplace = True)
print (trainData.shape)
# Log makes the distribution more gaussian. From the discussion forums, shift of 200
# seems to be giving the best results
y = np.log(y.add(200)) 

trainData = trainData.append(testData)
print (trainData.shape)
trainData.head()
trainData.drop('id', axis=1, inplace = True)

from sklearn.preprocessing import LabelEncoder
labCatEncode = LabelEncoder()
trainData.ix[:,0:116] = trainData.ix[:,0:116].apply(labCatEncode.fit_transform)
train = trainData.iloc[:188318]
test = trainData.iloc[188318:]

# Params for xgboost : Shamelessly copied from the user Tilii's kernel
params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0.5290
params['min_child_weight'] = 4.2922
params['colsample_bytree'] = 0.3085
params['subsample'] = 0.9930
params['max_depth'] = 7
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1001

from sklearn.cross_validation import train_test_split
Xtrain,Xval,ytrain,yval = train_test_split(train.values, y.values, test_size  = 0.3)
dTrain = xgb.DMatrix(Xtrain, label=ytrain)
dVal = xgb.DMatrix(Xval, label=yval)
dTest = xgb.DMatrix(test.values)
watchlist = [(dTrain, 'train'), (dVal, 'eval')]
clf = xgb.train(params,dTrain,1000,watchlist,early_stopping_rounds=300)

Pred = pd.DataFrame()
Pred['id'] = testData['id']
Pred['loss'] = np.exp(clf.predict(dTest))
Pred['loss']  = Pred['loss'].add(-200)
Pred.to_csv('XGB_Starter.csv', index=False)
