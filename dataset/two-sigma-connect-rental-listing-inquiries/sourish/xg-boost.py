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
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def runxgb(train_x,train_y,test_x,seed_val,num_round):
    param={}
    param['objective']='multi:softprob'
    param['eval_metric']='mlogloss'
    param['eta'] = 0.04
    param['max_depth'] = 6
    param['subsample'] = 0.7
    pamam['colsample_bytree']=0.7
    param['seed'] = seed_val
    param['silent']=1
    param['num_class ']=3
    param['booster'] = "gbtree"
    num_round=num_round
    plist = list(param.items())
    xgtrain <- xgb.DMatrix(train_x, train_y, missing = NA)
    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

data_path = "../input/"
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
print(train.shape)
print(test.shape)
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
train['photo_no'] = train['photos'].apply(len)
test['photo_no'] = test['photos'].apply(len)
train["num_features"] = train["features"].apply(len)
test["num_features"] = test["features"].apply(len)
