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

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
import xgboost as xgb
import random

random.seed( 10 )

# get the categorical columns
cat_cols = ['v3','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75','v79','v107','v110','v112','v113','v125']

def impute_most_freq_value(df,colname):
    c = df[colname].value_counts()
    return c.index[0]

def load_data():
    train = pd.read_csv('../input/train.csv')
    test  = pd.read_csv('../input/test.csv')

    train.drop(['v22'], axis=1, inplace=True)
    train.drop(['v91'], axis=1, inplace=True)
    test.drop(['v22'], axis=1, inplace=True)
    test.drop(['v91'], axis=1, inplace=True)

    nas = {}
    for colname in cat_cols:
        nas[colname] = impute_most_freq_value(train,colname)

    for colname in cat_cols:
        train[colname].fillna(nas[colname],inplace=True)

    for colname in cat_cols:
        test[colname].fillna(nas[colname],inplace=True)

    cat_train = train[cat_cols]
    cat_test = test[cat_cols]

    #put the numerical as matrix
    train.drop(cat_cols, axis=1, inplace=True)
    test.drop(cat_cols, axis=1, inplace=True)

    print(cat_train.describe())

    # transform the categorical to dict
    dict_train_data = cat_train.T.to_dict().values()
    dict_test_data = cat_test.T.to_dict().values()

    #vectorize
    vectorizer = DV(sparse = False)
    features = vectorizer.fit_transform(dict_train_data)
    vec_data = pd.DataFrame(features)
    vec_data.columns = vectorizer.get_feature_names()
    #vec_data.rename(columns={'changed': 'vec_changed'}, inplace=True)
    #vec_data.rename(columns={'id': 'vec_id'}, inplace=True)
    vec_data.index = train.index
    train = train.join(vec_data)

    features = vectorizer.transform(dict_test_data)
    vec_data = pd.DataFrame(features)
    vec_data.columns = vectorizer.get_feature_names()
    vec_data.index = test.index
    test = test.join(vec_data)

    #merge numerical and categorical sets
    trainend = int(0.75*len(train))
    valid_inds = list(train[trainend:].index.values)
    train_inds = list(train.loc[~train.index.isin(valid_inds)].index.values)

    m = train.mean()
    train.fillna(m,inplace=True)
    test.fillna(m,inplace=True)

    return train, test, train_inds, valid_inds

ROUNDS=430

params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.035
#params["min_child_weight"] = 3
params["subsample"] = 0.67
params["colsample_bytree"] = 0.9
params["silent"] = 1
params["max_depth"] = 6
params["eval_metric"] = "logloss"
params["n_estimators"] = 100

train, test, train_inds, valid_inds = load_data()

valid = train.iloc[valid_inds]
train = train.iloc[train_inds]

trainlabels = train['target'].values
validlabels = valid['target'].values

X_train   = train.drop(['ID','target'],axis=1).values
X_valid   = valid.drop(['ID','target'],axis=1).values

ids       = test['ID'].values
X_test    = test.drop(['ID'],axis=1).values

sigma     = X_train.std()
X_train  /= sigma
X_valid  /= sigma
X_test   /= sigma

xgtrain   = xgb.DMatrix(X_train, label=trainlabels)
xgval     = xgb.DMatrix(X_valid, label=validlabels)
xgtest    = xgb.DMatrix(X_test)

plst      = list(params.items())
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model     = xgb.train(plst, xgtrain, ROUNDS, watchlist, early_stopping_rounds=100 )

y_pred    = model.predict(xgtest)

pd.DataFrame({"ID": ids, "PredictedProb": y_pred}).to_csv('submission_quick_xgb.csv',index=False)


