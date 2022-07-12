import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB

import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print (train.shape,test.shape)

test=test.append(train[train.ID>200000])
train=train[~(train.ID>200000)]

del test['target']
print (train.shape,test.shape)

train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
test[['ID']].to_csv('sample_submission.csv',index=False)


def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, test):

    todrop = ['v22', 'v112', 'v125', 'v74', 'v1', 'v110', 'v47']
    print(todrop)

    train.drop(todrop,
               axis=1, inplace=True)
    test.drop(todrop,
              axis=1, inplace=True)

    features = train.columns[2:]
    for col in features:
        if((train[col].dtype == 'object')):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)

    features = train.columns[2:]
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    return train, test


def Mother(train, test):
    features = train.columns[2:]
    num_rounds = 100
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.01
    params["min_child_weight"] = 3
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["max_depth"] = 10
    params["eval_metric"] = "logloss"

    trainpredictions = pd.DataFrame({'ID': train.ID.values,
                                     'target': train.target.values})
    testpredictions = pd.DataFrame({'ID': test.ID.values})

    dvisibletrain = xgb.DMatrix((train[features].values),
                                train.target.values,
                                silent=True)

    dblindtrain = xgb.DMatrix((train[features].values),
                              train.target.values,
                              silent=True)
    dblindtest = xgb.DMatrix((test[features].values),
                             silent=True)

    watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
    gbm = xgb.train(params, dvisibletrain, num_rounds,
                    evals=watchlist, early_stopping_rounds=50,
                    verbose_eval=True)

    predictions1 = gbm.predict(dblindtrain)
    score = log_loss(train.target.values,
                     predictions1)

    trainpredictions['PredictedProb'] = predictions1
    predictions2 = gbm.predict(dblindtest)
    testpredictions['PredictedProb'] = predictions2
    return score, trainpredictions, testpredictions


if __name__ == "__main__":
    print('Start')
    print('Importing Data')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(train[train.target == 0].shape[0])
    print(train[train.target == 1].shape[0])
    print('Munge Data')
    train, test = MungeData(train, test)
    print('Start Train')
    score, secondtrain, secondtest = Mother(train, test)
    print('Start Output')
    secondtrain.to_csv('2ndnbxgbtrain.csv', index=False)
    secondtest.to_csv('2ndnbxgbtest.csv', index=False)
    print('Primary Score', score)
    print('Finish')