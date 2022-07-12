import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


def PrepareData(train, test):
    trainids = train.ID.values
    testids = test.ID.values
    targets = train['target'].values
    tokeep = ['v3', 'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31',
              'v34', 'v38', 'v40',
              'v50',
              'v52', 'v56', 'v62', 'v66',
              'v71', 'v72', 'v74', 'v75', 'v79',
              'v91',  # 'v107',
              'v47',  # 'v110',
              'v112', 'v113', 'v114', 'v125', 'v129']
    features = train.columns[2:]
    todrop = list(set(features).difference(tokeep))
    train.drop(todrop, inplace=True, axis=1)
    test.drop(todrop, inplace=True, axis=1)
    print(train.columns)
    features = train.columns[2:]
    for col in features:
        print(col)
        if((train[col].dtype == 'object')):
            train.loc[~train[col].isin(test[col]), col] = 'Orphans'
            test.loc[~test[col].isin(train[col]), col] = 'Orphans'
            train[col].fillna('Missing', inplace=True)
            test[col].fillna('Missing', inplace=True)
            train[col], tmp_indexer = pd.factorize(train[col])
            test[col] = tmp_indexer.get_indexer(test[col])
            traincounts = train[col].value_counts().reset_index()
            traincounts.rename(columns={'index': col,
                                        col: col+'_count'}, inplace=True)
            traincounts = traincounts[traincounts[col+'_count'] >= 50]
            # train = train.merge(traincounts, how='left', on=col)
            # test = test.merge(traincounts, how='left', on=col)
            g = train[[col, 'target']].copy().groupby(col).mean().reset_index()
            g = g[g[col].isin(traincounts[col])]
            g.rename(columns={'target': col+'_avg'}, inplace=True)
            train = train.merge(g, how='left', on=col)
            test = test.merge(g, how='left', on=col)
            h = train[[col, 'target']].copy().groupby(col).std().reset_index()
            h = h[h[col].isin(traincounts[col])]
            h.rename(columns={'target': col+'_std'}, inplace=True)
            train = train.merge(h, how='left', on=col)
            test = test.merge(h, how='left', on=col)
            train.drop(col, inplace=True, axis=1)
            test.drop(col, inplace=True, axis=1)

    features = train.columns[2:]
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
    ss = StandardScaler()
    train[features] = np.round(ss.fit_transform(train[features].values), 6)
    test[features] = np.round(ss.transform(test[features].values), 6)
    gptrain = pd.DataFrame()
    gptest = pd.DataFrame()
    gptrain.insert(0, 'ID', trainids)
    gptest.insert(0, 'ID', testids)
    gptrain = pd.merge(gptrain, train[list(['ID'])+list(features)], on='ID')
    gptest = pd.merge(gptest, test[list(['ID'])+list(features)], on='ID')
    gptrain['TARGET'] = targets
    del train
    del test
    gc.collect()
    return gptrain, gptest


if __name__ == "__main__":
    print('Started!')
    ss = StandardScaler()
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    gptrain, gptest = PrepareData(train, test)
    features = gptrain.columns[1:-1]
    print(features)
    dfulltrain = \
        xgb.DMatrix(gptrain[features],
                    gptrain.TARGET.values,
                    silent=True)
    dfulltest = \
        xgb.DMatrix(gptest[features],
                    silent=True)

    print('Change num_rounds to 1000 to get .46190')
    num_rounds = 1
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.01
    params["min_child_weight"] = 1
    params["subsample"] = 0.96
    params["colsample_bytree"] = 0.45
    params["silent"] = 1
    params["max_depth"] = 13
    params["eval_metric"] = "logloss"
    clf = xgb.train(params, dfulltrain, num_rounds)

    train_preds = clf.predict(dfulltrain)
    print('Log Loss:', log_loss(gptrain.TARGET.values,
                                train_preds))
    submission = pd.DataFrame({"ID": gptrain.ID,
                               "TARGET": gptrain.TARGET,
                               "PREDICTION": train_preds})
    submission.to_csv("ordinalxgbtrain.csv", index=False)
    test_preds = clf.predict(dfulltest)
    submission = pd.DataFrame({"ID": gptest.ID,
                               "PredictedProb": test_preds})
    submission.to_csv("ordinalxgbtest.csv", index=False)
    print('Finished!')