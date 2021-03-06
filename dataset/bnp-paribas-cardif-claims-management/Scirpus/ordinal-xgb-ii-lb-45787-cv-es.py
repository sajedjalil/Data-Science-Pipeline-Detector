import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from scipy.sparse import csr_matrix


def PrepareData(train, test):
    trainids = train.ID.values
    testids = test.ID.values
    targets = train['target'].values
    features = train.columns[2:]
    tokeep = ['v4', 'v6', 'v9', 'v10', 'v12', 'v14', 'v16',
              'v18', 'v19', 'v21', 'v34', 'v36', 'v38', 'v39',
              'v40', 'v45', 'v50', 'v53', 'v57', 'v58',
              'v62', 'v68', 'v69', 'v72', 'v80', 'v81',
              'v82', 'v85', 'v88', 'v90',
              'v93', 'v98', 'v99', 'v100', 'v114', 'v115',
              'v119', 'v120', 'v123', 'v124', 'v127', 'v129', 'v22',
              'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66',
              'v71', 'v75', 'v79', 'v112', 'v113', 'v125']

    features = train.columns[2:]
    todrop = list(set(features).difference(set(tokeep)))
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
            total = pd.concat([train[col], test[col]], axis=0).reset_index()
            totalcounts = total[col].value_counts().reset_index()
            totalcounts.rename(columns={'index': col,
                                        col: col+'_count'}, inplace=True)
            train = train.merge(totalcounts, how='left', on=col)
            test = test.merge(totalcounts, how='left', on=col)
            traincounts = train[col].value_counts().reset_index()
            traincounts.rename(columns={'index': col,
                                        col: col+'_count'}, inplace=True)
            traincounts = traincounts[traincounts[col+'_count'] >= 50]
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

    features = train.columns[2:]
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
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
    print('Change num_rounds to 1500')
    num_rounds = 1#  1500
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.01
    params["min_child_weight"] = 1
    params["subsample"] = 0.94
    params["colsample_bytree"] = 0.45
    params["silent"] = 1
    params["max_depth"] = 11
    params["eval_metric"] = "logloss"
    train_preds = np.zeros(len(train), dtype=np.float32)
    test_preds = None
    #test_predsa = None
    splits = 10

    dfulltest = xgb.DMatrix(csr_matrix(gptest[features]), silent=True)

    for X_train_index, X_test_index in \
        StratifiedKFold(gptrain.TARGET.values,
                        n_folds=splits,
                        shuffle=False,
                        random_state=42):

        visibletrain = gptrain.iloc[X_train_index].copy()
        blindtrain = gptrain.iloc[X_test_index].copy()
        dvisibletrain = xgb.DMatrix(csr_matrix(visibletrain[features]),
                                    visibletrain.TARGET.values,
                                    silent=True)
        dvisiblevalid = xgb.DMatrix(csr_matrix(blindtrain[features]),
                                    blindtrain.TARGET.values,
                                    silent=True)
        watchlist = [(dvisibletrain, 'train'), (dvisiblevalid, 'eval')]
        clf = xgb.train(params, dvisibletrain, num_rounds,
                        evals=watchlist, early_stopping_rounds=100,
                        verbose_eval=False)

        predictions = clf.predict(dvisiblevalid, ntree_limit = clf.best_ntree_limit)

        print(clf.best_ntree_limit, log_loss(blindtrain.TARGET.values,
                       predictions))

        train_preds[X_test_index] = predictions
        del predictions

        gc.collect()

        if(test_preds is None):
            test_preds = clf.predict(dfulltest, ntree_limit = clf.best_ntree_limit)
    #        test_predsa = clf.predict(dfulltest, ntree_limit = clf.best_ntree_limit)
        else:
            test_preds *= clf.predict(dfulltest, ntree_limit = clf.best_ntree_limit)
    #        test_predsa += clf.predict(dfulltest, ntree_limit = clf.best_ntree_limit)

        del clf
        gc.collect()

    #train_preds = np.power(train_preds, 1./splits)
    test_preds = np.power(test_preds, 1./splits)
    print('Average Log Loss:', log_loss(gptrain.TARGET.values, train_preds))
    submission = pd.DataFrame({"ID": gptrain.ID,
                               "TARGET": gptrain.TARGET,
                               "PREDICTION": train_preds})

    submission.to_csv("simplexgbtrain.csv", index=False)
    submission = pd.DataFrame({"ID": gptest.ID, "PredictedProb": test_preds})
    submission.to_csv("simplexgbtest.csv", index=False)
    print('Finished!')