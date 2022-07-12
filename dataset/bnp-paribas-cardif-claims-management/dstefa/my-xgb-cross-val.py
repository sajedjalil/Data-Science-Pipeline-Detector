import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import scipy as sp
from pandas import DataFrame
import copy
#############################################################
def splitData(train):
    count_0 = 27300 
    count_1 = 87021

    headers = ['ID', 'target', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v46', 'v47', 'v48', 'v49', 'v50', 'v51', 'v52', 'v53', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v64', 'v65', 'v66', 'v67', 'v68', 'v69', 'v70', 
    'v71', 'v72', 'v73', 'v74', 'v75', 'v76', 'v77', 'v78', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v105', 'v106', 'v107', 'v108', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v123', 'v124', 'v125', 'v126', 'v127', 'v128', 'v129', 'v130', 'v131']

    newTrainData = []
    newTest = []
    act = []
    c_0 = 0
    c_1 = 0
    
    for line in train.values:
        if line[1] == 0:
            if c_0 < count_0/2:
                newTrainData.append(line)
                c_0 +=1
            else:
                newTest.append(line)
                act.append(line[1])
        else:
            if c_1 < count_1/2:
                newTrainData.append(line)
                c_1 +=1
            else:
                newTest.append(line)
                act.append(line[1])
    
    dfTrain = DataFrame(newTrainData, columns=headers)
    dfTest = DataFrame(newTest, columns=headers)
    dfTest2 = DataFrame(newTest, columns=headers)
    del dfTest['target']

    return dfTrain, dfTest, act, dfTest2
######################################################################

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
    train, test, act, range_test = splitData(train)
    
    #test = pd.read_csv('../input/test.csv')
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
    num_rounds = 10#######################################
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
    #submission = pd.DataFrame({"ID": gptrain.ID,
    #                           "TARGET": gptrain.TARGET,
    #                           "PREDICTION": train_preds})
    #submission.to_csv("ordinalxgbtrain.csv", index=False)
    test_preds = clf.predict(dfulltest)
    submission = pd.DataFrame({"ID": gptest.ID,
                               "PredictedProb": test_preds})
    submission.to_csv("ordinalxgbtest.csv", index=False)
    print('Finished!')