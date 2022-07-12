import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter


def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName)['Response'].mean().reset_index()
    grpCount = data1.groupby(columnName)['Response'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.Response
    if(useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 1]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['Response'].values
    x = pd.merge(data2[[columnName, 'Response']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['Response']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
        #  x = x + np.random.normal(0, .01, x.shape[0])
    return x.fillna(x.mean())


def GrabData():
    directory = '../input/'
    trainfiles = ['train_categorical.csv',
                  'train_date.csv',
                  'train_numeric.csv']
    testfiles = ['test_categorical.csv',
                 'test_date.csv',
                 'test_numeric.csv']

    cols = [['Id'],
            ['Id',
             'L3_S30_D3496', 'L3_S30_D3506',
             'L3_S30_D3501', 'L3_S30_D3516',
             'L3_S30_D3511'],
            ['Id',
             'L1_S24_F1846', 'L3_S32_F3850',
             'L1_S24_F1695', 'L1_S24_F1632',
             'L3_S33_F3855', 'L1_S24_F1604',
             'L3_S29_F3407', 'L3_S33_F3865',
             'L3_S38_F3952', 'L1_S24_F1723',
             'Response']]
    traindata = None
    testdata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()
    del cols[2][-1]  # Test doesn't have response!
    for i, f in enumerate(testfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()
    testdata['Response'] = 0  # Add Dummy Value
    visibletraindata = traindata[::2]
    blindtraindata = traindata[1::2]
    print(blindtraindata.columns)
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, False).values
            testdata.loc[:, col] = LeaveOneOut(visibletraindata,
                                               testdata, col, False).values
    del visibletraindata
    gc.collect()
    testdata.drop('Response', inplace=True, axis=1)
    return blindtraindata, testdata


def Train():
    train, test = GrabData()
    print('Train:', train.shape)
    print('Test', test.shape)
    features = train.columns[1:-1]
    print(features)
    num_rounds = 50
    params = {}
    params['objective'] = "binary:logistic"
    params['eta'] = 0.021
    params['max_depth'] = 7
    params['colsample_bytree'] = 0.82
    params['min_child_weight'] = 3
    params['base_score'] = 0.005
    params['silent'] = True

    print('Fitting')
    trainpredictions = None
    testpredictions = None

    dvisibletrain = \
        xgb.DMatrix(train[features],
                    train.Response,
                    silent=True)
    dtest = \
        xgb.DMatrix(test[features],
                    silent=True)

    folds = 1
    for i in range(folds):
        print('Fold:', i)
        params['seed'] = i
        watchlist = [(dvisibletrain, 'train'), (dvisibletrain, 'val')]
        clf = xgb.train(params, dvisibletrain,
                        num_boost_round=num_rounds,
                        evals=watchlist,
                        early_stopping_rounds=20,
                        feval=mcc_eval,
                        maximize=True
                        )
        limit = clf.best_iteration+1
        # limit = clf.best_ntree_limit
        predictions = \
            clf.predict(dvisibletrain, ntree_limit=limit)

        best_proba, best_mcc, y_pred = eval_mcc(train.Response,
                                                predictions,
                                                True)
        print('tree limit:', limit)
        print('mcc:', best_mcc)
        print(matthews_corrcoef(train.Response,
                                y_pred))
        if(trainpredictions is None):
            trainpredictions = predictions
        else:
            trainpredictions += predictions
        predictions = clf.predict(dtest, ntree_limit=limit)
        if(testpredictions is None):
            testpredictions = predictions
        else:
            testpredictions += predictions
        imp = get_importance(clf, features)
        print('Importance array: ', imp)

    best_proba, best_mcc, y_pred = eval_mcc(train.Response,
                                            trainpredictions/folds,
                                            True)
    print(matthews_corrcoef(train.Response,
                            y_pred))

    submission = pd.DataFrame({"Id": train.Id,
                               "Prediction": trainpredictions/folds,
                               "Response": train.Response})
    submission[['Id',
                'Prediction',
                'Response']].to_csv('rawtrainxgbsubmission'+str(folds)+'.csv',
                                    index=False)

    submission = pd.DataFrame({"Id": test.Id.values,
                               "Response": testpredictions/folds})
    submission[['Id', 'Response']].to_csv('rawxgbsubmission'+str(folds)+'.csv',
                                          index=False)
    y_pred = (testpredictions/folds > .08).astype(int)
    submission = pd.DataFrame({"Id": test.Id.values,
                               "Response": y_pred})
    submission[['Id', 'Response']].to_csv('xgbsubmission'+str(folds)+'.csv',
                                          index=False)

if __name__ == "__main__":
    print('Started')
    Train()
    print('Finished')
