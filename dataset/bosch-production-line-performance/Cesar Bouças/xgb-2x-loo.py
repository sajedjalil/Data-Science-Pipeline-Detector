import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from operator import itemgetter
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

NROUNDS = 30
NFOLDS = 1
NROWS = 50000
CHUNKSIZE = 10000
DATA_DIR = "../input/"


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
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
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

def resultsToCsv(filename, ids, y):
    df = pd.DataFrame({'Id': ids, 'Response': y})
    count1 = df[df.Response == 1].shape[0]
    print('Response 1: {0} in {1}.csv'.format(count1, filename))
    if count1 > 0:
        df[['Id', 'Response']].to_csv("{0}.csv".format(filename), index=False)

def GrabData():

    trainfiles = [DATA_DIR + 'train_categorical.csv',
                  DATA_DIR + 'train_date.csv',
                  'train_ex.csv',
                  DATA_DIR + 'train_numeric.csv'
    ]
    testfiles = [DATA_DIR + 'test_categorical.csv',
                 DATA_DIR + 'test_date.csv',
                 'test_ex.csv',
                 DATA_DIR + 'test_numeric.csv'
    ]

    cols = [['Id',
             'L1_S24_F1559', 'L3_S32_F3851',
             'L1_S24_F1827', 'L1_S24_F1582',
             'L3_S32_F3854', 'L1_S24_F1510',
             'L3_S36_F3941','L3_S38_F3954','L3_S38_F3955','L3_S38_F3958',
             'L3_S38_F3959','L3_S38_F3962','L3_S38_F3963','L3_S39_F3965',
             'L1_S24_F1525'],
            ['Id',
             'L3_S29_D3474', 
             'L3_S30_D3496', 'L3_S30_D3506',
             'L3_S30_D3501', 'L3_S30_D3516',
             'L3_S30_D3511', 
             'L3_S32_D3852', 
             'L3_S33_D3856', 'L3_S33_D3858',
             'L3_S37_D3942',
             'L3_S37_D3943','L3_S37_D3945',
             'L3_S37_D3947','L3_S37_D3949',
             'L3_S37_D3951','L3_S38_D3953',
             'L3_S38_D3957','L3_S38_D3961'
             ],
             ['Id', 'StartTime', 'FinishTime', 'Duration',
             '0_¯\_(ツ)_/¯_1','0_¯\_(ツ)_/¯_2',
             '0_¯\_(ツ)_/¯_3','0_¯\_(ツ)_/¯_4'],
            ['Id',
             'L0_S4_F109', 'L0_S15_F403', 'L0_S13_F354',
             'L1_S24_F1846', 'L1_S24_F1695', 'L1_S24_F1632', 'L1_S24_F1604',
             'L1_S24_F1723', 'L1_S24_F1844', 'L1_S24_F1842',
             'L2_S26_F3106', 'L2_S26_F3036', 'L2_S26_F3113', 'L2_S26_F3073',
             'L3_S29_F3407', 'L3_S29_F3376', 'L3_S29_F3324', 'L3_S29_F3382', 'L3_S29_F3479',
             'L3_S30_F3704', 'L3_S30_F3774', 'L3_S30_F3554',
             'L3_S32_F3850', 'L3_S32_F3850',
             'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3865',
             'L3_S37_F3944', 'L3_S37_F3946', 'L3_S37_F3948', 'L3_S37_F3950', 
             'L3_S38_F3956', 'L3_S38_F3960', 'L3_S38_F3952', 

             'L3_S30_F3604', 'L3_S30_F3749', 'L0_S0_F20', 'L3_S30_F3559', 'L3_S30_F3819', 'L3_S29_F3321', 'L3_S29_F3373',
             'L3_S30_F3569', 'L3_S30_F3569', 'L3_S30_F3579', 'L3_S30_F3639', 'L3_S29_F3449', 'L3_S36_F3918', 'L3_S30_F3609',
             'L3_S30_F3574', 'L3_S29_F3354', 'L3_S30_F3759', 'L0_S6_F122', 'L3_S30_F3664', 'L3_S30_F3534', 'L0_S1_F24', 'L3_S29_F3342',
             'L0_S7_F138', 'L2_S26_F3121', 'L3_S30_F3744', 'L3_S30_F3799', 'L3_S33_F3859', 'L3_S30_F3784', 'L3_S30_F3769', 'L2_S26_F3040',
             'L3_S30_F3804', 'L0_S5_F114', 'L0_S12_F336', 'L0_S9_F170', 'L3_S29_F3330', 'L3_S29_F3351', 'L3_S29_F3339', 'L3_S29_F3427', 'L3_S30_F3829',
             'L0_S0_F22', 'L3_S30_F3589', 'L3_S30_F3494', 'L3_S29_F3421', 'L3_S29_F3327', 'L0_S5_F116', 'L3_S29_F3318', 'L3_S30_F3524', 'L3_S29_F3379',
             'L3_S29_F3333', 'L3_S29_F3455', 'L3_S29_F3430', 'L3_S30_F3529', 'L0_S0_F0', 'L3_S30_F3754', 'L3_S36_F3920', 'L0_S3_F96', 'L3_S29_F3407', 
             'L3_S29_F3473', 'L3_S29_F3476', 'L3_S30_F3674',
             'Response']]

    n = 0
    for i in range(4):
        for col in cols[i][1:]:
            n = n + 1
    print('Feature count:{0}'.format(n-1))

    traindata = None
    testdata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        nrows = 0
        for i, chunk in enumerate(pd.read_csv(f,
                                              usecols=cols[i],
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            #print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
            
            nrows += CHUNKSIZE
            if nrows >= NROWS:
                break
            
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()        
    del cols[3][-1]  # Test doesn't have response!
    for i, f in enumerate(testfiles):
        print(f)
        subset = None
        nrows = 0
        for i, chunk in enumerate(pd.read_csv(f,
                                              usecols=cols[i],
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            #print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
            
            nrows += CHUNKSIZE
            if nrows >= NROWS:
                break
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()


    
    testdata['Response'] = 0
    visibletraindata = traindata[::2]
    blindtraindata = traindata[1::2]

    blindtraindata2 = traindata[::2].copy()
    visibletraindata2 = traindata[1::2].copy()
    testdata2 = testdata.copy()

    # To not LOO the Start and Finish
#    del cols[2][1:3]

    for i in range(3):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, False).values
            testdata.loc[:, col] = LeaveOneOut(visibletraindata,
                                               testdata, col, False).values

    for i in range(3):
        for col in cols[i][1:]:
            print(col)
            blindtraindata2.loc[:, col] = LeaveOneOut(visibletraindata2,
                                                     blindtraindata2,
                                                     col, False).values
            testdata2.loc[:, col] = LeaveOneOut(visibletraindata2,
                                               testdata2, col, False).values

    del visibletraindata
    del visibletraindata2
    gc.collect()
    testdata.drop('Response', inplace=True, axis=1)
    testdata2.drop('Response', inplace=True, axis=1)
    return blindtraindata, testdata, blindtraindata2, testdata2

def applyXGB(train, test):
    features = train.columns[1:-1]
    #print(features)
    num_rounds = NROUNDS
		
    """response1 = train[train.Response == 1].shape[0]
    response0 = train.shape[0] - response1
    print("Response1 = {0}".format(response1))
    print("Response0 = {0}".format(response0))
    scale_pos_weight = float(response0 / response1)
    print("scale_pos_weight = {0}".format(scale_pos_weight))"""
    
    params = {}
#    params['booster'] = "gbtree"
    params['objective'] = "binary:logistic"
    params['n_estimators'] = 500
#    params['eta'] = 0.021
    params['eta'] = 0.025
    params['max_depth'] = 8
    params['colsample_bytree'] = 0.82
    params['min_child_weight'] = 3
    params['base_score'] = 0.005
    params['silent'] = True
#    params['scale_pos_weight'] =  scale_pos_weight
#    params['max_delta_step'] = 5

    print('Fitting')
    trainpredictions = None
    testpredictions = None

    y_true = train.Response.values
    dvisibletrain = xgb.DMatrix(train[features], 
                                y_true, 
                                silent=True)
        
    dtest = xgb.DMatrix(test[features], silent=True)

    folds = NFOLDS
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

        best_proba, best_mcc, y_pred = eval_mcc(y_true,
                                                predictions,
                                                True)
        print('tree limit:', limit)
        print('mcc:', best_mcc)
        matthews_corr = matthews_corrcoef(y_true,
                                y_pred)
        print(matthews_corrcoef(y_true,
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

    best_proba, best_mcc, y_pred = eval_mcc(y_true,
                                            trainpredictions/folds,
                                            True)
    print('best_proba: {0} best_mcc: {1}'.format(best_proba, best_mcc))
    print(print(np.bincount(y_pred)))
    matthews_corr = matthews_corrcoef(y_true, y_pred)
    print(params)
    print('Folds:{0}'.format(NFOLDS))
    return testpredictions/folds, best_mcc

def ExtractFeatures():
    ID_COLUMN = 'Id'
    TARGET_COLUMN = 'Response'
    TRAIN_NUMERIC = "{0}train_numeric.csv".format(DATA_DIR)
    TRAIN_DATE = "{0}train_date.csv".format(DATA_DIR)
    TEST_NUMERIC = "{0}test_numeric.csv".format(DATA_DIR)
    TEST_DATE = "{0}test_date.csv".format(DATA_DIR)

    train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)
    test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)

    train["StartTime"] = -1
    test["StartTime"] = -1
    train["FinishTime"] = -1
    test["FinishTime"] = -1
    train["Duration"] = -1
    test["Duration"] = -1
        
    nrows = 0
    for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
        feats = np.setdiff1d(tr.columns, [ID_COLUMN])
    
        stime_tr = tr[feats].min(axis=1).values
        stime_te = te[feats].min(axis=1).values
    
        train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
        test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te
        
        ftime_tr = tr[feats].max(axis=1).values
        ftime_te = te[feats].max(axis=1).values
    
        train.loc[train.Id.isin(tr.Id), 'FinishTime'] = ftime_tr
        test.loc[test.Id.isin(te.Id), 'FinishTime'] = ftime_te
        
        duration_tr = ftime_tr - stime_tr
        duration_te = ftime_te - stime_te
        
        train.loc[train.Id.isin(tr.Id), 'Duration'] = duration_tr
        test.loc[test.Id.isin(te.Id), 'Duration'] = duration_te
    
        del tr
        del te
        gc.collect()
        
        nrows += CHUNKSIZE
        if nrows >= NROWS:
            break

    ntrain = train.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)
    
    train_test['0_¯\_(ツ)_/¯_1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
    train_test['0_¯\_(ツ)_/¯_2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
    
    train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)
    
    train_test['0_¯\_(ツ)_/¯_3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
    train_test['0_¯\_(ツ)_/¯_4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
    
    train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)
    train = train_test.iloc[:ntrain, :]
    test = train_test.iloc[ntrain:, :]
    
    train.to_csv('train_ex.csv', Index=False)
    test.drop(['Response'], axis = 1, inplace = True, errors = 'ignore')
    test.to_csv('test_ex.csv', Index=False)
    
    del train
    del test
    del train_test
    gc.collect()
    
def Train():
    train, test, train2, test2 = GrabData()
    print('Train:', train.shape)
    print('Test', test.shape)
    print('Train2:', train2.shape)
    print('Test2', test2.shape)

#    xgbGridSearch(train)
#    xgbGridSearch(train2)

    testpredictions1, score1 = applyXGB(train, test)
    testpredictions2, score2 = applyXGB(train2, test2)

    testpredictions_avg = (testpredictions1 + testpredictions2)/2
    testpredictions_w = (testpredictions1*score1 + testpredictions2*score2)/(score1 + score2)

    print('Model1 best score: {0}'.format(score1))
    print(testpredictions1)
    print('Model2 best score: {0}'.format(score2))
    print(testpredictions2)
    print('Avg score: {0}'.format((score1 + score2)/2))
    print(testpredictions_avg)
    print('Weighted:')
    print(testpredictions_w)

    submission = pd.DataFrame({"Id": test.Id.values, "Response": testpredictions1})
    submission[['Id', 'Response']].to_csv('rawxgbsubmission1.csv', index=False)
    submission = pd.DataFrame({"Id": test.Id.values, "Response": testpredictions2})
    submission[['Id', 'Response']].to_csv('rawxgbsubmission2.csv', index=False)
    submission = pd.DataFrame({"Id": test.Id.values, "Response": testpredictions_avg})
    submission[['Id', 'Response']].to_csv('rawxgbsubmission_mean.csv', index=False)
    submission = pd.DataFrame({"Id": test.Id.values, "Response": testpredictions_w})
    submission[['Id', 'Response']].to_csv('rawxgbsubmission_wmean.csv', index=False)

    # pick the best threshold out-of-fold
#    thresholds = np.linspace(0.01, 0.99, 50)
#    mcc = np.array([matthews_corrcoef(train.Response.values, testpredictions_w>thr) for thr in thresholds])
#    plt.plot(thresholds, mcc)
#    best_threshold = thresholds[mcc.argmax()]
#    print(mcc.max())

#    d = pd.read_csv('rawxgbsubmission_wmean.csv')
#    resultsToCsv('xgbsubmission3', d.Id.values, (d.Response.values > .3).astype(int))

    resultsToCsv('xgbsubmission20', test.Id.values, (testpredictions_avg > .2).astype(int))
    resultsToCsv('xgbsubmission25', test.Id.values, (testpredictions_avg > .25).astype(int))
    resultsToCsv('xgbsubmission255', test.Id.values, (testpredictions_avg > .255).astype(int))
    resultsToCsv('xgbsubmission30', test.Id.values, (testpredictions_avg > .3).astype(int))

    resultsToCsv('xgbsubmissionw20', test.Id.values, (testpredictions_w > .2).astype(int))
    resultsToCsv('xgbsubmissionw25', test.Id.values, (testpredictions_w > .25).astype(int))
    resultsToCsv('xgbsubmissionw255', test.Id.values, (testpredictions_w > .255).astype(int))
    resultsToCsv('xgbsubmissionw30', test.Id.values, (testpredictions_w > .3).astype(int))

if __name__ == "__main__":
    print('Started')    
    ExtractFeatures()
    Train()
    print('Finished')
    