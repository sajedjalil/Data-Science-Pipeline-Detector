import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter

NROWS = 20000000000
CHUNKSIZE = 35000
DATA_DIR = "../input/"

# per raddar, all date features except for stations 24+25 are identical

def get_date_features():
    directory = '../input/'
    trainfile = 'train_date.csv'
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          chunksize=1,
                                          low_memory=False)):
        features = list(chunk.columns)
        del chunk
        break

    seen = np.zeros(52)
    rv = []
    for f in features:
        if f == 'Id' or 'S24' in f or 'S25' in f:
            rv.append(f)
            continue
            
        station = int(f.split('_')[1][1:])
        
        if seen[station]:
            continue
        
        seen[station] = 1
        rv.append(f)
        
    return rv
        
usefuldatefeatures = get_date_features()

def get_mindate():
    directory = '../input/'
    trainfile = 'train_date.csv'
    testfile = 'test_date.csv'
    
    features = None
    subset = None
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          usecols=usefuldatefeatures,
                                          chunksize=50000,
                                          low_memory=False)):
        print(i)
        
        if features is None:
            features = list(chunk.columns)
            features.remove('Id')
        
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        
        if subset is None:
            subset = df_mindate_chunk.copy()
        else:
            subset = pd.concat([subset, df_mindate_chunk])
            
        del chunk
        del df_mindate_chunk
        gc.collect()

    for i, chunk in enumerate(pd.read_csv(directory + testfile,
                                          usecols=usefuldatefeatures,
                                          chunksize=50000,
                                          low_memory=False)):
        print(i)
        
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        subset = pd.concat([subset, df_mindate_chunk])
        
        del chunk
        del df_mindate_chunk
        gc.collect()      
        
    return subset

def getdf_mindate():
    df_mindate = get_mindate()
    
    df_mindate.sort_values(by=['mindate', 'Id'], inplace=True)
    
    df_mindate['mindate_id_diff'] = df_mindate.Id.diff()
    
    midr = np.full_like(df_mindate.mindate_id_diff.values, np.nan)
    midr[0:-1] = -df_mindate.mindate_id_diff.values[1:]
    
    df_mindate['mindate_id_diff_reverse'] = midr
    del midr
    return df_mindate

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
    
def Train32_33_34() :
    STATIONS = ['S32', 'S33', 'S34']
    test_date_part = pd.read_csv('../input/test_date.csv', nrows=10000)
    
    
    train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    date_cols = date_cols[date_cols['station'].isin(STATIONS)]
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
    print(date_cols)
    train_date = pd.read_csv('../input/train_date.csv', usecols=['Id'] + date_cols)
    print(train_date.columns)
    train_date.columns = ['Id'] + STATIONS
    for station in STATIONS:
        train_date[station] = 1 * (train_date[station] >= 0)

    response = pd.read_csv('../input/train_numeric.csv', usecols=['Id'])
    response['Response_magic'] = 0
    train = response.merge(train_date, how='left', on='Id')
    train['cnt'] = 1
    
    train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
    # Train start dates
    train_start_date = pd.read_csv('../input/train_date.csv', usecols=['Id'] + date_cols)
    train_start_date['start_date'] = train_start_date[date_cols].min(axis=1)
    train_start_date = train_start_date.drop(date_cols, axis=1)
    print(train_start_date.shape)
    # Test start dates
    test_start_date = pd.read_csv('../input/test_date.csv', usecols=['Id'] + date_cols)
    test_start_date['start_date'] = test_start_date[date_cols].min(axis=1)
    test_start_date = test_start_date.drop(date_cols, axis=1)
    print(test_start_date.shape)
    start_date = pd.concat([train_start_date, test_start_date])
    print(start_date.shape)
    del train_start_date, test_start_date
    gc.collect()
    start_date.head()
    
    
    
    train_id = pd.read_csv('../input/train_numeric.csv', usecols=['Id'])
    test_id = pd.read_csv('../input/test_numeric.csv', usecols=['Id'])
    train_id = train_id.merge(start_date, on='Id')
    test_id = test_id.merge(start_date, on='Id')
    train_test_id = pd.concat((train_id, test_id)).reset_index(drop=True).reset_index(drop=False)
    train_test_id = train_test_id.sort_values(by=['start_date', 'Id'], ascending=True)
    train_test_id['IdDiff1'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
    train_test_id['IdDiff2'] = train_test_id['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
    train_test_id['Magic'] = 1 + 2 * (train_test_id['IdDiff1'] > 1) + 1 * (train_test_id['IdDiff2'] < -1)
    
    train_with_magic = train.merge(train_test_id[['Id', 'Magic']], on='Id')
    return train_with_magic


def Test32_33_34() :
    
    STATIONS = ['S32', 'S33', 'S34']
    
    
    train_date_part = pd.read_csv('../input/test_date.csv', nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    date_cols = date_cols[date_cols['station'].isin(STATIONS)]
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
    print(date_cols)
    train_date = pd.read_csv('../input/test_date.csv', usecols=['Id'] + date_cols)
    print(train_date.columns)
    train_date.columns = ['Id'] + STATIONS
    for station in STATIONS:
        train_date[station] = 1 * (train_date[station] >= 0)
    response = pd.read_csv('../input/test_numeric.csv', usecols=['Id'])
    response['Response_magic'] = 0
    
    print(response.shape)
    train = response.merge(train_date, how='left', on='Id')
    train['cnt'] = 1
    
    train_date_part = pd.read_csv('../input/test_date.csv', nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    
    
    train_start_date = pd.read_csv('../input/test_date.csv', usecols=['Id'] + date_cols)
    train_start_date['start_date'] = train_start_date[date_cols].min(axis=1)
    train_start_date = train_start_date.drop(date_cols, axis=1)
    print(train_start_date.shape)
    
    
    start_date = train_start_date
    print(start_date.shape)
    
    
    del train_start_date
    gc.collect()
    start_date.head()
    
    
    train_id = pd.read_csv('../input/test_numeric.csv', usecols=['Id'])
    print("train_id org")
    print(train_id)
    
    train_id = train_id.merge(start_date, on='Id')
    
    print("train_id")
    print(train_id)
    
    train_test_id = train_id.reset_index(drop=True).reset_index(drop=False)
    train_test_id = train_test_id.sort_values(by=['start_date', 'Id'], ascending=True)
    train_test_id['IdDiff1'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
    train_test_id['IdDiff2'] = train_test_id['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
    train_test_id['Magic'] = 1 + 2 * (train_test_id['IdDiff1'] > 1) + 1 * (train_test_id['IdDiff2'] < -1)
    
    print("train_test_id")
    print(train_test_id)
    
    train_with_magic = train.merge(train_test_id[['Id', 'Magic']], on='Id')
    train_with_magic.head()
    train_with_magic['Response_magic'] = 100 * train_with_magic['Magic'] + 10 * train_with_magic['S32'] + train_with_magic['S33']
    train_with_magic['Response_magic'] = 1 * ((train_with_magic['Response_magic'] == 310) | (train_with_magic['Response_magic'] == 110))
    return train_with_magic

def GrabData():

    trainfiles = [DATA_DIR + 'train_categorical.csv',
                  DATA_DIR + 'train_date.csv',
                  'train_ex.csv',
                  DATA_DIR + 'train_numeric.csv']
    testfiles = [DATA_DIR + 'test_categorical.csv',
                 DATA_DIR + 'test_date.csv',
                 'test_ex.csv',
                 DATA_DIR + 'test_numeric.csv']

    cols = [['Id',
             'L1_S24_F1559', 'L3_S32_F3851',
             'L1_S24_F1827', 'L1_S24_F1582',
             'L3_S32_F3854', 'L1_S24_F1510',
             'L1_S24_F1525'],
            ['Id',
             'L3_S30_D3496', 'L3_S30_D3506',
             'L3_S30_D3501', 'L3_S30_D3516',
             'L3_S30_D3511'],
             ['Id', 'StartTime',
             'test_1','test_2',
             'test_3','test_4'],
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
        nrows = 0
        for i, chunk in enumerate(pd.read_csv(f,
                                              usecols=cols[i],
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
            
            nrows += CHUNKSIZE
            if nrows >= NROWS:
                print("break!!!!!")
                print(nrows)
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
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
            
            nrows += CHUNKSIZE
            if nrows >= NROWS:
                print("break!!!!!")
                print(nrows)
                break
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()
        
    df_mindate = getdf_mindate()
    traindata = traindata.merge(df_mindate, on='Id')
    testdata = testdata.merge(df_mindate, on='Id')
    del df_mindate
    #traindata = traindata.merge(Train32_33_34(), on='Id')
    #testdata = testdata.merge(Test32_33_34(), on='Id')
        
    testdata['Response'] = 0  # Add Dummy Value
    visibletraindata = traindata[::2]
    blindtraindata = traindata[1::2]
    print(blindtraindata.columns)
    for i in range(3):
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
    features = list(train.columns)
    features.remove('Response')
    features.remove('Id')
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

    #submission = pd.DataFrame({"Id": train.Id,
    #                           "Prediction": trainpredictions/folds,
    #                           "Response": train.Response})
    #submission[['Id',
    #           'Prediction',
    #           'Response']].to_csv('rawtrainxgbsubmission'+str(folds)+'.csv',
    #                               index=False)
    #submission = pd.DataFrame({"Id": test.Id.values,
    #                           "Response": testpredictions/folds})
    #submission[['Id', 'Response']].to_csv('rawxgbsubmission'+str(folds)+'.csv',
    #                                      index=False)
    
    pd.DataFrame({"Id": test.Id.values, 
    "Response": (testpredictions/folds > .08).astype(int)}).to_csv(
        'xgbsubmission08.csv', index=False)
        
    pd.DataFrame({"Id": test.Id.values, 
    "Response": (testpredictions/folds > .09).astype(int)}).to_csv(
        'xgbsubmission09.csv', index=False)
        
    pd.DataFrame({"Id": test.Id.values, 
    "Response": (testpredictions/folds > .1).astype(int)}).to_csv(
        'xgbsubmission10.csv', index=False)        
        
    pd.DataFrame({"Id": test.Id.values, 
    "Response": (testpredictions/folds > .2).astype(int)}).to_csv(
        'xgbsubmission20.csv', index=False)
        
    pd.DataFrame({"Id": test.Id.values, 
    "Response": (testpredictions/folds > .3).astype(int)}).to_csv(
        'xgbsubmission30.csv', index=False)                

if __name__ == "__main__":
    print('Started')
    
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
        
    nrows = 0
    for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
        feats = np.setdiff1d(tr.columns, [ID_COLUMN])
    
        stime_tr = tr[feats].min(axis=1).values
        stime_te = te[feats].min(axis=1).values
    
        train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
        test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te
    
        nrows += CHUNKSIZE
        if nrows >= NROWS:
            break


    ntrain = train.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)
    
    train_test['test_1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
    train_test['test_2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
    
    train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)
    
    train_test['test_3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
    train_test['test_4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
    
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
    
    Train()
    print('Finished')