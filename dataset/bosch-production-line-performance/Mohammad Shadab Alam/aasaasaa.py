import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter

# per raddar, all date features except for stations 24+25 are identical

def get_date_features():
    directory = '../input/'
    trainfile = 'train_date.csv'
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          chunksize=1,
                                          low_memory=False)):
        features = list(chunk.columns)
        break

    seen = np.zeros(52)
    rv = []
    for f in features:
        if f == 'Id' or 'S24' in f or 'S25' in f:
            rv.append(f)
            continue
            
        station = int(f.split('_')[1][1:])
#        print(station)
        
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
        gc.collect()      
        
    return subset


df_mindate = get_mindate()

df_mindate.sort_values(by=['mindate', 'Id'], inplace=True)

df_mindate['mindate_id_diff'] = df_mindate.Id.diff()

midr = np.full_like(df_mindate.mindate_id_diff.values, np.nan)
midr[0:-1] = -df_mindate.mindate_id_diff.values[1:]

df_mindate['mindate_id_diff_reverse'] = midr

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


def LeaveOneOut(data1, data2, columnName, useLOO=False,num=1):
    grpOutcomes = data1.groupby(columnName)['Response'].mean().reset_index()
    grpCount = data1.groupby(columnName)['Response'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.Response
    if(useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > num]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['Response'].values
    x = pd.merge(data2[[columnName, 'Response']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['Response']
   # if(useLOO):
       # x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
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

    cols = [['Id',
             'L1_S24_F1559', 'L3_S32_F3851',
             'L1_S24_F1827', 'L1_S24_F1582',
             'L3_S32_F3854', 'L1_S24_F1510',
             'L1_S24_F1525'],
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
        
    traindata = traindata.merge(df_mindate, on='Id')
    testdata = testdata.merge(df_mindate, on='Id')
        
    testdata['Response'] = 0  # Add Dummy Value
    visibletraindata = traindata[1::2]
    blindtraindata = traindata[::2]
    print(blindtraindata.columns)
    num=30
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col+str(num)] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, True,num).values
            testdata.loc[:, col+str(num)] = LeaveOneOut(visibletraindata,
                                               testdata, col, True,num).values
                                               
    num=20
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col+str(num)] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, True,num).values
            testdata.loc[:, col+str(num)] = LeaveOneOut(visibletraindata,
                                               testdata, col, True,num).values
    num=10
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col+str(num)] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, True,num).values
            testdata.loc[:, col+str(num)] = LeaveOneOut(visibletraindata,
                                               testdata, col, True,num).values
    
    num=5
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, True,num).values
            testdata.loc[:, col] = LeaveOneOut(visibletraindata,
                                               testdata, col, True,num).values  
                                               
    num=2
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, True,num).values
            testdata.loc[:, col] = LeaveOneOut(visibletraindata,
                                               testdata, col, True,num).values 
    del visibletraindata
    gc.collect()
    testdata.drop('Response', inplace=True, axis=1)
    return blindtraindata, testdata


def Train():
    train, test = GrabData()
    
    train.to_csv('train_xtra.csv.gz',index=False,compression="gzip")
    test.to_csv('test_xtra.csv.gz',index=False,compression="gzip")

if __name__ == "__main__":
    print('Started')
    Train()
    print('Finished')

