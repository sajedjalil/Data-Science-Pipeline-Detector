# In The Name of Allah

import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold, train_test_split
from operator import itemgetter


categorical_features = [
'Id',  "L3_S32_F3854", "L1_S25_F1852"
]

numeric_features = [
'Id', "L0_S0_F0", "L0_S0_F2", "L0_S0_F4", "L0_S0_F6", "L0_S0_F8",
"L0_S0_F10", "L0_S0_F12", "L0_S0_F14", "L0_S0_F16", "L0_S0_F18",
"L0_S0_F20", "L0_S0_F22", "L0_S1_F24", "L0_S1_F28", "L0_S2_F36",
"L0_S2_F44", "L0_S2_F60", "L0_S3_F72", "L0_S3_F80", "L0_S3_F96",
"L0_S3_F100", "L0_S5_F114", "L0_S5_F116", "L0_S6_F118", "L0_S6_F122",
"L0_S6_F132", "L0_S7_F138", "L0_S8_F144", "L0_S8_F149", "L0_S9_F155",
"L0_S9_F160", "L0_S9_F170", "L0_S9_F185", "L0_S9_F195", "L0_S10_F224",
"L0_S10_F239", "L0_S10_F244", "L0_S10_F249", "L0_S10_F254", "L0_S10_F259",
"L0_S10_F274", "L0_S11_F282", "L0_S11_F286", "L0_S11_F294", "L0_S11_F298",
"L0_S11_F302", "L0_S11_F306", "L0_S11_F310", "L0_S11_F314", "L0_S11_F318",
"L0_S11_F322", "L0_S11_F326", 
'L1_S24_F1844', 'L1_S24_F1846', 'L2_S26_F3073',
"L3_S29_F3315" ,"L3_S29_F3318" ,"L3_S29_F3321",
"L3_S29_F3324", "L3_S29_F3327" ,"L3_S29_F3330" ,"L3_S29_F3336" ,"L3_S29_F3339",
"L3_S29_F3342", "L3_S29_F3345" ,"L3_S29_F3348" ,"L3_S29_F3351" ,"L3_S29_F3354",
"L3_S29_F3373", "L3_S29_F3376", "L3_S29_F3379" ,"L3_S29_F3382" ,"L3_S29_F3407",
"L3_S29_F3427", "L3_S29_F3430", "L3_S29_F3433" ,"L3_S29_F3436" ,"L3_S29_F3461",
"L3_S29_F3464", "L3_S29_F3479", "L3_S30_F3519" ,"L3_S30_F3534", "L3_S30_F3554",
"L3_S30_F3574", "L3_S30_F3579", "L3_S30_F3589", "L3_S30_F3604", "L3_S30_F3609",
"L3_S30_F3629", "L3_S30_F3639", "L3_S30_F3649", "L3_S30_F3669", "L3_S30_F3679",
"L3_S30_F3689", "L3_S30_F3704", "L3_S30_F3744", "L3_S30_F3749", "L3_S30_F3754",
"L3_S30_F3759", "L3_S30_F3764", "L3_S30_F3769", "L3_S30_F3774", "L3_S30_F3779",
"L3_S30_F3794", "L3_S30_F3799", "L3_S30_F3804", "L3_S30_F3809", "L3_S30_F3829",
"L3_S33_F3855", "L3_S33_F3857", "L3_S33_F3859", "L3_S33_F3865", "L3_S35_F3896",
"L3_S36_F3920", "L3_S38_F3956", "Response"
]


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
#        if f == 'Id' or 'S24' in f or 'S25' in f:
        if f == 'Id' in f:
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
#        print(i)
        
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
#        print(i)
        
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        subset = pd.concat([subset, df_mindate_chunk])
        
        del chunk
        gc.collect()      
        
    return subset


df_mindate = get_mindate()

#df_mindate.sort_values(by=['mindate', 'Id'], inplace=True)

#df_mindate['mindate_id_diff'] = df_mindate.Id.diff()

#midr = np.full_like(df_mindate.mindate_id_diff.values, np.nan)
#midr[0:-1] = -df_mindate.mindate_id_diff.values[1:]

#df_mindate['mindate_id_diff_reverse'] = midr

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


def GrabData1():
    directory = '../input/'
    trainfiles = ['train_categorical.csv',
                  'train_date.csv',
                  'train_numeric.csv']
    
    testfiles = ['test_categorical.csv',
                 'test_date.csv',
                 'test_numeric.csv']

    cols = [ categorical_features,
            usefuldatefeatures,
            numeric_features]


    traindata = None
    testdata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
#            print(i)
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

 #   del cols[2][-1]  # Test doesn't have response!
  #  for i, f in enumerate(testfiles):
   #     print(f)
    #    subset = None
     #   for i, chunk in enumerate(pd.read_csv(directory + f,
      #                                        usecols=cols[i],
       #                                       chunksize=50000,
        #                                      low_memory=False)):
#            print(i)
         #   if subset is None:
          #      subset = chunk.copy()
           # else:
            #    subset = pd.concat([subset, chunk])
            #del chunk
            #gc.collect()
        #if testdata is None:
         #   testdata = subset.copy()
        #else:
         #   testdata = pd.merge(testdata, subset.copy(), on="Id")
        #del subset
        #gc.collect()

    traindata = traindata.merge(df_mindate, on='Id')
#    testdata = testdata.merge(df_mindate, on='Id')
        
    #testdata['Response'] = 0  # Add Dummy Value
    visibletraindata = traindata[::2]
    blindtraindata = traindata[1::2]

    for i in range(2):
        for col in cols[i][1:]:
##            print(col)
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, False).values
     #       testdata.loc[:, col] = LeaveOneOut(visibletraindata,
      #                                         testdata, col, False).values
    del visibletraindata
    gc.collect()
    #testdata.drop('Response', inplace=True, axis=1)
    return blindtraindata


def Train1():
    train = GrabData1()
    print('Train:', train.shape)
#    print('Test', test.shape)
    features = list(train.columns)
    features.remove('Response')
    features.remove('Id')
#    print(features)
    num_rounds = 25
    params = {}
    params['objective'] = "binary:logistic"
    params['eta'] = 0.021
    params['max_depth'] = 20
    params['colsample_bytree'] = 0.82
    params['min_child_weight'] = 3
    params['base_score'] = 0.005
    params['silent'] = True


    print('Fitting')
#    trainpredictions = None
#    testpredictions = None
    
    
#    X_train, X_valid = train_test_split(train, test_size=0.2)



    dtrain = xgb.DMatrix(train[features], train.Response, silent=True)
    del train
    
#    dtrain = xgb.DMatrix(X_train[features], X_train.Response, silent=True)
 #   dvalid = xgb.DMatrix(X_valid[features], X_valid.Response, silent=True)
                    
#    del X_train, X_valid
    
    folds = 1
    for i in range(folds):
        print('Fold:', i)
        params['seed'] = i
        watchlist = [(dtrain, 'train'), (dtrain, 'val')]
        clf = xgb.train(params, dtrain,
                        num_boost_round=num_rounds,
                        evals=watchlist,
                        early_stopping_rounds=20,
                        feval=mcc_eval, maximize=True)
        limit = clf.best_iteration+1

        imp = get_importance(clf, features)
        print('Importance array: ', imp)
        
        del dtrain        

        #dtest =  xgb.DMatrix(test[features])
        #Idtest = test.Id.values

        #del test
        #predictions = clf.predict(dtest, ntree_limit=limit)



    #submission = pd.DataFrame({"Id": Idtest,
     #                          "Response": predictions/folds})
    #submission[['Id', 'Response']].to_csv('rawxgb_11_11_end.csv',
     #                                     index=False)

if __name__ == "__main__":
    print('Started')
    Train1()
    print('Finished')

