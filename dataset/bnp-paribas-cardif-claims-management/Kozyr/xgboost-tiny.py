import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.metrics import log_loss
import xgboost as xgb
import random

train_path         = '../input/train.csv'
test_path          = '../input/test.csv'
num_rounds         = 150
na_value           = -100
drop_columns       = ['ID','target']
drop_columns      += ['v22','v91']
#drop_columns     += ['v1','v100','v101','v102','v103','v104','v105','v106','v108','v109','v11','v111','v115','v116','v117','v118','v119','v120','v121','v122','v123','v124','v126','v127','v128','v13','v130','v131','v15','v16','v17','v18','v19','v2','v20','v23','v25','v26','v27','v28','v29','v32','v33','v35','v36','v37','v39','v4','v41','v42','v43','v44','v45','v46','v48','v49','v5','v51','v53','v54','v55','v57','v58','v59','v6','v60','v61','v63','v64','v65','v67','v68','v69','v7','v70','v73','v76','v77','v78','v8','v80','v81','v82','v83','v84','v85','v86','v87','v88','v89','v9','v90','v92','v93','v94','v95','v96','v97','v98','v99',]
categorical_colums = ['v3','v22','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75','v79','v107','v110','v112','v113','v125']
target_column      = 'target'

use_columns1       = ['v10','v100','v101','v106','v110','v113','v114','v119','v12','v121','v123','v129','v130','v14','v17','v21','v31','v33','v34','v36','v38','v4','v44','v47','v48','v50','v51','v55','v61','v62','v64','v66','v72','v76','v83','v85','v88','v93']
use_columns2       = ['v10','v100','v101','v106','v110','v113','v114','v119','v12','v121','v123','v129','v130','v14','v17','v21','v31','v33','v34','v36','v38','v4','v44','v47','v48','v50','v51','v55','v61','v62','v64','v66','v72','v76','v83','v85','v88','v93']

#x2                 = ['v12','v21','v71?',]

# binner from the script:
# https://www.kaggle.com/happycube/bnp-paribas-cardif-claims-management/simple-cv-example-one-feature-near-rf/code
def binner(all_data, key, maxbins = 101, na = -100, percent_per_bin = 1):
    akey = all_data.loc[all_data[key] != na, key].copy()
    count = len(akey.unique())
    
    if count < maxbins:
        return (all_data[key], None)
    try:
        bins = np.unique(np.percentile(akey, np.arange(0, 100, percent_per_bin)))
        # Add a bin for NA
        if np.min(all_data[key]) == na:
            bins = np.insert(bins, 0, na + 1)
        count = len(bins)
    
        # print(key, count)
    
        return (np.digitize(all_data[key], bins), bins)
    except:
        return (all_data[key], None)

def load_data(cross_validation = 0.2):
    print('Loading data')    
    train   = pd.read_csv(train_path)
    test    = pd.read_csv(test_path)
    
    # combine train and test
    all_data = train.append(test)

    print('Eliminate missing values')
    all_data.fillna(na_value, inplace=True)
    
    print('Categorical values')
    for col in categorical_colums:
        if not col in drop_columns:
            all_data[col] = pd.factorize(all_data[col])[0]

    #print('Bins')
    #for col in ['v10', 'v14', 'v114', 'v34', 'v50', 'v56']: #all_data.columns:
    #    if not col in drop_columns:
    #        (all_data[col], col_bins) = binner(all_data, col, percent_per_bin = 5)

    #print('Log transformation')
    #for col in all_data.columns:
    #    if not col in drop_columns:
    #        m = all_data[col].min()
    #        all_data[col].apply(lambda x: np.log(1 + m + x))

    print('Arctan transformation')
    for col in ['v1', 'v2', 'v4']:
        if not col in drop_columns:
            m = all_data[col].mean()
            all_data[col].apply(lambda x: np.arctan(x-m))


    # split train and test
    N       = train.shape[0]
    N_cv    = int(N * cross_validation)
    N_train = N - N_cv
    
    train = all_data[all_data[target_column] >=0][:N_train].copy()
    cv    = all_data[all_data[target_column] >=0][N_train:N].copy()
    test  = all_data[all_data[target_column] < 0].copy()

    train1 = train[train['v1'] != na_value]
    train2 = train[train['v1'] == na_value]

    cv1    = cv[cv['v1'] != na_value]
    cv2    = cv[cv['v1'] == na_value]

    test1  = test[test['v1'] != na_value]
    test2  = test[test['v1'] == na_value]
    
    return (train1, cv1, test1), (train2, cv2, test2)

def calc():
    (train1, cv1, test1), (train2, cv2, test2) = load_data(0.2)
    
    #drop_columns = list(set(train1.columns) - set(use_columns1))
    
    y1      = train1[target_column]
    X1      = train1.drop(drop_columns, axis=1)
    y1_cv   = cv1[target_column]
    X1_cv   = cv1.drop(drop_columns, axis=1)
    X1_test = test1.drop(drop_columns, axis=1)
    
    #drop_columns = list(set(train2.columns) - set(use_columns2))
    
    y2      = train2[target_column]
    X2      = train2.drop(drop_columns, axis=1)
    y2_cv   = cv2[target_column]
    X2_cv   = cv2.drop(drop_columns, axis=1)
    X2_test = test2.drop(drop_columns, axis=1)

    print('Train model')
    xgtrain   = xgb.DMatrix(X1, y1)
    xgval     = xgb.DMatrix(X1_cv, y1_cv)
    #xgtest    = xgb.DMatrix(X1_test)

    params = {
        'objective':'binary:logistic',
        'eval_metric':'logloss',
        'eta' :             0.05,
        'min_child_weight': 1,
        'subsample':        0.9,
        'colsample_bytree': 0.9,
        'max_depth':        8,
        'silent':           0,
    }

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(list(params.items()), xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

    xgtrain   = xgb.DMatrix(X2, y2)
    xgval     = xgb.DMatrix(X2_cv, y2_cv)
    #xgtest    = xgb.DMatrix(X2_test)

    model = xgb.train(list(params.items()), xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

    #print('Make prediction')
    #y_pred = model.predict(xgtest)

    #print('Write to CSV')
    #submit = pd.read_csv('../input/sample_submission.csv')
    #submit['PredictedProb'] = np.clip(y_pred, 0.01, 0.99)
    #submit.to_csv('xgb.csv',index=False)

calc()