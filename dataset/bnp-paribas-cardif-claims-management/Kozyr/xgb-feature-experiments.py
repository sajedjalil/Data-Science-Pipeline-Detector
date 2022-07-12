import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.metrics import log_loss
import xgboost as xgb
import random

train_path         = '../input/train.csv'
test_path          = '../input/test.csv'
num_rounds         = 100
drop_columns       = ['ID','target']
drop_columns      += ['v22','v91']
categorical_colums = ['v3','v22','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75','v79','v107','v110','v112','v113','v125']
target_column      = 'target'

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

def prob_replacer(raw):
    elements, repeats = np.unique(raw, return_counts=True)
    prob_dict = dict(zip(elements, repeats))
    
    rows = sum(repeats)
    data  = np.zeros((rows, 1))
    for idx, value in raw.iteritems():
        data[idx] = prob_dict[value] / float(rows)

    return data
    
def load_data(cross_validation = 0.2):
    print('Loading data')    
    train   = pd.read_csv(train_path)
    test    = pd.read_csv(test_path)
    
    N       = train.shape[0]
    N_cv    = int(N * cross_validation)
    N_train = N - N_cv
    
    # combine train and test
    all_data = train.append(test)

    #print('Eliminate missing values')
    #all_data.fillna(-100, inplace=True)
    
    print('Categorical values')
    for col in categorical_colums:
        all_data[col] = pd.factorize(all_data[col])[0]

    #print('Arctan transformation')
    #for col in list(set(all_data.columns) - set(categorical_colums) - set(drop_columns)):
    #    m = all_data[col].mean()
    #    all_data[col+'_atan'] = all_data[col].apply(lambda x: np.arctan(x-m))

    #print('Bins')
    #for col in all_data.columns:
    #    if not col in drop_columns:
    #        (all_data[col+'bin'], col_bins) = binner(all_data, col, percent_per_bin = 5)

    #print('Probabilities')
    #for col in all_data.columns[:4]:
    #    if not col in drop_columns:
    #        all_data[col+'_p'] = prob_replacer(all_data[col])

    #print('Log transformation')
    #for col in all_data.columns:
    #    if not col in drop_columns:
    #        m = all_data[col].min()
    #        all_data[col+'_log'] = all_data[col].copy()
    #        all_data[col+'_log'].apply(lambda x: np.log(1 + m + x))

    # split on v1
    print('Origin:',all_data.shape)
    all_data = all_data[all_data['v1'].notnull()]
    print('No blank v1:',all_data.shape)

    # split train and test
    train = all_data[all_data[target_column] >=0][:N_train].copy()
    cv    = all_data[all_data[target_column] >=0][N_train:N].copy()
    test  = all_data[all_data[target_column] < 0].copy()
    
    return train, cv, test

def calc():
    train, cv, test = load_data(0.2)
    
    data = pd.DataFrame()
    data[target_column] = train[target_column]

    v = []
    corr_single = train.corr()[target_column]
    for idx, value in corr_single.iteritems():
        if not idx in drop_columns:
            if np.abs(value) > 0.045:
                print(idx,value)
                v.append(idx)
            
            for col1 in train.columns:
                if not col1 in drop_columns:
                    col = idx+'_x_'+col1
        
                    data['v'] = train[idx] * train[col1]
                
                    c = data.corr()[target_column]
                    cc = np.abs(c['v'])
                    if cc > 0.05 and cc > np.abs(value) and cc > np.abs(corr_single[col1]):
                        print(col, c['v'])
                        v.append(col)
                
    print(v)
    return
    
    corr = pd.DataFrame()
    
    for col1 in train.columns:
        if not col1 in drop_columns:
            data = pd.DataFrame()
            data[target_column] = train[target_column]
            data[col1] = train[col1]
            
            for col2 in train.columns:
                if not col2 in drop_columns:
                    if col1 != col2:
                        col = col1+'_x_'+col2
                        data[col] = train[col1] * train[col2]

            corr_col = pd.DataFrame(data.corr())
            corr = pd.concat([corr, corr_col[target_column]], axis=0)

    corr.to_csv('corr.csv',index=True)
    return

    y      = train[target_column]
    X      = train.drop(drop_columns, axis=1)
    y_cv   = cv[target_column]
    X_cv   = cv.drop(drop_columns, axis=1)
    X_test = test.drop(drop_columns, axis=1)
    
    print('Train model')
    xgtrain   = xgb.DMatrix(X, y)
    xgval     = xgb.DMatrix(X_cv, y_cv)
    xgtest    = xgb.DMatrix(X_test)

    params = {
        'objective':'binary:logistic',
        'eval_metric':'logloss',
        'eta' :             0.05,
        'min_child_weight': 1,
        'subsample':        0.9,
        'colsample_bytree': 0.9,
        'max_depth':        6,
        'silent':           0,
    }

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(list(params.items()), xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    
    print('Make prediction')
    y_pred = model.predict(xgtest)

    submit = pd.read_csv('../input/sample_submission.csv')
    submit['PredictedProb'] = np.clip(y_pred, 0.01, 0.99)
    submit.to_csv('xgb.csv',index=False)

calc()