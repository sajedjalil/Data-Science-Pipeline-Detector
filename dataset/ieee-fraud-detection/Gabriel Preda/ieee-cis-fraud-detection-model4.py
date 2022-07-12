import numpy as np 
import pandas as pd
import os
import re
import warnings
import time
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
np.random.seed(2019)

# ref: https://www.kaggle.com/duykhanh99/lightgbm-starter-with-r-0-9493-lb
# ref: https://www.kaggle.com/kyakovlev/ieee-simple-lgbm
# ref: https://www.kaggle.com/tolgahancepel/lightgbm-single-model-and-feature-engineering

def load_data():
    start_time = time.time()
    print('load data')
    train_identity_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'train_identity.csv'))
    train_transaction_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'train_transaction.csv'))
    test_identity_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'test_identity.csv'))
    test_transaction_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'test_transaction.csv'))
    print(f"Completed load data:  time: {format(round(time.time() - start_time,2))} sec.\n") 
    return train_identity_df, train_transaction_df, test_identity_df, test_transaction_df

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """            
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def merge_and_reduce_memory(train_identity_df, train_transaction_df, test_identity_df, test_transaction_df):
    start_time = time.time()
    print('reduce memory usage - train')
    identity_cols = list(train_identity_df.columns.values)
    transaction_cols = list(train_transaction_df.drop('isFraud', axis=1).columns.values)

    X_train = pd.merge(train_transaction_df[transaction_cols + ['isFraud']], train_identity_df[identity_cols], how='left')
    X_train = reduce_mem_usage(X_train)
    print('reduce memory usage - test')
    X_test = pd.merge(test_transaction_df[transaction_cols], train_identity_df[identity_cols], how='left')
    X_test = reduce_mem_usage(X_test)

    X_train_id = X_train.pop('TransactionID')
    X_test_id = X_test.pop('TransactionID')
    del train_identity_df,train_transaction_df, test_identity_df, test_transaction_df

    print(f"Completed merge and reduce memory usage:  time: {format(round(time.time() - start_time,2))} sec.\n") 
    
    return X_train, X_test, X_train_id, X_test_id

    
def mail_domain(data, new_p_column, new_r_column, domain_sub_set):
    data[new_p_column] = 0
    data.loc[data['P_emaildomain'].isin(domain_sub_set), new_p_column] = 1
    data[new_r_column] = 0
    data.loc[data['R_emaildomain'].isin(domain_sub_set), new_r_column] = 1
    return data

def feature_engineering(X_train, X_test, X_train_id, X_test_id):
    start_time = time.time()
    print('feature engineering')
    all_data = X_train.append(X_test, sort=False).reset_index(drop=True)

       
        
    vcols = [f'V{i}' for i in range(1,340)]
    sc = preprocessing.MinMaxScaler()
    pca = PCA(n_components=3) #0.99
    vcol_pca = pca.fit_transform(sc.fit_transform(all_data[vcols].fillna(-1)))
    all_data['_vcol_pca0'] = vcol_pca[:,0]
    all_data['_vcol_pca1'] = vcol_pca[:,1]
    all_data['_vcol_pca2'] = vcol_pca[:,2]    
    all_data['_vcol_nulls'] = all_data[vcols].isnull().sum(axis=1)

    vcols1 = [f'V{i}' for i in range(1,12)]
    vcols2 = [f'V{i}' for i in range(12,35)]
    vcols3 = [f'V{i}' for i in range(35,53)]
    vcols4 = [f'V{i}' for i in range(53,75)]
    vcols5 = [f'V{i}' for i in range(75,95)]
    vcols6 = [f'V{i}' for i in range(95,138)]
    vcols7 = [f'V{i}' for i in range(138,167)]
    vcols8 = [f'V{i}' for i in range(167,217)]
    vcols9 = [f'V{i}' for i in range(217,279)]
    vcols10 = [f'V{i}' for i in range(279,322)]
    vcols11 = [f'V{i}' for i in range(322,340)]

    i = 1
    for _vcols in [vcols1, vcols2, vcols3, vcols4, vcols5, vcols6, vcols7, vcols8, vcols9, vcols10, vcols11]:
        sc = preprocessing.MinMaxScaler()
        pca = PCA(n_components=3)
        vcol_pca = pca.fit_transform(sc.fit_transform(all_data[_vcols].fillna(-1)))
        all_data[f'_vcol_{i}_pca0'] = vcol_pca[:,0]
        all_data[f'_vcol_{i}_pca1'] = vcol_pca[:,1]
        all_data[f'_vcol_{i}_pca2'] = vcol_pca[:,2]    
        all_data['f_vcol_{i}_nulls'] = all_data[_vcols].isnull().sum(axis=1)
        i = i + 1
    
    all_data.drop(vcols, axis=1, inplace=True)
  
    
    all_data['_P_emaildomain__addr1'] = all_data['P_emaildomain'] + '__' + all_data['addr1'].astype(str)
    all_data['_P_emaildomain__C2'] = all_data['P_emaildomain'] + '__' + all_data['C2'].astype(str)
    all_data['_R_emaildomain__addr1'] = all_data['R_emaildomain'] + '__' + all_data['addr1'].astype(str)
    all_data['_card1__card2'] = all_data['card1'].astype(str) + '__' + all_data['card2'].astype(str)
    all_data['_card1__addr1'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str)
    all_data['_card2__addr1'] = all_data['card2'].astype(str) + '__' + all_data['addr1'].astype(str)
    all_data['_card1__addr2'] = all_data['card1'].astype(str) + '__' + all_data['addr2'].astype(str)
    all_data['_card2__addr2'] = all_data['card2'].astype(str) + '__' + all_data['addr2'].astype(str)
    all_data['_card1__dist1'] = all_data['card1'].astype(str) + '__' + all_data['dist1'].astype(str)
    all_data['_card2__dist2'] = all_data['card2'].astype(str) + '__' + all_data['dist2'].astype(str)
    all_data['_card1__card5'] = all_data['card1'].astype(str) + '__' + all_data['card5'].astype(str)
    all_data['_card12__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)
    all_data['_card12__addr2'] = all_data['_card1__card2'] + '__' + all_data['addr2'].astype(str)
    all_data['_card3__card4'] = all_data['card3'].astype(str) + '__' + all_data['card4'].astype(str)
    all_data['_card4__card5'] = all_data['card4'].astype(str) + '__' + all_data['card5'].astype(str)
    all_data['_card5__card6'] = all_data['card5'].astype(str) + '__' + all_data['card6'].astype(str)
    all_data['_card34__addr1'] = all_data['_card3__card4'] + '__' + all_data['addr1'].astype(str)
    all_data['_card34__addr2'] = all_data['_card3__card4'] + '__' + all_data['addr2'].astype(str)
    all_data['_card45__addr1'] = all_data['_card4__card5'] + '__' + all_data['addr1'].astype(str)
    all_data['_card45__addr2'] = all_data['_card4__card5'] + '__' + all_data['addr2'].astype(str)
    all_data['_card56__addr1'] = all_data['_card5__card6'] + '__' + all_data['addr1'].astype(str)
    all_data['_card56__addr2'] = all_data['_card5__card6'] + '__' + all_data['addr2'].astype(str)
    all_data['_P_emaildomain__card5'] = all_data['P_emaildomain'] + '__' + all_data['card5'].astype(str)
    
    all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)
    all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))
    all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))
    
    cols = ['ProductCD','card1','card2', 'card3', 'card4','card5','card6','P_emaildomain', 'R_emaildomain', '_card1__dist1', '_card2__dist2', \
            '_card12__addr1', '_card12__addr2','_P_emaildomain__addr1', '_P_emaildomain__card5', '_P_emaildomain__C2', 'id_02', 'D15', \
           "C13","C1","C14"]
    
    for f in cols:
        all_data[f'_amount_mean_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('mean')
        all_data[f'_amount_median_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('median')
        all_data[f'_amount_std_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('std')
        all_data[f'_amount_pct_{f}'] = (all_data['TransactionAmt'] - all_data[f'_amount_mean_{f}']) / all_data[f'_amount_std_{f}']

    for f in cols:
        vc = all_data[f].value_counts(dropna=False)
        all_data[f'_count_{f}'] = all_data[f].map(vc)

    yahoo = ["yahoo.fr", "yahoo.de", "yahoo.es", "yahoo.co.uk", "yahoo.com", "yahoo.com.mx", "ymail.com", "rocketmail.com", "frontiernet.net"]  
    all_data = mail_domain(all_data,'P_isyahoo', 'D_isyahoo', yahoo)

    microsoft = ["hotmail.com", "live.com.mx", "live.com", "msn.com", "hotmail.es", "outlook.es", "hotmail.fr", "hotmail.de", "hotmail.co.uk"]
    all_data = mail_domain(all_data, "P_ismfst", "R_ismfst", microsoft)

    mac = ["icloud.com", "mac.com", "me.com"]
    all_data = mail_domain(all_data, "P_ismac", "R_ismac", mac)

    att = ["prodigy.net.mx", "att.net", "sbxglobal.net"]
    all_data = mail_domain(all_data, "P_isatt", "R_isatt", att)

    centurylink = ["centurylink.net", "embarqmail.com", "q.com"]
    all_data = mail_domain(all_data, "P_iscenturylink", "R_iscenturylink", centurylink)

    aol =["aim.com", "aol.com"]
    all_data = mail_domain(all_data,"P_isaol", "R_isaol", aol)

    spectrum =["twc.com", "charter.com"]
    all_data = mail_domain(all_data, "P_isspectrum", "R_isspectrum", spectrum)

    proton =["protonmail.com"]
    all_data = mail_domain(all_data, "P_isproton", "R_isproton", proton)

    scomcast = ["comcast.net"]
    all_data = mail_domain(all_data, "P_iscomcast", "R_iscomcast",scomcast)

    google = ["gmail.com"]
    all_data = mail_domain(all_data,  "P_isgoogle", "R_isgoogle", google)

    anonymous = ["anonymous.com"]
    all_data = mail_domain(all_data,   "P_isanon", "R_isanon", anonymous)

    all_data['P_isNA'] = 0
    all_data.loc[all_data['P_emaildomain'].isna(), 'P_isNA'] = 1
    all_data['R_isNA'] = 0
    all_data.loc[all_data['R_emaildomain'].isna(), 'R_isNA'] = 1
         
    cat_cols = [f'id_{i}' for i in range(12,39)]
    for i in cat_cols:
        if i in all_data.columns:
            all_data[i] = all_data[i].astype(str)
            all_data[i].fillna('unknown', inplace=True)

    enc_cols = []
    for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():
        if t == object:
            enc_cols.append(i)
            all_data[i] = pd.factorize(all_data[i])[0]

    X_train = all_data[all_data['isFraud'].notnull()]
    X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)
    Y_train = X_train.pop('isFraud')
    del all_data
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Completed feature engineering:  time: {format(round(time.time() - start_time,2))} sec.\n") 

    return X_train, X_test, Y_train
    

def run_model(X_train, X_test, Y_train):
    start_time = time.time()
    print('run model')
    params={'learning_rate': 0.005,
        'objective': 'binary',
        'metric': 'auc',
        'num_threads': -1,
        'num_leaves': 256,
        'verbose': 1,
        'random_state': 42,
        'bagging_fraction': 0.75,
        'feature_fraction': 0.75
       }
    
    folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=2319)
    oof = np.zeros(X_train.shape[0])
    predictions = np.zeros(X_test.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, Y_train.values)):
        fold_time = time.time()
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=Y_train.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx], label=Y_train.iloc[val_idx])
        clf = lgb.train(params, trn_data, 2500, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
        oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)
        predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        print(f"Completed current fold:  time: {format(round(time.time() - fold_time,2))} sec.\n") 

    print("CV score: {:<8.5f}".format(roc_auc_score(Y_train, oof)))
    
    print(f"Completed train and prediction:  time: {format(round(time.time() - start_time,2))} sec.\n") 
    return predictions

def submission(X_test_id, predictions):
    print('submission')
    submission = pd.DataFrame()
    submission['TransactionID'] = X_test_id
    submission['isFraud'] = predictions
    submission.to_csv('submission.csv', index=False)    
 
def main():
    train_identity_df,train_transaction_df, test_identity_df, test_transaction_df = load_data()
    X_train, X_test, X_train_id, X_test_id = merge_and_reduce_memory(train_identity_df, train_transaction_df, test_identity_df, test_transaction_df)
    X_train, X_test, Y_train = feature_engineering(X_train, X_test, X_train_id, X_test_id)
    predictions = run_model(X_train, X_test, Y_train)
    submission(X_test_id, predictions)
    
if __name__ == '__main__':
    main()



