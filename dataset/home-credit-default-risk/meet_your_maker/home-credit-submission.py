# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import gc
gc.enable()
import os
#print(os.listdir("../input"))   
#train = pd.read_csv("../input/application_train.csv")
#test = pd.read_csv("../input/application_test.csv")
#y = train["TARGET"]
#del train["TARGET"]

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler#, MinMaxScaler
from tqdm import tqdm

def transform_train_test():
    train = pd.read_csv("../input/application_train.csv")
    test = pd.read_csv("../input/application_test.csv")
    sk_id = train["SK_ID_CURR"]
    train = train.drop("SK_ID_CURR", axis=1)
    sk_id_test = test["SK_ID_CURR"]
    test = test.drop("SK_ID_CURR", axis=1)
    train_annuity = train["AMT_ANNUITY"]
    test_annuity = test["AMT_ANNUITY"]
    y = train["TARGET"]
    corr = train.corr()
    corr = corr["TARGET"].sort_values(ascending=False)
    corr = pd.DataFrame(corr)
    c = corr.loc[(corr["TARGET"] >= 0.03) | (corr["TARGET"] <= -0.03)]
    cols_use = list(c.index)
    cols_use.remove("TARGET")
    train = train[cols_use]
    test = test[cols_use]
    train['NEW_SOURCES_PROD'] = train['EXT_SOURCE_1'] * train['EXT_SOURCE_2'] * train['EXT_SOURCE_3']
    train['NEW_EXT_SOURCES_MEAN'] = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    train['NEW_SCORES_STD'] = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    test['NEW_SOURCES_PROD'] = test['EXT_SOURCE_1'] * test['EXT_SOURCE_2'] * test['EXT_SOURCE_3']
    test['NEW_EXT_SOURCES_MEAN'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    test['NEW_SCORES_STD'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    train["AMT_ANNUITY"] = train_annuity 
    test["AMT_ANNUITY"] = test_annuity
    train['PAYMENT_RATE'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
    test['PAYMENT_RATE'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']
    train['DAYS_EMPLOYED'].replace(365243, 0, inplace= True)
    test['DAYS_EMPLOYED'].replace(365243, 0, inplace= True)
    cols_use.append("AMT_ANNUITY")
    cols_use.append("PAYMENT_RATE")
    train = train.fillna(0)
    test = test.fillna(0)
    sc = StandardScaler()
    numeric = train.select_dtypes(include=['int64', 'float64'])
    numeric_fill = numeric.mean()
    
    numeric = numeric.fillna(numeric_fill)
    
    train[numeric.columns] = numeric
    test[numeric.columns] = test[numeric.columns].fillna(numeric_fill)
    train = pd.concat(
        [train, pd.DataFrame(
            sc.fit_transform(numeric),
            columns=['sc_{}'.format(i) for i in numeric.columns],
            index=train.index
        )], axis=1)
    test = pd.concat(
        [test, pd.DataFrame(
            sc.transform(test[numeric.columns].fillna(numeric_fill)),
            columns=['sc_{}'.format(i) for i in numeric.columns],
            index=test.index
        )], axis=1)
        
    train["SK_ID_CURR"] = sk_id
    test["SK_ID_CURR"] = sk_id_test
    train = train.drop(cols_use, axis=1)
    test = test.drop(cols_use, axis=1)
    train.columns = [f_.upper() for f_ in train.columns]
    test.columns = [f_.upper() for f_ in test.columns]
    del corr, c, cols_use, sk_id, sk_id_test, train_annuity, test_annuity, numeric_fill, numeric
    gc.collect()
    return train, test, y
def transform_cats_and_nums(df):
    '''
    cleaning of other files, copied technique from @AivinSolatorio https://github.com/avsolatorio
    '''
    df = df.copy()
    sk_id = df["SK_ID_CURR"]
    df = df.drop("SK_ID_CURR", axis=1)
    cols = set(df.columns)
    cat_cols = []
    obj = [ob for ob in df.columns if df[ob].dtype == 'object']
    numeric = df.select_dtypes(include=['int64','float64'])
    numeric_fill = 0
    
    numeric = numeric.fillna(numeric_fill)
    
    #df[numeric.columns] = numeric
    sc = StandardScaler()
    #mx = MinMaxScaler()

    df = pd.concat(
        [df, pd.DataFrame(
            sc.fit_transform(numeric),
            columns=['sc_{}'.format(i) for i in numeric.columns],
            index=df.index
        )], axis=1)

    #df = pd.concat(
        #[df, pd.DataFrame(
           # mx.fit_transform(numeric),
            #columns=['mx_{}'.format(i) for i in numeric.columns],
           #index=df.index
       # )], axis=1)
    for col in cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('N/A')
            df[col] = df[col].apply(str)

            le = LabelEncoder()
            ohe = OneHotEncoder()

            df_vals = list(df[col].unique())

            le.fit(df_vals)
            df[col] = le.transform(df[col])
            cat_cols.append(col)

    if obj != []:        

        df_ohe = pd.get_dummies(df[cat_cols].astype(str), dummy_na=True)
        df = pd.concat([df, df_ohe], axis=1)
        del df_ohe
        gc.collect()

    df = df.drop(cols, axis=1)

    df["SK_ID_CURR"] = sk_id
    df.columns = [f_.upper() for f_ in df.columns]
    del numeric, numeric_fill, sk_id, obj, cat_cols, cols
    gc.collect()
    return df


def transform_bb():
    '''
    cleaning bureau_bal.csv
    '''
    buro_bal = pd.read_csv('../input/bureau_balance.csv')
    #bureau = pd.read_csv('bureau.csv')
    buro_bal["STATUS"] = buro_bal["STATUS"].replace([str(i) for i in range(1,5)], "dpd")
    buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)
    buro_bal['buro_count'] = buro_bal.reset_index(0).groupby('SK_ID_BUREAU').count().max(axis=1)
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    bb_agg = buro_bal.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    buro_bal = pd.concat([buro_bal, bb_agg], axis=1)
    buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()
    buro_bal.columns = [f_.upper() for f_ in buro_bal.columns]
    del bb_agg, bb_aggregations
    gc.collect()
    #bureau = bureau.join(bb, how='left', on='SK_ID_BUREAU')
    return buro_bal
    

def transform_dataset():
    print("transforming buro")
    bb = transform_bb()
    buro = pd.read_csv('../input/bureau.csv')
    buro_id = buro["SK_ID_BUREAU"]
    buro = buro.drop("SK_ID_BUREAU", axis=1)
    #uro = transform_cats_and_nums(buro)
    buro["SK_ID_BUREAU"] = buro_id
    buro = buro.join(bb, how='left', on='SK_ID_BUREAU')
    buro = buro.drop("SK_ID_BUREAU", axis=1)
    buro_avg = buro.groupby("SK_ID_CURR").mean()
   #buro_avg.columns = pd.Index(['BURO_'+ e[0] + "_" + e[1].upper() for e in buro_avg.columns.tolist()])
    del buro_id, bb, buro
    gc.collect()
    print("done! for buro... transforming prevs")
    prevs = pd.read_csv("../input/previous_application.csv")
    prevs['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prevs['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prevs['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prevs['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prevs['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    #prevs = prevs.dropna(thresh=0.75*len(prevs), axis=1)
    nb_prevs = prevs[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prevs['SK_ID_PREV'] = prevs['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    prevs = transform_cats_and_nums(prevs)
    prevs_avg = prevs.groupby("SK_ID_CURR").mean()
    #revs_avg.columns = pd.Index(['PREVS_' + e[0] + "_" + e[1].upper() for e in prevs_avg.columns.tolist()])
    #prevs_avg = prevs.groupby("SK_ID_CURR").mean()
    del prevs, nb_prevs
    gc.collect()
    print("done! for prevs... transforming pos")
    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    #pos = pos.dropna(thresh=0.75*len(pos), axis=1)
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    pos = transform_cats_and_nums(pos)
    pos_avg = pos.groupby("SK_ID_CURR").mean()
    #os_avg.columns = pd.Index(['POS_'+ e[0] + "_" + e[1].upper() for e in pos_avg.columns.tolist()])
    del nb_prevs, pos
    gc.collect()
    print("done! for pos... transforming credit card")
    cc_bal = pd.read_csv('../input/credit_card_balance.csv')
    #cc_bal = cc_bal.dropna(thresh=0.75*len(cc_bal), axis=1)
    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    cc_bal = transform_cats_and_nums(cc_bal)
    cc_bal_avg = cc_bal.groupby("SK_ID_CURR").mean()
   #cc_bal_avg.columns = pd.Index(['CC_BAL_'+ e[0] + "_" + e[1].upper() for e in cc_bal_avg.columns.tolist()])
    del nb_prevs, cc_bal
    gc.collect()
    print("done! for pos... transforming installments")
    install = pd.read_csv('../input/installments_payments.csv')
    #install = install.dropna(thresh=0.75*len(install), axis=1)
    nb_prevs = install[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    install['SK_ID_PREV'] = install['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    install_avg = install.groupby("SK_ID_CURR").mean()
   #install_avg.columns = pd.Index(['INSTALL_'+ e[0] + "_" + e[1].upper() for e in install_avg.columns.tolist()])
    del nb_prevs, install
    gc.collect()
    print("done! for installation... transforming train and test set")
    train, test, y = transform_train_test()
    train = train.merge(right=buro_avg.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=buro_avg.reset_index(), how='left', on='SK_ID_CURR')
    
    train = train.merge(right=prevs_avg.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=prevs_avg.reset_index(), how='left', on='SK_ID_CURR')
    
    train = train.merge(right=pos_avg.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=pos_avg.reset_index(), how='left', on='SK_ID_CURR')
    
    train = train.merge(right=cc_bal_avg.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=cc_bal_avg.reset_index(), how='left', on='SK_ID_CURR')
    
    train = train.merge(right=install_avg.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=install_avg.reset_index(), how='left', on='SK_ID_CURR')
    del install_avg, buro_avg, prevs_avg, cc_bal_avg
    gc.collect()
    train = train.loc[:,~train.columns.duplicated()]
    test = test.loc[:,~test.columns.duplicated()]

    print("done transforming dataset")
    return train, test, y


def train_model(data_, test_, y_, folds_):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    #feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.05,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=200  #30
               )
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
    
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
    
    test_['TARGET'] = sub_preds
    
    return oof_preds, test_[['SK_ID_CURR', 'TARGET']]

  
if __name__ == '__main__':
    gc.enable()
    
    train, test, y = transform_dataset()
    
    folds = KFold(n_splits=5, shuffle=True, random_state=2018) 
    
    oof_preds, test_preds = train_model(train, test, y, folds)
    
    test_preds.to_csv("base_lgbm_with_nan_sub1.csv", index=False)
    
    
    
    
    
    
    
    


    
# Any results you write to the current directory are saved as output.   