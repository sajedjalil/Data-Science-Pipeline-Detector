# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
import gc
# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

def merge_data():
    bur_bal = pd.read_csv('../input/bureau_balance.csv')
    print('bureau_balance shape:', bur_bal.shape)
    #bur_bal.head()
    bur_bal = pd.concat([bur_bal, pd.get_dummies(bur_bal.STATUS, prefix='bur_bal_status')],
                       axis=1).drop('STATUS', axis=1)
    bur_cnts = bur_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    bur_bal['bur_cnt'] = bur_bal['SK_ID_BUREAU'].map(bur_cnts['MONTHS_BALANCE'])
    avg_bur_bal = bur_bal.groupby('SK_ID_BUREAU').mean()
    avg_bur_bal.columns = ['bur_bal_' + f_ for f_ in avg_bur_bal.columns]
    del bur_bal
    gc.collect()

    bur = pd.read_csv('../input/bureau.csv')
    print('bureau shape:', bur.shape)
    #bur.head()
    bur_credit_active_dum = pd.get_dummies(bur.CREDIT_ACTIVE, prefix='ca')
    bur_credit_currency_dum = pd.get_dummies(bur.CREDIT_CURRENCY, prefix='cc')
    bur_credit_type_dum = pd.get_dummies(bur.CREDIT_TYPE, prefix='ct')

    bur_full = pd.concat([bur, bur_credit_active_dum, bur_credit_currency_dum, bur_credit_type_dum], axis=1).drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'], axis=1)
    del bur_credit_active_dum, bur_credit_currency_dum, bur_credit_type_dum
    gc.collect()
    bur_full = bur_full.merge(right=avg_bur_bal.reset_index(), how='left', on='SK_ID_BUREAU',suffixes=('', '_bur_bal'))
    nb_bureau_per_curr = bur_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    bur_full['SK_ID_BUREAU'] = bur_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
    avg_bur = bur_full.groupby('SK_ID_CURR').mean()
    avg_bur.columns = ['bur_' + f_ for f_ in avg_bur.columns]
    del bur, bur_full, avg_bur_bal
    gc.collect()

    prev = pd.read_csv('../input/previous_application.csv')
    print('previous_application shape:', prev.shape)
    #prev.head()
    prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_)], axis=1)
    prev = pd.concat([prev, prev_dum],axis=1)
    del prev_dum
    gc.collect()
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    avg_prev.columns = ['prev_' + f_ for f_ in avg_prev.columns]
    del prev
    gc.collect()

    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    print('pos_cash_balance shape:', pos.shape)
    #pos.head()
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'], prefix='ncs')], axis=1)
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_pos = pos.groupby('SK_ID_CURR').mean()
    avg_pos.columns = ['pos_' + f_ for f_ in avg_pos.columns]
    del pos, nb_prevs
    gc.collect()

    cc_bal = pd.read_csv('../input/credit_card_balance.csv')
    print('credit_card_balance shape:', cc_bal.shape)
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='ncs')], axis=1)
    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    del cc_bal, nb_prevs
    gc.collect()

    inst = pd.read_csv('../input/installments_payments.csv')
    print('installment_payment shape:', inst.shape)
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    del inst, nb_prevs
    gc.collect()

    train = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    print('train shape:', train.shape)
    print('test shape:', test.shape)
    y = train['TARGET']
    del train['TARGET']
    cat_feats = [f_ for f_ in train.columns if train[f_].dtype == 'object']
    for f_ in cat_feats:
        train[f_], indexer = pd.factorize(train[f_])#类似于类似于类似于label encoder
        test[f_] = indexer.get_indexer(test[f_])
    train = train.merge(right = avg_bur.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_bur.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    del avg_bur, avg_prev, avg_pos, avg_cc_bal, avg_inst
    gc.collect()
    return train, test, y
    
def train_model(train_, test_, y_, folds_):
    train_ = pd.DataFrame()
    test_ = pd.DataFrame()
    oof_preds = np.zeros(train_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    feats = [f_ for f_ in train.columns if f_ not in ['SK_ID_CURR']]
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(train_)):
        trn_x, trn_y = train_[[feats]].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = train_[[feats]].iloc[val_idx], y_.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators = 4000,
            learning_rate = 0.03,
            num_leaves = 30,
            colsample_bytree = .8,
            subsample = .9,
            max_depth = 7,
            reg_alpha = .1,
            min_split_gain = .01,
            min_child_weight = 2,
            silent = -1,
            verbose = -1
            )
        clf.fit(trn_x, trn_y, 
            eval_set = [(trn_x, trn_y), (val_x, val_y)],
            eval_metric = 'auc', verbose = 100, early_stopping_rounds = 100)
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration = clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = feats
        fold_importance_df['importance'] = clf.feature_importances_
        fold_importance_df['fold'] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('fold %2d AUC %.6f'%(n_fold+1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    print('full AUC score %.6f'%roc_auc_score(y, oof_preds))
    test_['TARGET'] = sub_preds
    
    return oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df
    
def display_importance(feature_importance_df_, num):
    cols = feature_importance_df_[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:num].index
    best_features = feature_importance_df_[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8, 10))
    sns.barplot(x='importance', y='feature',
        data = best_features.sort_values(by = 'importance', ascending=False))
    plt.title('LightGBM Feature(average over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    
    return cols
    
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
def feat_ext_source(df):
    x1 = df['EXT_SOURCE_1'].fillna(-1) + 1e-1
    x2 = df['EXT_SOURCE_2'].fillna(-1) + 1e-1
    x3 = df['EXT_SOURCE_3'].fillna(-1) + 1e-1
    
    df['EXT_SOURCE_1over2_NAminus1_Add0.1'] = x1/x2
    df['EXT_SOURCE_2over1_NAminus1_Add0.1'] = x2/x1
    df['EXT_SOURCE_1over3_NAminus1_Add0.1'] = x1/x3
    df['EXT_SOURCE_3over1_NAminus1_Add0.1'] = x3/x1
    df['EXT_SOURCE_2over3_NAminus1_Add0.1'] = x2/x3
    df['EXT_SOURCE_3over2_NAminus1_Add0.1'] = x3/x2
    df['EXT_SOURCE_1_log'] = np.log(df['EXT_SOURCE_1'] + 1)
    df['EXT_SOURCE_2_log'] = np.log(df['EXT_SOURCE_2'] + 1)
    df['EXT_SOURCE_3_log'] = np.log(df['EXT_SOURCE_3'] + 1) 
    return df

    
gc.enable()
train, test, y = merge_data()
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train = feat_ext_source(train)
test = feat_ext_source(test)
#train.to_csv('train.csv', index=False)
#test.to_csv('test.csv', index=False)
#train.isnull().sum()
train = train.fillna(-1)
#print(train.isnull().sum())
test = test.fillna(-1)
sm = SMOTE(random_state=42, kind='borderline2')
train, y = sm.fit_sample(train, y)
folds = KFold(n_splits=5, shuffle=True, random_state=0)
oof_preds, test_preds, importances = train_model(train, test, y, folds)
test_preds.to_csv('submission2.csv', index=False)
