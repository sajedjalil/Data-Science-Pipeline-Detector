#modified from
#https://www.kaggle.com/cttsai/good-fun-with-ligthgbm-meanenc-more-features
#https://www.kaggle.com/yekenot/catboostarter

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import gc
gc.enable()

plt.switch_backend('agg')

def PivotGroupBy(df, groupby_id, target_id, feature_name='', cutoff=0.05):
    print('Pivot and Group on: {0}, {1}'.format(target_id, df.shape))
    cnt_name = 'cnt_{0}'.format(target_id)
    tmp = df.groupby([groupby_id])[target_id].value_counts(normalize=True)
    if len(tmp) > 2:
        tmp = tmp.loc[tmp >= cutoff].rename(cnt_name).reset_index()
    else:
        tmp = tmp.iloc[:1].rename(cnt_name).reset_index()
    tmp = tmp.pivot(index=groupby_id, columns=target_id, values=cnt_name)
    tmp.rename(columns={f:'{0}_r={1}'.format(feature_name, f) for f in tmp.columns}, inplace=True)
    return tmp

def LabelMeanEnc(X, y, test_X, encode_feats):    
    for f_ in encode_feats:
        df = pd.concat([y, X[f_]], axis=1).groupby(f_).agg(['mean'])
        df = df.squeeze().rename('mean_{}'.format(f_))
        print(f_, df.shape)
        X = X.merge(df.to_frame(), how='left', left_on=f_, right_index=True)
        test_X = test_X.merge(df.to_frame(), how='left', left_on=f_, right_index=True)
        del df; gc.collect()
        
    return X, test_X

def CatMeanEnc(df, index_name, groupby_ids):

    cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    not_cat_features = [f for f in df.columns if f not in cat_features]
    
    #pivot on categorcals
    list_df_pivots = [PivotGroupBy(df, index_name, f_, feature_name=f_) for f_ in cat_features]
    df_pivots = pd.concat(list_df_pivots, axis=1)
    del list_df_pivots; gc.collect()
    df_pivots['na_cnt_{0}'.format(index_name)] = pd.isnull(df_pivots).apply(lambda x: sum(x.astype(int)), axis=1)
    #print(df_pivots.head())
    
    #miao on
    df_miao = df[not_cat_features].groupby(groupby_ids).agg({k:["sum", "mean", "max", "min", "std"] for k in not_cat_features})
    df_miao.columns = pd.Index(["{0}_{1}".format(e[0], e[1]) for e in df_miao.columns.tolist()])
    df_ret = df_miao.join(df_pivots, how='left')
    #print(df_miao.head())

    del df_miao, df_pivots; gc.collect()
    
    #print(df_ret.describe())
    df_ret['cnt_{:}'.format(index_name)] = df[[groupby_ids, index_name]].groupby(groupby_ids).count()[index_name]
    if index_name in df_ret.columns:
        del df_ret[index_name]
        print('del index {0}'.format(index_name))
    return df_ret
    
def JoinMeanEnc(main_df, join_dfs=[]):
    for df in join_dfs:
        print(main_df.shape, df.shape)
        f_join = [f_ for f_ in df.columns if f_ not in main_df.columns]
        main_df = main_df.join(df[f_join], how='left')
        print('concat {0} to {1}'.format(df.shape, main_df.shape))
    return main_df

def OOFPreds(X, y, test_X, params, n_splits=5, random_state=42, clf='lgb'):
    
    feature_importance = []
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds, sub_preds = np.zeros(X.shape[0]), np.zeros(test_X.shape[0])
    print(X.shape, test_X.shape)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data, y)):
        trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]
        trn_init_score = pd.Series([0.95] * len(trn_x), index=trn_x.index) 
        val_init_score = pd.Series([0.95] * len(val_x), index=val_x.index)
        if clf == 'lgb':
            gbm = CatBoostClassifier(iterations=1000,
                              learning_rate=0.1,
                              depth=6,
                              l2_leaf_reg=40,
                              bootstrap_type='Bernoulli',
                              subsample=0.7,
                              scale_pos_weight=5,
                              eval_metric='AUC',
                              metric_period=50,
                              od_type='Iter',
                              od_wait=45,
                              random_seed=17,
                              allow_writing_files=False)  
            gbm.fit(trn_x, trn_y, eval_set=(val_x, val_y),use_best_model=True,verbose=True)
            pred_val  = gbm.predict_proba(val_x)[:, 1]
            pred_test = gbm.predict_proba(test_X)[:, 1]

        oof_preds[val_idx] = pred_val
        sub_preds += pred_test / folds.n_splits
        print('Fold {:02d} AUC: {:.6f}'.format(n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del trn_x, trn_y, val_x, val_y; gc.collect()

        feature_importance.append(pd.DataFrame({
                'importance': gbm.feature_importances_,
                'fold': [n_fold + 1] * X.shape[1],
                'feature': X.columns.tolist()},).set_index('feature'))
        
    oof_preds = pd.Series(oof_preds.flatten(), index=X.index).rename('TARGET')
    sub_preds = pd.Series(sub_preds.flatten(), index=test_X.index).rename('TARGET')
    return oof_preds, sub_preds, feature_importance


LoanID = 'SK_ID_CURR'
data   = pd.read_csv('../input/application_train.csv').set_index(LoanID)
test   = pd.read_csv('../input/application_test.csv').set_index(LoanID)
prev   = pd.read_csv('../input/previous_application.csv')
buro   = pd.read_csv('../input/bureau.csv')
burobl = pd.read_csv('../input/bureau_balance.csv')
credit = pd.read_csv('../input/credit_card_balance.csv')
print(data.shape, test.shape)

# Attach bureau_balance to Bureau
tmp = PivotGroupBy(burobl, 'SK_ID_BUREAU', 'STATUS', feature_name='bureau_balance')
tmp['LONGEST_MONTHS'] = burobl.groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].size()
buro = buro.join(tmp, how='left', on='SK_ID_BUREAU')
print(buro.head())
del burobl, tmp; gc.collect()

# factorize         
categorical_feats = [f for f in data.columns if data[f].dtype == 'object']
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])

y = data['TARGET']
del data['TARGET']

print('\nCat Mean Encode')
avg_dfs = [CatMeanEnc(prev,   index_name='SK_ID_PREV', groupby_ids=LoanID), 
           CatMeanEnc(buro,   index_name='SK_ID_BUREAU', groupby_ids=LoanID), 
           CatMeanEnc(credit, index_name='SK_ID_PREV', groupby_ids=LoanID)]
data = JoinMeanEnc(data, join_dfs=avg_dfs)
test = JoinMeanEnc(test, join_dfs=avg_dfs)

print('\nLabel Mean Encode')
data, test = LabelMeanEnc(data, y, test, encode_feats=categorical_feats)

f_to_clean = [f_ for f_ in data.columns if len(data[f_].value_counts()) < 2]
f_to_clean.extend([f_ for f_ in test.columns if len(test[f_].value_counts()) < 2])       
f_to_clean = sorted(list(set(f_to_clean)))
print(data.shape[1], len(f_to_clean))

excluded_feats = [] + f_to_clean #['SK_ID_CURR']
features = [f_ for f_ in data.columns if f_ not in excluded_feats]

params_LGBM = {
    'n_estimators': 4000,
	'learning_rate': 0.02,
    'num_leaves': 31,
    'colsample_bytree': .8,
    'subsample': .8,
    'subsample_freq': 5,
    'max_depth': 16,
    'reg_alpha': .001,
    'reg_lambda': .1,
    'min_split_gain': .01,
    'silent': True,
    }

oof_preds, sub_preds, feature_importance_dfs = OOFPreds(X=data[features].fillna(-999), y=y, 
                                                       test_X=test[features].fillna(-999), 
                                                       params=params_LGBM, 
                                                       n_splits=5, 
                                                       random_state=42)

score = roc_auc_score(y, oof_preds)
print('Full AUC score {:.6f}'.format(score))

subm = sub_preds.to_frame()
subm.to_csv('subm_cat_auc{:.8f}.csv'.format(score), index=True, float_format='%.8f')