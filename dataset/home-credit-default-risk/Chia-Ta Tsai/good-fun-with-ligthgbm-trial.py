import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

import gc
gc.enable()

def CatMeanEnc(df, index_name, groupby_ids):
###################################
# PLEASE DON'T DO THIS AT HOME LOL
# Averaging factorized categorical features defeats my own reasoning
################################### 
    cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    for f_ in cat_features:
        df[f_], _ = pd.factorize(df[f_])    
    df_ret = df.groupby(LoanID).mean()
    print(df_ret.head())
    df_ret['cnt_{:}'.format(index_name)] = df[[groupby_ids, index_name]].groupby(groupby_ids).count()[index_name]
    del df_ret[index_name]
    return df_ret

def JoinMeanEnc(main_df, join_dfs=[]):
    for df in join_dfs:
        print(main_df.shape, df.shape)
        f_join = [f_ for f_ in df.columns if f_ not in main_df.columns]
        main_df = main_df.join(df[f_join], how='left')
    return main_df

def OOFPreds(X, y, test_X, params, n_splits=5, random_state=42):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds, sub_preds = np.zeros(X.shape[0]), np.zeros(test_X.shape[0])

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data, y)):
        trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]
        trn_init_score = pd.Series([0.95] * len(trn_x), index=trn_x.index) 
        val_init_score = pd.Series([0.95] * len(val_x), index=val_x.index)
        gbm = LGBMClassifier(**params)    
        gbm.fit(trn_x, trn_y, init_score=trn_init_score,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],                     
                eval_init_score=[trn_init_score, val_init_score],
                eval_metric='auc', verbose=10, early_stopping_rounds=50)
        pred_val  = gbm.predict_proba(val_x, num_iteration=gbm.best_iteration_)[:, 1]
        pred_test = gbm.predict_proba(test_X, num_iteration=gbm.best_iteration_)[:, 1]

        oof_preds[val_idx] = pred_val
        sub_preds += pred_test / folds.n_splits
        print('Fold {:02d} AUC: {:.6f}'.format(n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del trn_x, trn_y, val_x, val_y; gc.collect()
    
    oof_preds = pd.Series(oof_preds.flatten(), index=X.index).rename('TARGET')
    sub_preds = pd.Series(sub_preds.flatten(), index=test_X.index).rename('TARGET')
    return oof_preds, sub_preds


LoanID = 'SK_ID_CURR'
data = pd.read_csv('../input/application_train.csv').set_index(LoanID)
test = pd.read_csv('../input/application_test.csv').set_index(LoanID)
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')

# factorize         
categorical_feats = [f for f in data.columns if data[f].dtype == 'object']
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
    
print(data.shape, test.shape)

y = data['TARGET']
del data['TARGET']

avg_dfs = [CatMeanEnc(prev, index_name='SK_ID_PREV', groupby_ids=LoanID), 
           CatMeanEnc(buro, index_name='SK_ID_BUREAU', groupby_ids=LoanID)]
data = JoinMeanEnc(data, join_dfs=avg_dfs)
test = JoinMeanEnc(test, join_dfs=avg_dfs)

excluded_feats = [] #['SK_ID_CURR']
features = [f_ for f_ in data.columns if f_ not in excluded_feats]

params_LGBM = {
    'n_estimators'     : 1000,
    'num_leaves'       : 31,
    'colsample_bytree' : 0.8,
    'subsample'        : 0.9,
    'subsample_freq'   : 5,
    'max_depth'        : 15,
    'reg_alpha'        : .1,
    'reg_lambda'       : .1,
    'min_split_gain'   : .01,
    'min_child_weight' : 10,
    'silent'           : True,
    }

oof_preds, sub_preds = OOFPreds(X=data[features].fillna(-999), y=y, test_X=test[features].fillna(-999), 
                                params=params_LGBM, n_splits=5, random_state=42)

score = roc_auc_score(y, oof_preds)
print('Full AUC score {:.6f}'.format(score))
subm = sub_preds.to_frame()
subm.to_csv('subm_lgbm_auc{:.8f}.csv'.format(score), index=True, float_format='%.8f')

