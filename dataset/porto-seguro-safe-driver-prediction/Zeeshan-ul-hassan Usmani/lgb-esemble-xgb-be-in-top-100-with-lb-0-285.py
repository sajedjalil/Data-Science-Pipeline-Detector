
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from multiprocessing import * 

import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

print(train.values.shape, test.values.shape)

d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}



## XGB 
## Thanks to The1owl for providing well tuned parameters for xgb == https://www.kaggle.com/the1owl/forza-baseline/code

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df


x1 = multi_transform(x1)
x2 = multi_transform(x2)
lgbtest=test
test = multi_transform(test)

col = [c for c in x1.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
print(x1.values.shape, x2.values.shape)

#remove duplicates just in case
tdups = multi_transform(train)
dups = tdups[tdups.duplicated(subset=col, keep=False)]

x1 = x1[~(x1['id'].isin(dups['id'].values))]
x2 = x2[~(x2['id'].isin(dups['id'].values))]
print(x1.values.shape, x2.values.shape)

y1 = x1['target']
y2 = x2['target']
x1 = x1[col]
x2 = x2[col]


watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=200)
test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
test[['id','target']].to_csv('xgb_submission.csv', index=False, float_format='%.5f')


#### LGB Ensemble
##Thanks to Vladimir Demidov == source: https://www.kaggle.com/yekenot/simple-stacker-lb-0-284/code 

test = lgbtest
train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res

        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['random_state'] = 99

#incorporated one more layer of my defined lgb params 
lgb_params4 = {}
lgb_params4['n_estimators'] = 1450
lgb_params4['max_bin'] = 20
lgb_params4['max_depth'] = 6
lgb_params4['learning_rate'] = 0.25 # shrinkage_rate
lgb_params4['boosting_type'] = 'gbdt'
lgb_params4['objective'] = 'binary'
lgb_params4['min_data'] = 500         # min_data_in_leaf
lgb_params4['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params4['verbose'] = 0

lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

lgb_model4 = LGBMClassifier(**lgb_params4)

log_model = LogisticRegression()

stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3, lgb_model4))        
        
y_pred = stack.fit_predict(train, target_train, test)        

lgbsub = pd.DataFrame()
lgbsub['id'] = id_test
lgbsub['target'] = y_pred
lgbsub.to_csv('lgb_esm_submission.csv', index=False)


df1 = pd.read_csv('lgb_esm_submission.csv')
df2 = pd.read_csv('xgb_submission.csv') 
df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
blend = pd.merge(df1, df2, how='left', on='id')
for c in df1.columns:
    if c != 'id':
        blend[c] = (blend[c]*0.07)  + (blend[c+'_']*0.03)
blend = blend[df1.columns]
blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0,1)
blend.to_csv('final_submission.csv', index=False, float_format='%.6f')

