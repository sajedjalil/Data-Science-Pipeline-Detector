#**THERE IS NO MAGIC IN THIS KERNEL, ONLY DATA SCIENCE TECNIQUES**
#A lot of people in the discussion talked about 'magics'. These 'magics' are only tecniques that most of people here already know.

import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

path = '../'

#First thing we need to do is grab the indexes of real test. This code came from this brilliant kernel:
#https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split

df_test = pd.read_csv(path+"input/test.csv")
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test = df_test.values

unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in range(df_test.shape[1]):
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

del df_test
gc.collect()

train = pd.read_csv(path+"input/train.csv")
test  = pd.read_csv(path+"input/test.csv")

test_ids = test['ID_code']
train.drop(['ID_code'], axis=1, inplace=True)
test.drop('ID_code', axis=1, inplace=True)
y = train.target
train.drop('target', axis=1, inplace=True)

# Here it is necessary to transform all the fake data in NA, with this the values will not be used in the frequency features.
test.iloc[synthetic_samples_indexes] = np.nan

nrow_train = train.shape[0]
merge = pd.concat([train, test], sort=False)

#Here we do a lot of feature engineering with frequency of the values in each column.
#After that, we transform all the unique values in each column into NA values(we think that the inverse of this transformation was done by Santander).

for x in range(0,200):
    #print("Now is for var_"+str(x)+" :")
    merge['is_unique_var_'+str(x)] = merge.groupby('var_'+str(x))['var_'+str(x)].transform('count')
    
    merge['avg_by_frequency_var_'+str(x)] = merge['var_'+str(x)] / merge['is_unique_var_'+str(x)]
    merge['mul_by_frequency_var_'+str(x)] = merge['var_'+str(x)] * merge['is_unique_var_'+str(x)]
    merge['avg_by_frequency_pow05_var_'+str(x)] = merge['var_'+str(x)] / (merge['is_unique_var_'+str(x)]**0.5)
    merge['mul_by_frequency_pow05_var_'+str(x)] = merge['var_'+str(x)] * (merge['is_unique_var_'+str(x)]**0.5)
    merge['avg_by_frequency_pow01_var_'+str(x)] = merge['var_'+str(x)] / (merge['is_unique_var_'+str(x)]**0.1)
    merge['mul_by_frequency_pow01_var_'+str(x)] = merge['var_'+str(x)] * (merge['is_unique_var_'+str(x)]**0.1)
    
    merge['var_'+str(x)][merge['is_unique_var_'+str(x)] == 1] = np.nan
    
    del merge['is_unique_var_'+str(x)]

train = merge[:nrow_train]
test = merge[nrow_train:]

del merge
gc.collect()

oof_preds  = np.zeros(len(train))
test_preds_fold = np.zeros(len(test))
auc_folds  = []

params = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 5,
    'num_threads': -1,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1,
    'seed': 159
}

n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Fold: {}".format(fold_+1))
    trn_x, trn_y = train.loc[trn_], y.loc[trn_]
    val_x, val_y = train.loc[val_], y.loc[val_]
    
    dtrain = lgb.Dataset(trn_x, trn_y)
    dvalid = lgb.Dataset(val_x, val_y)
    del trn_x, trn_y
    gc.collect()

    execs = 1
    preds = np.zeros(len(val_x))
    test_preds_exec = np.zeros(len(test))
    
    for p in range(0,execs):
        params['seed'] = 159 + p
        model = lgb.train(params,
                        dtrain,
                        num_boost_round = 100000,
                        valid_sets = [dtrain, dvalid],
                        verbose_eval=10000,
                        early_stopping_rounds = 1500)
        
        preds += (model.predict(val_x) / execs)
        test_preds_exec += (model.predict(test) / execs)
    
    test_preds_fold += (test_preds_exec / n_folds)
    oof_preds[val_] = preds
    auc_folds.append(roc_auc_score(val_y, preds))
    print("FOLD AUC = {}".format(roc_auc_score(val_y, preds)))

print(auc_folds)
print("MEAN AUC = {}".format(np.mean(auc_folds)))
print("OOF AUC = {}".format(roc_auc_score(y, oof_preds)))

sub = pd.DataFrame({'ID_code': test_ids, 'target': test_preds_fold})
sub.to_csv('no_magic_sub.csv', index=False)