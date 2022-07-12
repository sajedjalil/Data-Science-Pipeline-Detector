import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']
param = {
    'bagging_freq': 5,  'bagging_fraction': 0.331,  'boost_from_average':'false',   
    'boost': 'gbdt',    'feature_fraction': 0.0405, 'learning_rate': 0.0083,
    'max_depth': -1,    'metric':'auc',             'min_data_in_leaf': 80,     
    'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,  'num_threads': 8,            
    'tree_learner': 'serial',   'objective': 'binary',       'verbosity': 1
}
folds = StratifiedKFold(n_splits=15, shuffle=False, random_state=2319)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = predictions
sub.to_csv("submission.csv", index=False)