import pandas as pd
import numpy as np
import lightgbm as lgb

num_submission = 1 
# Load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

params = {
    'task': 'train', 'max_depth': -1, 'boosting_type': 'gbdt',
    'objective': 'binary', 'num_leaves': 3, 'learning_rate': 0.01,
}

y_hat = 0.0
for feature in features: # loop over all features
    lgb_train = lgb.Dataset(train_df[feature].values.reshape(-1,1), train_df.target.values)
    gbm = lgb.train(params, lgb_train, 110, verbose_eval=5)
    y_hat += gbm.predict(test_df[feature].values.reshape(-1,1), num_iteration=gbm.best_iteration)
    
y_hat /= len(features)

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = y_hat
sub.to_csv('submission{}.csv'.format(num_submission), index=False)