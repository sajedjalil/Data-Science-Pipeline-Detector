import pandas as pd
import numpy as np
import lightgbm as lgb


# Load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# Merge test/train datasets into a single one and separate unneeded columns
target = train_df.pop('target')
len_train = len(train_df)
merged_df = pd.concat([train_df, test_df])
ID = merged_df.pop('ID_code')[len_train:]


# Use lightgbm for prediction
# Assume all features are independent, so fit model to one feature at a time
# Then final prediction is a product of all predictions based on a single feature
# Since data contains only one feature, do not use CV - just used fixed number of iterations
params = {
    'task': 'train', 'max_depth': 1, 'boosting_type': 'gbdt',
    'objective': 'binary', 'num_leaves': 3, 'learning_rate': 0.1,
    'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 1, 'lambda_l2': 60, 'verbose': -99
}
num_runs = merged_df.shape[1]
sub_preds = np.zeros([num_runs, merged_df.shape[0]-len_train])
for run in range(num_runs): # loop over all features
    lgb_train = lgb.Dataset(merged_df.iloc[:len_train, run:run+1], target)
    gbm = lgb.train(params, lgb_train, 45, verbose_eval=1000)
    sub_preds[run, :] = gbm.predict(merged_df.iloc[len_train:, run:run+1], num_iteration=gbm.best_iteration)


# Scale prediction by inverse average target - to avoid very small numbers
# Then multiply them for all features and write submission file
sub_preds2 = (10 * sub_preds).prod(axis=0)
out_df = pd.DataFrame({'ID_code': ID, 'target': sub_preds2.astype('float32')})
out_df.to_csv('sub1f.csv', index=False)