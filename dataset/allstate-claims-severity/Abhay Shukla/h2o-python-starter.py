import h2o
import numpy as np
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import pandas as pd

h2o.init()

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

label = train.loss
train.drop('loss', axis=1, inplace=True)

train['flag'] = 1
test['flag'] = -1

train_test = train.append(test, ignore_index=True)

cat_cols = list(train_test.columns[train_test.dtypes == object])
for col in cat_cols:
    train_test[col] = train_test[col].astype('category').cat.codes

train = train_test.loc[train_test.flag == 1, :]
test = train_test.loc[train_test.flag == -1, :]

train.drop('flag', axis=1, inplace=True)
test.drop('flag', axis=1, inplace=True)
train['loss'] = label

train = h2o.H2OFrame(train)
train['loss'] = (train['loss'] + 200).apply(lambda x: np.log(x))
# train.describe()

test = h2o.H2OFrame(test)
test_id = test['id']

# MODEL BUILDING
gbm_regressor = H2OGradientBoostingEstimator(distribution='gaussian', ntrees=2000, max_depth=6, min_rows=20,
                learn_rate=0.05, nfolds=4, ignore_const_cols=True, seed=0, sample_rate=0.7, col_sample_rate=0.7,
                col_sample_rate_per_tree=0.7, score_each_iteration=False, stopping_rounds=25, stopping_metric='deviance')
gbm_regressor.train(x=range(1, train.ncol-1), y=train.ncol-1, training_frame=train)

# gbm_regressor

test_predictions = gbm_regressor.predict(test[range(1, test.ncol)])
test_predictions = test_predictions.apply(lambda x: np.exp(x)) - 200

submission = pd.concat((h2o.as_list(test_id), h2o.as_list(test_predictions)), axis=1, ignore_index=True)
submission.columns = ['id', 'loss']
submission = submission.set_index(['id', 'loss'])
submission.to_csv('submission_h2o_kernel.csv')