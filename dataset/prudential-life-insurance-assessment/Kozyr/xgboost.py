import numpy as np
import pandas as pd
import xgboost as xgb
import csv

# Parameters
xgboost_params = { 
   "objective": "multi:softmax", #reg:linear
   "num_class": 8,
   "booster": "gbtree",
   "eval_metric": "auc",
   "eta": 0.01,
   "subsample": 0.75,
   "colsample_bytree": 0.68,
   "max_depth": 7,
}

print('Load data...')
train   = pd.read_csv("../input/train.csv")
target = train['Response'].astype(int)
train  = train.drop(['Id','Response'],axis=1)

test   = pd.read_csv("../input/test.csv")
ids    = test['Id'].values
test   = test.drop(['Id'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
    else:
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            train.loc[train_series.isnull(), train_name] = train_series.mean()
        
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = train_series.mean()

target -= 1

xgtrain = xgb.DMatrix(train.values, target.values)
xgtest  = xgb.DMatrix(test.values)

print('Fit the model...')
ROUNDS = 70
clf = xgb.train(xgboost_params,xgtrain,num_boost_round=ROUNDS,verbose_eval=True,maximize=False)

print('Predict...')
test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
final_test_preds = np.round(np.clip(test_preds + 1, 1, 8)).astype(int)

print('Saving results...')
predictions_file = open('xgboost_result.csv', 'w')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['Id', 'Response'])
open_file_object.writerows(zip(ids, final_test_preds))
predictions_file.close()

print('Done.')