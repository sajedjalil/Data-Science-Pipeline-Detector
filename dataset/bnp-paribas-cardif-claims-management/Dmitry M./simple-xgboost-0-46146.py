import pandas as pd
import xgboost as xgb
import csv

# XGBoost params:
xgboost_params = { 
   "objective": "binary:logistic",
   "booster": "gbtree",
   "eval_metric": "auc",
   "eta": 0.01, # 0.06, #0.01,
   #"min_child_weight": 240,
   "subsample": 0.75,
   "colsample_bytree": 0.68,
   "max_depth": 7
}

print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['target']
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv("../input/test.csv")
ids = test['ID'].values
test = test.drop(['ID'],axis=1)
#
print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            train.loc[train_series.isnull(), train_name] = train_series.mean()
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = train_series.mean()  #TODO

xgtrain = xgb.DMatrix(train.values, target.values)
xgtest = xgb.DMatrix(test.values)

#Now let's fit the model
print('Fit the model...')
boost_round = 5 #1800 CHANGE THIS BEFORE START
clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)

#Make predict
print('Predict...')
test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
# Save results
#
predictions_file = open("simple_xgboost_result.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(ids, test_preds))
predictions_file.close()
#
print('Done.')