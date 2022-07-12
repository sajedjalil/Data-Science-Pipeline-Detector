import pandas as pd
import os
# -*- coding: utf-8 -*-
  
import pandas as pd
 
from sklearn import preprocessing
  
import xgboost as xgb
 
path1 = '../input/train.csv'
path2 = '../input/test.csv'


def write_submission(test_ids,preds,filename):
    preds = pd.DataFrame(preds, columns=['Class_1', 'Class_2', 'Class_3',
                                             'Class_4', 'Class_5', 'Class_6',
                                             'Class_7', 'Class_8', 'Class_9'])
    
    submission = pd.concat([test_ids, preds], axis=1)
    file_name = filename
    submission.to_csv(file_name, index=False)
 
 
 
def xgboost_solution():
  
    train=pd.read_csv(path1)
    test=pd.read_csv(path2)
    le = preprocessing.LabelEncoder()
    le.fit(train['target'])
    train['target']=le.transform(train['target'])
    
    feature_cols= [col for col in train.columns if col  not in ['target','id']]
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    y=train['target']
    test_ids=test['id']
  
    dtrain = xgb.DMatrix(X_train, label=y)
    
    dtest = xgb.DMatrix(X_test, label=None)
    
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    
    other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
    
    
    watchlist  = [(dtrain,'train')]
    
    full_param = other.copy()
    full_param.update(param)
    
    
    plst = full_param.items()
    print(plst)
    bst= xgb.train(plst, dtrain, 300, watchlist)
    preds = bst.predict(dtest)
    write_submission(test_ids,preds,'xgboost_solution.csv')
 

    
def main():
  xgboost_solution()
     
if __name__ == '__main__':
    main()