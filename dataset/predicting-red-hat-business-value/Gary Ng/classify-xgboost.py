import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from operator import itemgetter
import time

def intersect(train,test):
    return list(set(train) & set(test))


def get_feature(train,test):
    train_col = list(train.columns.values)
    test_col = list(test.columns.values)
    common = intersect(train_col,test_col)
    common.remove('people_id')
    common.remove('activity_id')
    return sorted(common)

def read_data():
    print('Reading people file....')
    people = pd.read_csv('../input/people.csv',
                        dtype={'people_id':np.str,
                                'activity_id':np.str,
                                'char_38':np.int32},
                        parse_dates=['date'])
                        
    print('Reading train file ...')
    train = pd.read_csv('../input/act_train.csv',
                        dtype={'people_id':np.str,
                               'activity_id':np.str,
                               'outcome':np.int8},
                        parse_dates=['date'])

    print('Reading test file ....')
    test = pd.read_csv('../input/act_test.csv',
                       dtype={'people_id':np.str,
                              'activity_id':np.str},
                        parse_dates=['date'])
                        
    print('Processing data.....')
    for table in [train,test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop(['date'],axis=1,inplace=True)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        
        for i in range(1,11):
            table['char_'+str(i)].fillna('type -999',inplace=True)
            table['char_' + str(i)] = table['char_'+str(i)].str.lstrip('type ').astype(np.int32)
        
    
    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop(['date'],axis=1,inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1,10):
        people['char_' + str(i)] = people['char_'+str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10,38):
        people['char_'+str(i)] = people['char_'+str(i)].astype(np.int32)
    
        
    print('Merge data....')
    train = pd.merge(train,people,on='people_id',how='left')
    train.fillna(-999,inplace=True)
    test = pd.merge(test,people,on='people_id',how='left')
    test.fillna(-999,inplace=True)
    
    common_feature = get_feature(train,test)
    return train,test,common_feature

def create_feature_map(features):
    outfile = open('xgb.fmap','w')
    for i,feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

def get_importance(gbm,features):
    create_feature_map(features)
    importances = gbm.get_fscore(fmap='xgb.fmap')
    importances = sorted(importances.items() , key = itemgetter(1),reverse=True)
    return importances


def run_single(train,test,feature,target):
    
    eta = 0.2
    max_depth = 5
    subsample=0.8
    colsample_bytree = 0.8
    start_time = time.time()
    
    params = {
    'objective':'binary:logistic',
    'eta':eta,
    'booster':'gbtree',
    'max_depth':max_depth,
    'subsample':subsample,
    'colsample_bytree':colsample_bytree,
    'eval_metric':'auc',
    'silent':1
    }
    num_boost_round = 115
    early_stopping_round = 20
    test_size = 0.1
    X_train,X_valid = train_test_split(train,test_size=test_size)
    print('X_train shape : {}'.format(X_train.shape))
    print('X_valid shape : {}'.format(X_valid.shape))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[feature],label=y_train)
    dvalid = xgb.DMatrix(X_valid[feature],label=y_valid)
    watchlist = [(dtrain,'train'),(dvalid,'valid')]
    clf = xgb.train(params,dtrain,num_boost_round=num_boost_round,
             evals=watchlist,early_stopping_rounds = early_stopping_round,
             verbose_eval=True)
    print('Predicting validation.....')
    y_pred = clf.predict(dvalid,ntree_limit = clf.best_iteration + 1)
    score = roc_auc_score(y_valid,y_pred)
    print('roc auc score : {}'.format(score))
    
    imp = get_importance(clf,feature)
    print('Importance array : {}'.format(imp))
    
    
    print('Predicting test set....')
    y_pred = clf.predict(xgb.DMatrix(test[feature]),ntree_limit = clf.best_iteration +1)
    return y_pred.tolist(),score

  
train,test,common_feature = read_data()
print('Train Shape : {} '.format(train.shape))
print('Test Shape : {}'.format(test.shape))
print('Common Feature : {}'.format(common_feature))
prediction,score = run_single(train,test,common_feature,'outcome')
test_id = test['activity_id'].values
submission = pd.DataFrame({'activity_id':test_id,'outcome':prediction})
submission.to_csv('output.csv',index=False)


    