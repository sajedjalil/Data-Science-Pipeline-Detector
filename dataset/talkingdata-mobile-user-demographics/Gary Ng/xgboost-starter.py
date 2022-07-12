# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def read_data():
    start_time = time.time()
    print('Reading events File....')
    events = pd.read_csv('../input/events.csv')
    events = events.groupby(['device_id'])['event_id'].agg(['count']).reset_index()
    events = events[['device_id','count']].drop_duplicates('device_id',keep='first')
    print('Events Shape : {}'.format(events.shape))
    
    
    print('Reading phone brand device file ....')
    phone_brand = pd.read_csv('../input/phone_brand_device_model.csv')
    le = LabelEncoder()
    brand_fit = le.fit(phone_brand['phone_brand'])
    phone_brand['brand'] = brand_fit.transform(phone_brand['phone_brand'])
    m = phone_brand['phone_brand'].str.cat(phone_brand['device_model'])
    lemodel = le.fit(m)
    phone_brand['model'] = lemodel.transform(m)
    phone_brand.drop(['phone_brand','device_model'],axis=1,inplace=True)
    phone_brand = phone_brand.drop_duplicates('device_id',keep='first')
    print('Phone brand shape : {}'.format(phone_brand.shape))
    
    
    print('Reading training file....')
    train = pd.read_csv('../input/gender_age_train.csv')
    train.drop(['gender','age'],axis=1,inplace=True)
    train['group'] = le.fit_transform(train['group'])
    train = pd.merge(train,events,how='left',on='device_id',left_index=True)
    train = pd.merge(train,phone_brand,how='left',on='device_id',left_index=True)
    train.fillna(-1,inplace=True)
    print('Train shape : {}'.format(train.shape))
    print(train.head())
    
    print('Reading testing file ...')
    test = pd.read_csv('../input/gender_age_test.csv')
    test = pd.merge(test,events,how='left',on='device_id',left_index=True)
    test = pd.merge(test,phone_brand,how='left',on='device_id',left_index=True)
    test.fillna(-1,inplace=True)
    print('Test shape : {}'.format(test.shape))
    print(test.head())
    
    test_device_ids = test['device_id'].values
    test.drop(['device_id'],axis=1,inplace=True)
    train.drop(['device_id'],axis=1,inplace=True)
    features = train.columns.tolist()
    features.remove('group')
    print('Features : {} '.format(features))
    print('Time is %0.2f min' %((time.time() - start_time) / 60))
    del events,phone_brand,le
    return train,test,features,test_device_ids
    
def build_xgb_model(train,test,features,target):
    print('Building xgb model...')
    eta = 0.1
    max_depth = 7
    colsample_bytree = 0.8
    subsample = 0.8
    start_time = time.time()
    params  = {
    'objective':'multi:softprob',
    'eta':eta,
    'max_depth':max_depth,
    'colsample_bytree':colsample_bytree,
    'subsample':subsample,
    'silent':1,
    'random_state':42,
    'num_class':12,
    'eval_metric':'mlogloss',  ## for multi-class classification
    'booster':'gbtree'
    }
    test_size = 0.2
    num_boost_round = 500
    early_stopping_rounds = 20
    X_train,X_valid = train_test_split(train,test_size=test_size,random_state=42)
    print('X_train shape : {}'.format(X_train.shape))
    print('X_valid shape : {}'.format(X_valid.shape))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features],y_train)
    dvalid = xgb.DMatrix(X_valid[features],y_valid)
    watch_list = [(dtrain,'train'),(dvalid,'eval')]
    gbm = xgb.train(params,dtrain,num_boost_round,
                    early_stopping_rounds = early_stopping_rounds,
                    evals = watch_list,
                    verbose_eval=True)
    print('Predicting validation dataset...')
    validation_pred = gbm.predict(dvalid,ntree_limit = gbm.best_iteration)
    score = log_loss(y_valid,validation_pred)
    print('Log loss score : {}'.format(score))
    
    print('Predicting....')
    y_pred = gbm.predict(xgb.DMatrix(test[features]),ntree_limit = gbm.best_iteration)
    print('Time is %0.4f min ' %( (time.time() - start_time) / 60 ))
    return y_pred

def create_submission(test_ids,y_pred):
    print('Creating Submission')
    start_time = time.time()
    columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
    now = datetime.now()
    #filename = str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    '''
    f = open('output.csv','w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    for i in range(len(test_ids)):
        val = str(test_ids[i])
        for j in range(12):
            val += ',' + str(y_pred[i][j])
        val += '\n'
        f.write(val)
    f.close()
    '''
    output = pd.DataFrame(y_pred,columns=columns)
    output.insert(0,'device_id',test_ids)
    output.to_csv('output.csv',index=False)
    print('Time is %0.4f min ' %( (time.time() - start_time) / 60 ))

if __name__ == '__main__':
    train,test,features,test_ids = read_data()
    y_pred = build_xgb_model(train,test,features,'group')
    create_submission(test_ids,y_pred)
    
    
    
