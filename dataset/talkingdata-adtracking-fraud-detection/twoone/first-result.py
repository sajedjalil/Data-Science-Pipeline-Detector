# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# This kernel have improvement from Pranav Pandya and Andy Harless
# Pranav Kernel: https://www.kaggle.com/pranav84/xgboost-on-hist-mode-ip-addresses-dropped
# Andy Kernel: https://www.kaggle.com/aharless/jo-o-s-xgboost-with-memory-usage-enhancements

import gc
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from contextlib import contextmanager
from memory_profiler import profile
from memory_profiler import memory_usage
from sklearn.utils import shuffle
from collections import defaultdict

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#
#







import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
is_valid=True
path='../input/talkingdata-adtracking-fraud-detection/'
model_path='../input/xgboost-drop-datetime-model/'

@contextmanager
def timer(name):
    t0=time.time()
    yield
    print(f'[{name}] done in {time.time()-t0:.0f} s')
    
def cur_python_mem():
    mem_usage = memory_usage(-1, interval=0.2, timeout=1)
    return mem_usage

def timeFeatures(df,drop_datetime=True):  
    df['datetime']=pd.to_datetime(df['click_time'])  
    df['dow']=df['datetime'].dt.dayofweek            
    df['doy']=df['datetime'].dt.dayofyear            
    if(drop_datetime):                            
        df.drop(['click_time','datetime'],axis=1,inplace=True)   
    return df
    
start_time = time.time()

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
# Read the last lines because they are more impacting in training than the starting lines
#从184 447 044 negtive   456846 positive very imbalance

# def randomsample(sample_num,count):
#     with timer('read all train data'):
#         train = pd.read_csv(path+"train.csv", usecols=train_columns, dtype=dtypes)
#     negtive=train[train['is_attributed']==0]
#     positive=train[train['is_attributed']==1]
#     del train
#     gc.collect()
#     print(negtive.shape[0])
#     print(positive.shape[0])
#     for i in range(count):
#         sample=negtive.sample(sample_num)
#         res=pd.concat([sample,positive])
#         res.to_csv(f'sample{i}.csv')
#         del sample,res
#         gc.collect()
        
# randomsample(int(456846*2.5),10)
# exit(0)
# with timer('read train'):
#     train = pd.read_csv(path+"train.csv", skiprows=range(1,123903891), nrows=61000000, usecols=train_columns, dtype=dtypes)
# with timer('read test'):
#     test = pd.read_csv(path+"test_supplement.csv", usecols=test_columns, dtype=dtypes)     

# # print(train)    #数据量非常大，一共是61000000我行数据
# y = train['is_attributed']
# train.drop(['is_attributed'], axis=1, inplace=True)

# # Drop IP and ID from test rows
# sub = pd.DataFrame()
# #sub['click_id'] = test['click_id'].astype('int')
# test.drop(['click_id'], axis=1, inplace=True)
# gc.collect()

# nrow_train = train.shape[0]
# with timer('concat train and test'):
#     merge = pd.concat([train, test])

# del train, test
# gc.collect()

## Count the number of clicks by ip, 统计所有的ip
# with timer('merge group by ip'):
#     ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
# ip_count.columns = ['ip', 'clicks_by_ip']
# del merge
# gc.collect()
# # print(ip_count)
# with timer('merge merge and ip_count'):
#     merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
# merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
# merge.drop('ip', axis=1, inplace=True)

# train = merge[:nrow_train]
# test = merge[nrow_train:]

# del test, merge
# gc.collect()
# with timer('timefeatures'):
#     train = timeFeatures(train)    #
# gc.collect()

# # Set the params(this params from Pranav kernel) for xgboost model
# params = {'eta': 0.3,
#           'tree_method': "hist",
#           'grow_policy': "lossguide",
#           'max_leaves': 1400,  
#           'max_depth': 0, 
#           'subsample': 0.9, 
#           'colsample_bytree': 0.7, 
#           'colsample_bylevel':0.7,
#           'min_child_weight':0,
#           'alpha':4,
#           'objective': 'binary:logistic', 
#           'scale_pos_weight':9,
#           'eval_metric': 'auc', 
#           'nthread':8,
#           'random_state': 99, 
#           'silent': True}
          
# if (is_valid == True):
#     # Get 10% of train dataset to use as validation
#     x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
#     dtrain = xgb.DMatrix(x1, y1)
#     dvalid = xgb.DMatrix(x2, y2)
#     del x1, y1, x2, y2 
#     gc.collect()
#     watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
#     with timer('xgboost valid and train'):
#         model = xgb.train(params, dtrain, 100, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)
#     del dvalid
# else:
#     dtrain = xgb.DMatrix(train, y)
#     del train, y
#     gc.collect()
#     watchlist = [(dtrain, 'train')]
#     with timer('predict'):
#         model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)
    
# del dtrain
# gc.collect()

# # Plot the feature importance from xgboost
# plot_importance(model)
# plt.gcf().savefig('feature_importance_xgb.png')

# # Load the test for predict 
# print(cur_python_mem())
# with timer('save model'):
#     model.save_model('xgboost_drop_datetime.model')
#     # model.save_model('xgboost_drop_datetime.model')
#     # dump model
#     model.dump_model('dump.raw.txt')
#     # dump model with feature map
#     model.dump_model('dump.nice.txt', 'featmap.txt')
# #load model
# print('restore model')
# with timer('restore model'):
#     model = xgb.Booster(model_file='../input/xgboost-drop-datetime-model/xgboost_drop_datetime.model')
# with timer('read test'):
#     test = pd.read_csv(path+"test.csv", usecols=test_columns, dtype=dtypes)
# test = pd.merge(test, ip_count, on='ip', how='left', sort=False)
# del ip_count
# gc.collect()

# sub['click_id'] = test['click_id'].astype('int')

# test['clicks_by_ip'] = test['clicks_by_ip'].astype('uint16')
# test = timeFeatures(test)
# test.drop(['click_id', 'ip'], axis=1, inplace=True)
# dtest = xgb.DMatrix(test)
# del test
# gc.collect()

# # Save the predictions
# sub['is_attributed'] = model.predict(dtest)#, ntree_limit=model.best_ntree_limit)
# sub.to_csv('xgboost_drop_datetime_sub.csv', float_format='%.8f', index=False)

#################################################################################################################################
########################################### use split data#####################################################
#################################################################################################################################
#train 10 xgboost model
split_data_path='../input/split-dataset/'
models_predictions = defaultdict(list)
print(os.listdir('../input/xgboost-drop-datetime-model/'))
real_res=None
# Set the params(this params from Pranav kernel) for xgboost model
params = {'eta': 0.3,
             'tree_method': "hist",
             'grow_policy': "lossguide",
             'max_leaves': 1400,  
            'max_depth': 0, 
            'subsample': 0.9, 
              'colsample_bytree': 0.7, 
              'colsample_bylevel':0.7,
              'min_child_weight':0,
              'alpha':4,
              'objective': 'binary:logistic', 
              'scale_pos_weight':9,
              'eval_metric': 'auc', 
              'nthread':8,
              'random_state': 99, 
              'silent': True}
with timer('read test'):
    final_test = pd.read_csv(path+"test.csv", usecols=test_columns, dtype=dtypes)
with timer('read test'):
    test_sup= pd.read_csv(path+"test_supplement.csv", usecols=test_columns, dtype=dtypes)  
    test_sup.drop(['click_id'], axis=1, inplace=True)
    gc.collect()
for i in range(10): 
    with timer('read train'):
        train = pd.read_csv(split_data_path+"sample%d.csv" %(i),usecols=train_columns, dtype=dtypes)
        train=shuffle(train)

    y = train['is_attributed']
    train.drop(['is_attributed'], axis=1, inplace=True)

    nrow_train = train.shape[0]
    with timer('concat train and test'):
        merge = pd.concat([train, test_sup])
    
    del train
    gc.collect()
    
    # Count the number of clicks by ip, 统计所有的ip
    with timer('merge group by ip'):
        ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
    ip_count.columns = ['ip', 'clicks_by_ip']
    # print(ip_count)
    with timer('merge merge and ip_count'):
        merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
    merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
    merge.drop('ip', axis=1, inplace=True)
    
    train = merge[:nrow_train]
    test_ = merge[nrow_train:]
    
    del test_, merge
    gc.collect()
    # with timer('timefeatures'):
    #     train = timeFeatures(train)    #
    # gc.collect()
    
    # if (is_valid == True):
    #     # Get 10% of train dataset to use as validation
    #     x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
    #     dtrain = xgb.DMatrix(x1, y1)
    #     dvalid = xgb.DMatrix(x2, y2)
    #     del x1, y1, x2, y2 
    #     gc.collect()
    #     watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    #     with timer('xgboost valid and train'):
    #         model = xgb.train(params, dtrain, 100, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)
    #     del dvalid
    # else:
    #     dtrain = xgb.DMatrix(train, y)
    #     del train, y
    #     gc.collect()
    #     watchlist = [(dtrain, 'train')]
    #     with timer('predict'):
    #         model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)
        
    # del dtrain
    # gc.collect()
    
    # Plot the feature importance from xgboost
    # plot_importance(model)
    # plt.gcf().savefig(f'feature_importance_xgb{i}.png')
    
    # Load the test for predict 
    # print(cur_python_mem())
    # with timer(f'save model {i}'):
    #     model.save_model(f'xgboost_drop_datetime_split{i}.model')
        # model.save_model('xgboost_drop_datetime.model')
        # dump model
        # model.dump_model('dump.raw.txt')
        # # dump model with feature map
        # model.dump_model('dump.nice.txt', 'featmap.txt')
    ##load model
    print('restore model')
    with timer(f'restore model{i}'):
        
        model = xgb.Booster(model_file=f'../input/xgboost-drop-datetime-model/xgboost_drop_datetime{i}.model')
    
    test_ = pd.merge(final_test, ip_count, on='ip', how='left', sort=False)
    del ip_count
    gc.collect()
    
    test_['clicks_by_ip'] = test_['clicks_by_ip'].astype('uint16')
    test_ = timeFeatures(test_)
    test_.drop(['click_id', 'ip'], axis=1, inplace=True)
    dtest = xgb.DMatrix(test_)
    
    with timer('Save the predictions'):
        models_predictions[f'is_attributed{i}']=model.predict(dtest).reshape((1,-1))[0]
    del test_
    gc.collect()
    # sub[f'is_attributed{i}'] = model.predict(dtest).reshape((1,-1))[0]#, ntree_limit=model.best_ntree_limit)
    # sub.to_csv(f'xgboost_drop_datetime_sub{i}.csv', float_format='%.8f', index=False)
split_predict=pd.DataFrame(models_predictions)
split_predict.index = final_test['click_id'].astype('int')
print('del final_test')
del final_test
split_predict.mean(axis=1).to_csv(f'xgboost_drop_datetime_split.csv',float_format='%.8f', index=False)
#################################################################################################################################