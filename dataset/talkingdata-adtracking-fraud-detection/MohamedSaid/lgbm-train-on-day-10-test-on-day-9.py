import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import os

training_day_beginning = 144708152
training_day_end = 181878211
test_day_beginning = 82259195 
test_day_end = 118735619
chunksize = 1000000
total_data_size_train = training_day_end - training_day_beginning + 1
total_data_size_test = test_day_end - test_day_beginning + 1
total_data_size_submission = 18790469
train_data_size = 25000000
test_data_size =  1000000

reg_click_id_train_test = total_data_size_train/total_data_size_test
reg_click_id_train_test_submission = total_data_size_train/total_data_size_submission

def Data_Preparation(df):
    df['click_time'] = pd.to_datetime(df.click_time)
  
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['minuteofday'] = df['click_time'].dt.hour *60 + df['click_time'].dt.minute
    gc.collect()

    print('grouping by ip-hour combination...')
    gp = df[['ip','hour','channel']].groupby(by=['ip','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    df = df.merge(gp, on=['ip','hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    df = df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    df = df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()    

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_chl_var_hour')
    gp = df[['ip','hour','channel']].groupby(by=['ip','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    df = df.merge(gp, on=['ip','channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    df = df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    df = df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()
    df = df.fillna(0)
    print("vars and data type: ")
    df.info()
    df['ip_tcount'] = df['ip_tcount'].astype('uint16')
    df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')
    df['app'] = df['app'].astype('uint16')
    df['channel'] = df['channel'].astype('uint16')
    df['click_id'] = df['click_id'].astype('uint32')
    df['device'] = df['device'].astype('uint16')
    df['ip'] = df['ip'].astype('uint32')
    df['os'] = df['os'].astype('uint16')
    
    return df

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

iprocessing = 0
chunk_nb = 0
train_df = pd.DataFrame(columns = [ 'click_time' , 'app','device','os', 'channel','ip'  ,'is_attributed', 'click_id' ] , dtype = 'int')
for chunk in pd.read_csv('../input/train.csv' , nrows = training_day_end , 
                         dtype = dtypes, chunksize = chunksize): 
    chunk = chunk[chunk.index >= training_day_beginning]
    if chunk.shape[0] > 0:
        chunk['click_id'] = chunk.index 
        chunk['click_id'] = chunk.reset_index(drop=True).index
        chunk['click_id'] = chunk['click_id'].apply(lambda x : np.round(x + chunk_nb*chunksize))
        chunk_nb = chunk_nb + 1
        train_size_per_chunk = round(train_data_size/total_data_size_train * chunk.shape[0])
        init_chunk = chunk.index[0]  
        end_chunk = chunk.index[len(chunk.index)-1]+1
        chunk.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed', 'click_id']
        chunk = chunk.drop(['attributed_time'] , axis = 1)
        train_rows = np.random.choice(np.arange(init_chunk, end_chunk),
                                            size=train_size_per_chunk, replace=False)
        train_rows = np.sort(train_rows)
        train_df = pd.concat( [train_df,  chunk[chunk.index.isin(train_rows) ]] ) 
    iprocessing = iprocessing + 1
    print(iprocessing, " / " , int(training_day_end/chunksize)+1)
train_df = Data_Preparation(train_df)

iprocessing = 0
chunk_nb = 0
test_df = pd.DataFrame(columns =  ['click_time' , 'app','device','os', 'channel','ip'  ,'is_attributed' , 'click_id'] , dtype = 'int')
for chunk in pd.read_csv('../input/train.csv', nrows = test_day_end  ,  
                         dtype = dtypes, chunksize = chunksize):
    chunk = chunk[chunk.index >= test_day_beginning]
    if chunk.shape[0] > 0:
        chunk['click_id'] = chunk.index 
        chunk['click_id'] = chunk.reset_index(drop=True).index
        chunk['click_id'] = chunk['click_id'].apply(lambda x : np.round(x + chunk_nb*chunksize))
        chunk_nb = chunk_nb + 1
        test_size_per_chunk = round(test_data_size/total_data_size_test * chunk.shape[0])
        init_chunk = chunk.index[0]  
        end_chunk = chunk.index[len(chunk.index)-1]+1
        chunk.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed', 'click_id']
        chunk = chunk.drop(['attributed_time'] , axis = 1)
        test_rows = np.random.choice(np.arange(init_chunk, end_chunk),
                                            size=test_size_per_chunk, replace=False)
        test_rows = np.sort(test_rows)
        test_df = pd.concat( [test_df, 
                               chunk[chunk.index.isin(test_rows) ]] , axis = 0)
    iprocessing = iprocessing + 1
    print(iprocessing, " / " , int(test_day_end/chunksize)+1)

test_df = Data_Preparation(test_df)
test_df['is_attributed'] = test_df['is_attributed'].astype('uint16')
train_df['is_attributed'] = train_df['is_attributed'].astype('uint16')

del chunk, test_rows, train_rows, 
gc.collect()

target = 'is_attributed'
predictors = ['app', 'channel',  'device', 'ip',  'click_id',
       'os', 'hour', 'minuteofday', 'ip_tcount',
       'ip_app_count', 'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
       'ip_app_channel_mean_hour']
categorical = ['app', 'device', 'os', 'channel']


# Params = {
#     'metric': 'auc',
#     'learning_rate':   0.02 ,
#     'n_estimators': 153,
#     'subsample' : 0.8,
#     'boosting_type' : 'gbdt',
#     'subsample_for_bin': 120000,
#     'scale_pos_weight':409,
#     'verbose' : 1, 
#     'reg_alpha' : 3,
#     'reg_lambda' : 0 ,  
#     'num_leaves': 30,
#     'colsample_bytree' : 1,
#     'max_depth': 8,   
#     'min_child_samples': 22,
#     'subsample_freq': 1,  
#     'min_child_weight':  0.00009,  
#     'min_split_gain': 0 ,
#     'random_state' : 10,
#     'shuffle' : False
#     }
    
# params = {
#     'n_estimators ' : 153,
#     'subsample': 0.8,  
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'learning_rate': 0.03,
#     'num_leaves': 240,  
#     'max_depth': 9,  
#     'min_child_samples': 100,
#     'max_bin': 100,  
#     'subsample_freq':1,  
#     'colsample_bytree': 0.7,  
#     'min_child_weight': 11,  
#     'subsample_for_bin': 200000,  
#     'min_split_gain': 0,  
#     'reg_alpha': 0,  
#     'reg_lambda': 0,  
#     'nthread': 7,
#     'verbose': 0,
#     'scale_pos_weight':409,
#     'random_state' : 10,
#     'shuffle' : False,
#     }
    
params = {
    'boosting_type'  : 'gbdt',
    'n_estimators'  : 153,
    'objective'  : 'binary',
    'metric' : 'auc',
    'subsample_freq'  : 1,
    'learning_rate'  : 0.03,
    'max_depth' : -1,
    'reg_alpha'  :  0.0, 
    'reg_lambda' :0.0,
    'subsample'  : 0.8,
    'min_split_gain' : 0.0, 
    'random_state'  : 10, 
    'subsample_for_bin'  : 200000,
    'scale_pos_weight':409,

    'colsample_bytree'  : 1,
    'min_child_samples'  : 20,
    'min_child_weight'  : 2,
    'num_leaves'  : 23,
    }


# gridParams = {
#     'boosting_type'  : ['gbdt'],
#     'n_estimators'  : [153],
#     'objective'  : ['binary'],
#     'metric' : ['auc'],
#     'subsample_freq'  : [1],
#     'learning_rate'  : [0.03],
#     'max_depth' : [-1],
#     'reg_alpha'  :  [0.0], 
#     'reg_lambda' :[0.0],
#     'subsample'  : [0.8],
#     'min_split_gain' : [0.0], 
#     'random_state'  : [10],  
#     'min_child_samples'  : [20],
#     'min_child_weight'  :  [ 2],
#     'num_leaves'  : [ 23  ],
#     }
# grid = RandomizedSearchCV(lgb.LGBMClassifier(), gridParams, cv = 3,  
#                           verbose = 10, random_state = 10,scoring = 'roc_auc')  
# # Run the grid
# grid.fit(train_df[predictors], train_df['is_attributed'].astype('category'))




gc.collect()

print("Training...")
dtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
dvalid = lgb.Dataset(test_df[predictors].values, label=test_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )     
#del train_df
gc.collect()
                 
evals_results = {}
print("Training the model...")

lgb_model = lgb.train(  params, 
                 dtrain, 
                 valid_sets=[dtrain, dvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 #num_boost_round=1000,
                 #early_stopping_rounds=30,
                 #verbose_eval=50, 
                 #feval=None
                 )
test_target = test_df['is_attributed']
test_pred =  np.round(lgb_model.predict(test_df[predictors]))
print ("LGBM : \n" , 
       "accuracy: " , accuracy_score(test_target , test_pred ) , "\n" , 
       "recall: " , recall_score(test_target , test_pred), "\n" ,
       "precision: " , precision_score(test_target , test_pred), "\n" ,
       "f1: ", f1_score(test_target , test_pred) , "\n" ,  
       "roc_auc_score: " , roc_auc_score(test_target , test_pred), "\n"
       "confusion_matrix :" , "\n" , confusion_matrix(test_target , test_pred) )

import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')
del test_df
gc.collect()


#******************************************************************
#********************Submission**********************
#******************************************************************
submission_df = pd.DataFrame(columns=["is_attributed"])
submission_df.index.names = ['click_id']
submission_df.to_csv('submission.csv')
chunk_nb = 0
for test_df in  pd.read_csv("../input/test.csv"  , chunksize = chunksize):
    submission_df = pd.DataFrame(columns=["is_attributed"], index = test_df.index)
    submission_df.index.names = ['click_id']
    test_df['click_id'] = test_df.reset_index(drop=True).index
    test_df['click_id'] = test_df['click_id'].apply(lambda x : np.round((x + chunk_nb*chunksize)*reg_click_id_train_test))
    test_df = Data_Preparation(test_df)
    submission_df["is_attributed"] =  np.round(lgb_model.predict(test_df[predictors])).astype(int)
    submission_df.to_csv('submission.csv' , header = False,  mode = 'a' )