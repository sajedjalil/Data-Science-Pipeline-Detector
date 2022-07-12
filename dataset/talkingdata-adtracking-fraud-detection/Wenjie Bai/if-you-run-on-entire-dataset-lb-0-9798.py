"""
【If you run this script on the entire dataset within 16G RAM+64G swap ,then it will give LB:0.9798】
【Any questions and how to run on entire dataset please feel free to contact me:QQ/Wechat:496852768,496852768@qq.com】

Original kernel:non-blending lightGBM model LB: 0.977:https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
V0 Modified by Andy:Kaggle-runnable version of Baris Kanber's LightGBM:https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm/comments
A non-blending lightGBM model that incorporates portions and ideas from various public kernels. 

Modified by Bai:-----------------------------------------------------------------------------------------------------
References:
- Python学习笔记——可变对象和不可变对象:https://blog.csdn.net/taohuaxinmu123/article/details/39008281
- python id()函数, id()函数用于获取对象的内存地址。:https://www.cnblogs.com/dplearning/p/5998112.html
- def concatenate_block_managers(Source Code):https://github.com/pandas-dev/pandas/blob/v0.22.0/pandas/core/internals.py
* del:Reference count-1, gc.collect():manully garbage clean
- Frequently using small parameters have large reference count:https://segmentfault.com/q/1010000000509607
- pandas的4种引用与3种复制，是否copy只取决于采用了切片还是花式索引。：https://blog.csdn.net/qtlyx/article/details/70500145
- Python内存池管理与缓冲池设计:https://blog.csdn.net/zhzhl202/article/details/7547445
- python Pandas DataFrame copy(deep=False) vs copy(deep=True) vs '=':https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs#
* Deep copy creates new id's of every object it contains while normal copy only copies the elements from the parent and creates a new id for a variable to which it is copied to.
- 机器不学习：一文看懂机器学习时代神器—LightGBM:http://www.360doc.com/content/17/1231/23/40769523_718019029.shtml
- 比XGBOOST更快--LightGBM介绍:https://www.jianshu.com/p/48e82dbb142b
* xgboost is a little bit more accurate than lightgbm, but use 8*RAM than lightgbm
- 机器学习中，有哪些特征选择的工程方法？：https://www.zhihu.com/question/28641663/answer/107680749
* Feature selection exhaustivity is O(2^n), usally use greedy method(forward,backwrad) to find the second-best solution
- Pearson相关系数是用来衡量两个数据集合是否在一条线上面，它用来衡量定距变量间的线性关系。相关系数是研究变量之间线性相关程度的量。https://baike.baidu.com/item/Pearson%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0/6243913
- pandas.DataFrame.astype:http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.astype.html

Possible improvement:
- Use lightgbm continue training to train full dataset?
- next_click?
- combine the advantages of former kernels?
- tune the parameters?
- consider using blend?
- Add new features on entire dataset one by one?
- delete all 'day' related features
- For my own validation (applying full predictions from script, shifted back by 1 day, to test set analogue)

V0: * Use functions to optimize the coding structure, useful to debug
    * Add many new group by features and 'click_time'
    3372.4 seconds,[127]	train's auc: 0.987305	valid's auc: 0.972478, LB:0.9761, nchunk=25000000
V1: - Add all 'merge' function 'copy=False'【USEFUL for at least computational speed and prevent reach the RAM limit】
    Ran 2923 seconds, [129]	train's auc: 0.987555	valid's auc: 0.972648, LB:0.9764
V2: - Add all 'astype' function 'copy=False'
    * Use default seed: 'data_random_seed','bagging_seed','feature_fraction_seed':1,2,3
    - Add lgb.plot_importance() for both 'split' and 'gain'
    - Reorganize the lgb training function
    * 2'next_click' features contains large randomness but EXTREMELY useful
    * delete 2'next_click' features, result same to the fork, [231]	train's auc: 0.984031	valid's auc: 0.96488, LB:0.9661
    * then delete lowest score 'ip_app_channel_mean_hour' feature, [124]train's auc: 0.981878	valid's auc: 0.965349, LB:0.9662
V3: * 'nthread':for the best speed, set this to the number of real CPU cores, not the number of threads:http://lightgbm.readthedocs.io/en/latest/Parameters.html
    - delete all the gc.collect() in the feature function
    * Different 'nthread' lead to different result. It is fixed to 4 after test.
    * Same result with along same code can have large gap in time, maybe due to the server disk, so time IS NOT fixed
    - Train on local computer, using GPU
    - Using new method to generate feature 'nextClick' rather than hash
    kaggle cpu:[151]	train's auc: 0.988041	valid's auc: 0.973151,SAME to the fork, LB:0.9770, 3425.9s
    local cpu:[64]	train's auc: 0.985661	valid's auc: 0.972455,SAME to the fork, LB:0.9761, 1625s
    local gpu:[105]	train's auc: 0.987103	valid's auc: 0.97292，2235s
    local gpu double precesion:[64]	train's auc: 0.985661	valid's auc: 0.972455, 1857s, SAME to local cpu
V4: - change nchunk from 25000000 to 75000000
    CPU:[326]	train's auc: 0.985568	valid's auc: 0.990381，LB:0.9794, use 30G swap, almost 5hour
    - change nchunk from 75000000 to 140000000
    - delete 'day'.'day'只是用来切块每一天的某小时(区分不同天)，本身不用作分类
    - delete 'ip_tchan_count','ip_app_channel_mean_hour','ip_app_os_var','X7' temporarily for speed
    - [162]train's auc: 0.984942	valid's auc: 0.99025, 19709s, LB:0.9791
V5: - add 'ip_tchan_count','ip_app_channel_mean_hour','ip_app_os_var','X7' back(IMPORTANT)
    - change nchunk from 140000000 to 150000000
    [327]	train's auc: 0.985816	valid's auc: 0.990797
    - add 'day' for test
    - early_stopping_rounds=30 to 50
    [650]	train's auc: 0.986541	valid's auc: 0.99073, 16G RAM+62G swap, 28562s, LB:0.9796
V6: - delete 'day'
    - change nchunk from 150000000 to all
    [309]	train's auc: 0.985642	valid's auc: 0.990719, 16G RAM+40G swap, 23376s, LB:0.9798
    [15766.776413679123]: model training time
"""
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    # print('the Id of train_df while function before merge: ',id(df)) # the same with train_df
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    # print('the Id of train_df while function after merge: ',id(df)) # id changes
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def lgb_modelfit_nocv(dtrain, dvalid, predictors, target='target', feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None,metrics='auc'):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.2, # 【consider using 0.1】
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'scale_pos_weight': 200, # because training data is extremely unbalanced
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth), default=31
        'max_depth': 3,  # -1 means no limit, default=-1
        'min_data_per_leaf': 100,  # alias=min_data_per_leaf , min_data, min_child_samples, default=20
        'max_bin': 100,  # Number of bucketed bin for feature values,default=255
        'subsample': 0.7,  # Subsample ratio of the training instance.default=1.0, alias=bagging_fraction
        'subsample_freq': 1,  # k means will perform bagging at every k iteration, <=0 means no enable,alias=bagging_freq,default=0
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.alias:feature_fraction
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf),default=1e-3,Like min_data_in_leaf, it can be used to deal with over-fitting
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4, # should be equal to REAL cores:http://xgboost.readthedocs.io/en/latest/how_to/external_memory.html
        'verbose': 0
        #         'device': 'gpu',
#         'gpu_platform_id':1
        # gpu_use_dp, default=false,set to true to use double precision math on GPU (default using single precision)
#         'gpu_device_id': 2 #default=-1,default value is -1, means the default device in the selected platform
        # 'random_state':666 [LightGBM] [Warning] Unknown parameter: random_state
        # 'feature_fraction_seed': 666,
        # 'bagging_seed': 666, # alias=bagging_fraction_seed
        # 'data_random_seed': 666 # random seed for data partition in parallel learning (not include feature parallel)
    }
    # lgb_params.update(params) # Python dict.update()

    print("load train_df into lgb.Dataset...")
    # free_raw_data (bool, optional (default=True)) – If True, raw data is freed after constructing inner Dataset.
    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    del dtrain
    print("load valid_df into lgb.Dataset...")
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    del dvalid
    gc.collect()

    evals_results = {}
    
    # Warning:basic.py:681: UserWarning: categorical_feature in param dict is overrided.
    # https://github.com/Microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py#L679
    # https://github.com/Microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py#L483
    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)
    
    del xgtrain, xgvalid
    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])
    gc.collect()

    return (bst1,bst1.best_iteration)

# --------------------------------------------------------------------------------------------------------------
def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8', # 【consider bool?need test】
            'click_id'      : 'uint32', # 【consider 'attributed_time'?】
            }
    
    print('loading train data...',frm,to)
    # usecols:Using this parameter results in much faster parsing time and lower memory usage.
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df) # Shouldn't process individually,because of lots of count,mean,var variables
    # train_df['is_attributed'] = train_df['is_attributed'].fillna(-1)
    train_df['is_attributed'].fillna(-1,inplace=True)
    train_df['is_attributed'] = train_df['is_attributed'].astype('uint8',copy=False)
    # train_df['click_id'] = train_df['click_id'].fillna(-1)
    train_df['click_id'].fillna(-1,inplace=True)
    train_df['click_id'] = train_df['click_id'].astype('uint32',copy=False)
    
    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    gc.collect()
    
    # print('the Id of train_df before function: ',id(train_df))
    train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=False ); gc.collect()
    # print('the Id of train_df after function: ',id(train_df)) # the same id with 'df' returned
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint16', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel', 'X6','uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8','uint8', show_max=False ); gc.collect()
    train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=False ); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'channel', 'A0', show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'channel'], 'A1', show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'device', 'os','app'], 'A2', show_max=False ); gc.collect()
    # ip-device-hour?

    train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount','uint16',show_max=False ); gc.collect()
#     train_df = do_count( train_df, ['ip', 'hour'], 'ip_tcount2','uint32',show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count','uint32', show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=False ); gc.collect()
    train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=False ); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=False ); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=False ); gc.collect()
    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=False ); gc.collect()


# nextclick----------------------------------------------------------------------------------------------------------
    # print('doing nextClick')
    # predictors=[]
    # new_feature = 'nextClick'
    # filename='nextClick_%d_%d.csv'%(frm,to)

    # if os.path.exists(filename):
    #     print('loading from save file')
    #     QQ=pd.read_csv(filename).values
    # else:
    #     D=2**26
    #     train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
    #         + "_" + train_df['os'].astype(str)).apply(hash) % D
    #     # from 1970/1/1, 50year*365day*24*60*60=1,576,800,000 seconds, so 2,000,000,000 is enough
    #     click_buffer= np.full(D, 3000000000, dtype=np.uint32) # Return a new array of given shape and type, filled with fill_value.
        
    #     train_df['epochtime']= train_df['click_time'].astype(np.int64,copy=False) // 10 ** 9
    #     next_clicks= []
    #     # After reverse, the time becomes future to past, make next_clicks positive
    #     for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
    #         next_clicks.append(click_buffer[category]-t)
    #         click_buffer[category]= t
    #     del(click_buffer)
    #     QQ= list(reversed(next_clicks))

    #     if not debug:
    #         print('saving')
    #         pd.DataFrame(QQ).to_csv(filename,index=False)
            
    # train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)

    # train_df[new_feature] = pd.Series(QQ).astype('float32',copy=False)
    # predictors.append(new_feature)
    # train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
    # predictors.append(new_feature+'_shift')
    
    # del QQ
    # gc.collect()
    
#=====================================================================================================
    print('doing nextClick 2...')
    predictors=[]
    
    train_df['click_time'] = (train_df['click_time'].astype(np.int64,copy=False) // 10 ** 9).astype(np.int32,copy=False)
    train_df['nextClick'] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - train_df.click_time).astype(np.float32,copy=False)
    print(train_df['nextClick'].head(30))
    train_df.drop(['click_time','day'], axis=1, inplace=True)
    predictors.append('nextClick')
    gc.collect()
    
#----------------------------------------------------------------------------------------------------------------
    print("vars and data type: ")
    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour',
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
    categorical = ['app', 'device', 'os', 'channel', 'hour',]
    print('predictors',predictors)

    test_df = train_df[len_train:]
    test_df.drop(columns='is_attributed',inplace=True)
    train_df.drop(columns='click_id',inplace=True)
    val_df = train_df[(len_train-val_size):len_train] # Validation set
    train_df = train_df[:(len_train-val_size)]
    
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))
    train_df.info()

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id']
    gc.collect()

    print("Training...")
    start_time = time.time()

    (bst,best_iteration) = lgb_modelfit_nocv(
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            early_stopping_rounds=50, 
                            verbose_eval=True, 
                            num_boost_round=2000, 
                            categorical_features=categorical)
    del train_df
    del val_df
    gc.collect()
    print('[{}]: model training time'.format(time.time() - start_time))

    print('Plot feature importances...')
    lgb.plot_importance(bst)
    # plt.show()
    plt.gcf().savefig('feature_importance_runnablelightgbm_split.png')
    lgb.plot_importance(bst,importance_type='gain')
    plt.gcf().savefig('feature_importance_runnablelightgbm_gain.png')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    del test_df
    if not debug:
        print("writing...")
        sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    del sub
    gc.collect()
    print("All done...")
    

# Main function-------------------------------------------------------------------------------------
if __name__ == '__main__':
    inpath = '../input/'
    
    #【In order to get 0.9798, you have to change nchunk to all and frm to 0 to use entire dataset】
    nrows=184903891-1 # the first line is columns' name
    nchunk=25000000 # 【The more the better】
    val_size=2500000
    frm=nrows-75000000
    
    debug=False
    # debug=True
    if debug:
        print('*** Debug: this is a test run for debugging purposes ***')
        frm=0
        nchunk=100000
        val_size=10000
    
    to=frm+nchunk
    
    DO(frm,to,6) # fileno start from 0