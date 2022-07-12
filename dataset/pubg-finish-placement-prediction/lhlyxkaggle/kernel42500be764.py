
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from lightgbm.sklearn import LGBMRegressor
import warnings
import os

startTime = time.time()
def memory_reduce(df):
    start_memory = df.memory_usage().sum()/1024**2
    print("Start memory usage is {:.2f} MB".format(start_memory))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            min_val = df[col].min()
            max_val = df[col].max()
            if str(col_type)[:3] == 'int':
                if min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif min_val > np.iinfo(np.int16).min and max_val < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif min_val > np.iinfo(np.int32).min and max_val < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif min_val > np.iinfo(np.int64).min and max_val < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if min_val > np.finfo(np.float16).min and max_val < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#         else:
#             print(col)
#             df[col] = df[col].astype('category')
    end_memory = df.memory_usage().sum()/1024**2
    print('End memory usage is {:.2f} MB'.format(end_memory))
    print('Decreased by {:.2f}%'.format(100 * (start_memory - end_memory) / start_memory))
    return df

def feature_engineering(is_train, debug):
    test_Idx = None
    if (is_train):
        print('processing train data')
        if (debug):
            df = memory_reduce(pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv', nrows=10000))      
        else:
            df = memory_reduce(pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv'))
            df = df[pd.notnull(df['winPlacePerc'])]     
            # df = df[df['winPlacePerc'].notnull]
    else:
        print('processing test data')
        if (debug):
            df = memory_reduce(pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv', nrows=10000))
        else:
            df = memory_reduce(pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv'))
        test_Idx = df.Id
        
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['healthItem'] = df['heals'] + df['boosts']
    df['headshotRate'] = df['headshotKills'] / df['kills']
    df['killStreakRate'] = df['killStreaks'] / df['kills']
    
    print('remove many feature')
    target = 'winPlacePerc'
    features = list(df.columns)     
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    features.remove('matchType')    
    y = None
    if (is_train):
        print('get target')         
        y = np.array(df.groupby(['matchId', 'groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print('get group mean featuers')
    agg = df.groupby(['matchId', 'groupId'])[features].agg('mean')            
    agg_rank = agg.groupby(['matchId'])[features].rank(pct=True).reset_index()  
    if (is_train):
        df_out = agg.reset_index()[['matchId', 'groupId']]     
    else:
        df_out = df[['matchId', 'groupId']]                     
    df_out = df_out.merge(agg.reset_index(), suffixes=['', ''], how='left', on=['matchId', 'groupId'])      
    df_out = df_out.merge(agg_rank, suffixes=['_mean', '_mean_rank'], how='left', on=['matchId', 'groupId'])    

    print('get group max features')
    agg = df.groupby(['matchId', 'groupId'])[features].agg('max')             
    agg_rank = agg.groupby(['matchId'])[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=['', ''], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=['_max', '_max_rank'], how='left', on=['matchId', 'groupId'])

    print('get group min features')
    agg = df.groupby(['matchId', 'groupId'])[features].agg('min')             
    agg_rank = agg.groupby(['matchId'])[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=['', ''], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=['_min', '_min_rank'], how='left', on=['matchId', 'groupId'])

    print('get group size feature')                                           
    agg = df.groupby(['matchId', 'groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    print('get group matchId mean')                                           
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=['', '_match_mean'], how='left', on='matchId')

    print('get match size feature')                                          
    agg = df.groupby(['matchId']).size().reset_index(name='matchSize')
    df_out = df_out.merge(agg, on='matchId', how='left')

    df_out.drop(['matchId', 'groupId'], axis=1, inplace=True)                 
    X = df_out
    columnsName = list(df_out.columns)
    del df, df_out, agg, agg_rank            

    gc.collect()                             
    return X, y, columnsName, test_Idx

def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective": "regression", "metric": "mae", 'n_estimators': 20000, 'early_stopping_rounds': 200,
              "num_leaves": 31, "learning_rate": 0.05, "bagging_fraction": 0.7,
              "bagging_seed": 0, "num_threads": 4, "colsample_bytree": 0.7
              }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)

    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test_y, model

if __name__ == '__main__':
    x_train, y_train, train_columns, _ = feature_engineering(True, False)
    x_test, _, _, test_Idx = feature_engineering(False, False)  
    print('The time of feature engineering is {:.2f}'.format(time.time() - startTime))

    
    # 再次内存优化
    startTime = time.time()
    x_train = memory_reduce(x_train)
    x_test = memory_reduce(x_test)
    print('The time of memory reduce is {:.2f}'.format(time.time() - startTime))


    warnings.filterwarnings('ignore')       
    startTime = time.time()
    # print(startTime)
    train_index = round(int(x_train.shape[0] * 0.8))    
    dev_X = x_train[:train_index]
    val_X = x_train[train_index:]
    dev_y = y_train[:train_index]
    val_y = y_train[train_index:]
    gc.collect()

    # 训练预测模型
    pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)
    print('The time of training the model is {:.2f}'.format(time.time() - startTime))
    
    startTime = time.time()
    df_sub = pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")
    df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
    df_sub['winPlacePerc'] = pred_test
    # 恢复一些列
    df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

    # Sort, rank, and assign adjusted ratio
    df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
    df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
    df_sub_group = df_sub_group.merge(
        df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(),
        on="matchId", how="left")
    df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

    df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
    df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

    # Deal with edge cases
    df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
    df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

    # Align with maxPlace
    # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
    subset = df_sub.loc[df_sub.maxPlace > 1]
    gap = 1.0 / (subset.maxPlace.values - 1)
    new_perc = np.around(subset.winPlacePerc.values / gap) * gap
    df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

    # Edge case
    df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
    assert df_sub["winPlacePerc"].isnull().sum() == 0

    df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)
    print('The time of prediction is {:.2f}'.format(time.time() - startTime))
