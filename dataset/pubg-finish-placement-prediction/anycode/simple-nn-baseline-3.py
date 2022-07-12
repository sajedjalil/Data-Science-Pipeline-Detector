import numpy as np
import pandas as pd
from sklearn import preprocessing
"""
changed from https://www.kaggle.com/anycode/simple-nn-baseline

to run this kernel, pip install ultimate first from your custom packages
"""
from ultimate.mlp import MLP 
import gc, sys
gc.enable()

INPUT_DIR = "../input/"

# refer https://www.kaggle.com/chocozzz/how-to-save-time-and-memory-with-big-datasets
# def reduce_mem_usage(df):
#     """ iterate through all the columns of a dataframe and modify the data type
#         to reduce memory usage.        
#     """
#     #start_mem = df.memory_usage().sum() / 1024**2
#     #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

#     for col in df.columns:
#         col_type = df[col].dtype

#         if col_type != object:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)  
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)

#     #end_mem = df.memory_usage().sum() / 1024**2
#     #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#     #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

#     return df
    
def feature_engineering(is_train=True):
    if is_train: 
        print("processing train.csv")
        df = pd.read_csv(INPUT_DIR + 'train_V2.csv')

        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        df = pd.read_csv(INPUT_DIR + 'test_V2.csv')
    
    # df = reduce_mem_usage(df)
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    
    # df = df[:100]
    
    print("remove some columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    
    features.remove("matchType")
    
    # matchType = pd.get_dummies(df['matchType'])
    # df = df.join(matchType)    
    
    y = None
    
    print("get target")
    if is_train: 
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].agg('sum')
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank.reset_index(), suffixes=["_sum", "_sum_pct"], how='left', on=['matchId', 'groupId'])
    
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = np.array(df_out, dtype=np.float64)
    
    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names
    
x_train, y, feature_names = feature_engineering(True)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)

print("x_train", x_train.shape, x_train.max(), x_train.min())
scaler.transform(x_train)
print("x_train", x_train.shape, x_train.max(), x_train.min())

y = y * 2 - 1
print("y", y.shape, y.max(), y.min())

# score=0.248
# epoch_train = 18
# rate_init=0.08
# hidden_size = 32
# verbose=1
# epoch_decay=1

epoch_train = 36
rate_init=0.08
hidden_size = 32
verbose=1
epoch_decay=2


mlp = MLP(layer_size=[x_train.shape[1], hidden_size, hidden_size, hidden_size, 1],bias_rate=[], regularization=1,importance_mul=0.0001, output_shrink=0.1, output_range=[-1,1], loss_type="hardmse")
feature_importance = mlp.train(x_train, y, verbose=verbose, importance_out=True, iteration_log=20000, rate_init=rate_init, rate_decay=0.8, epoch_train=epoch_train, epoch_decay=epoch_decay)
del x_train, y
gc.collect()

feature_importance = list(zip(feature_names, feature_importance))
feature_importance.sort(key=lambda x:x[1], reverse=True)

print(feature_importance)

x_test, _, _ = feature_engineering(False)
scaler.transform(x_test)
print("x_test", x_test.shape, x_test.max(), x_test.min())
np.clip(x_test, out=x_test, a_min=-1, a_max=1)
print("x_test", x_test.shape, x_test.max(), x_test.min())

pred = mlp.predict(x_test)
del x_test
gc.collect()

pred = pred.reshape(-1)
pred = (pred + 1) / 2

df_test = pd.read_csv(INPUT_DIR + 'test_V2.csv')

# df_test = df_test[:100]

print("fix winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap
    
    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0    
    pred[i] = winPlacePerc

    if (i + 1) % 100000 == 0:
        print(i, flush=True, end=" ")

df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)