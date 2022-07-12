import numpy as np
import pandas as pd
from sklearn import preprocessing
"""
changed from https://www.kaggle.com/anycode/simple-nn-baseline

to run this kernel, pip install ultimate first from your custom packages
"""
from ultimate.mlp import MLP 
import gc
gc.enable()

def feature_engineering(is_train=True):
    if is_train: 
        print("processing train.csv")
        df = pd.read_csv('../input/train.csv')
    else:
        print("processing test.csv")
        df = pd.read_csv('../input/test.csv')
    
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    y = None
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
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = np.array(df_out, dtype=np.float64)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y
    
x_train, y = feature_engineering(True)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)

print("x_train", x_train.shape, x_train.max(), x_train.min())
scaler.transform(x_train)
print("x_train", x_train.shape, x_train.max(), x_train.min())

y = y * 2 - 1
print("y", y.shape, y.max(), y.min())

epoch_train = 15
mlp = MLP(layer_size=[x_train.shape[1], 28, 28, 28, 1], regularization=1, output_shrink=0.1, output_range=[-1,1], loss_type="hardmse")
mlp.train(x_train, y, verbose=2, iteration_log=20000, rate_init=0.08, rate_decay=0.8, epoch_train=epoch_train, epoch_decay=1)
del x_train, y
gc.collect()

x_test, _ = feature_engineering(False)
scaler.transform(x_test)
print("x_test", x_test.shape, x_test.max(), x_test.min())
np.clip(x_test, out=x_test, a_min=-1, a_max=1)
print("x_test", x_test.shape, x_test.max(), x_test.min())

pred = mlp.predict(x_test)
del x_test
gc.collect()

pred = pred.reshape(-1)
pred = (pred + 1) / 2

df_test = pd.read_csv('../input/test.csv')
df_test['winPlacePercPred'] = np.clip(pred, a_min=0, a_max=1)
aux = df_test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
df_test = df_test.merge(aux, how='left', on=['matchId','groupId'])
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)