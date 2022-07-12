import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

from copy import deepcopy

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import gc, sys
gc.enable()

import os
# print(os.listdir("../input"))



#
# Framework
#


# Thanks to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#        else:
#            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def add_new_features_4(df):
    
    df['headshotRate'] = df['kills'] / df['headshotKills']
    df['killStreakRate'] = df['killStreaks'] / df['kills']
    df['healsAndBoosts'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
    df['killPlaceOverMaxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKillsOverKills'] = df['headshotKills'] / df['kills']
    df['distanceOverWeapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistanceOverHeals'] = df['walkDistance'] / df['heals']
    df['walkDistanceOverKills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df['headshotKills'] + df['roadKills']
    
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    
    df.fillna(0, inplace=True)


def feature_engineering(df, is_train=True):
    
    # fix rank points
    df['rankPoints'] = np.where(df['rankPoints'] <= 0, 0, df['rankPoints'])
    
    features = list(df.columns)
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    if 'winPlacePerc' in features:
        features.remove('winPlacePerc')
    
    y = None
    
    # average y for training dataset
    if is_train:
        y = df.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean')
    elif 'winPlacePerc' in df.columns:
        y = df['winPlacePerc']
    
    # mean by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train:
        df_out = agg.reset_index()[['matchId','groupId']]
    else:
        df_out = df[['matchId','groupId']]
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # max by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # max by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # number of players in group
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # mean by match
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # number of groups in match
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # drop match id and group id
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    del agg, agg_rank
    
    return df_out, y


class Estimator(object):
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedException
    
    def predict(self, x):
        raise NotImplementedException


class ScikitLearnEstimator(Estimator):
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        self.estimator.fit(x_train, y_train)
    
    def predict(self, x):
        return self.estimator.predict(x)


def fit_step(estimator, x_train, y_train, train_idx, valid_idx, oof):
    
    # prepare train and validation data
    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]
    
    # fit estimator
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)
    
    # collect OOF
    oof_part = estimator.predict(x_train_valid)
    
    mae = mean_absolute_error(y_train_valid, oof_part)
    print('MAE:', mae)
    
    oof[valid_idx] = oof_part
    
    return estimator, mae


def fit(estimator, x_train, y_train, n_splits=5):
    
    oof = np.zeros(x_train.shape[0])
    
    kf = KFold(n_splits=n_splits, random_state=42)
    
    trained_estimators = []
    
    for train_idx, valid_idx in kf.split(x_train):
        
        e, mae = fit_step(estimator, x_train, y_train, train_idx, valid_idx, oof)
        
        trained_estimators.append(deepcopy(e))
    
    print('Final MAE:', mean_absolute_error(y_train, oof))
    
    return oof, trained_estimators


def predict(trained_estimators, x_test):
    
    y = np.zeros(x_test.shape[0])
    
    for estimator in trained_estimators:
        
        y_part = estimator.predict(x_test)
        
        # average predictions for test data
        y += y_part / len(trained_estimators)
    
    return y


def pipeline_fit(estimator, df_train, n_splits=5, scaler=None):
    
    # add new features
    add_new_features_4(df_train)
    
    # feature engineering
    x_train, y_train = feature_engineering(df_train, is_train=True)
    x_train = reduce_mem_usage(x_train)
    gc.collect()
    
    # scale
    if not (scaler is None):
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
    else:
        scaled_x_train = x_train.values
    
    del x_train
    gc.collect()
    
    # fit
    oof, trained_estimators = fit(estimator, scaled_x_train, y_train.values, n_splits)
    
    del scaled_x_train
    del y_train
    gc.collect()
    
    return oof, trained_estimators


def pipeline_predict(trained_estimators, df_test, scaler=None):
    
    # add new features
    add_new_features_4(df_test)
    
    # feature engineering
    x_test, _ = feature_engineering(df_test, is_train=False)
    x_test = reduce_mem_usage(x_test)
    gc.collect()
    
    # scale
    if not (scaler is None):
        scaled_x_test = scaler.transform(x_test)
    else:
        scaled_x_test = x_test.values
    
    del x_test
    gc.collect()
    
    # predict
    y = predict(trained_estimators, scaled_x_test)
    
    del scaled_x_test
    gc.collect()
    
    return y




#
# PyTorch
#


print('--- PyTorch ---')

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PyTorch(Estimator):
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        
        train_tensor = TensorDataset(
            torch.from_numpy(x_train.astype('float32')),
            torch.from_numpy(y_train.astype('float32')))
        train_loader = DataLoader(train_tensor, batch_size=256, shuffle=True)
        
        self.model = nn.Sequential(
            weight_norm(nn.Linear(x_train.shape[1], 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 1))).to(device)
        
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight_v)
                nn.init.kaiming_normal_(m.weight_g)
                nn.init.constant_(m.bias, 0)
        
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=1e-3)

        self.model.train()
        n_epochs = 30 # 50 is the best known value
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for train_part, y_part in train_loader:
                optimizer.zero_grad()
                y_pred_part = self.model(train_part.to(device))
                loss = criterion(y_pred_part.reshape(-1), y_part.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += y_pred_part.shape[0] * loss.item()
            print('Epoch %3d / %3d. Loss = %.5f' % (epoch + 1, n_epochs, epoch_loss / x_train.shape[0]))
    
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.from_numpy(x.astype('float32'))
        x_loader = DataLoader(x_tensor, batch_size=256, shuffle=False)
        y_pred = np.empty(0)
        with torch.no_grad():
            for x_part in x_loader:
                y_pred_part = self.model(x_part.to(device)).data.cpu().numpy().reshape(-1)
                y_pred = np.append(y_pred, y_pred_part)
        return y_pred


print('Load train data')

df_train = reduce_mem_usage(pd.read_csv('../input/train_V2.csv', index_col='Id'))

df_train.drop(df_train[df_train['winPlacePerc'].isnull()].index, inplace=True)

gc.collect()

print('Fit')

pytorch_scaler = StandardScaler()
pytorch_oof, pytorch_trained_estimators = pipeline_fit(PyTorch(), df_train, scaler=pytorch_scaler)

del df_train

gc.collect()

print('Load test data')

df_test = reduce_mem_usage(pd.read_csv('../input/test_V2.csv', index_col = 'Id'))

# df_test_id = pd.DataFrame(index=df_test.index)

gc.collect()

print('Predict')

pytorch_y = pipeline_predict(pytorch_trained_estimators, df_test, pytorch_scaler)

del df_test
del pytorch_trained_estimators
del pytorch_scaler

del pytorch_oof

gc.collect()




#
# LightGBM
#


print('--- LightGBM ---')

import lightgbm as lgb

class LightGBM(Estimator):
    
    def __init__(self, params):
        self.params = params
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        
        lgb_train = lgb.Dataset(data=x_train.astype('float32'), label=y_train.astype('float32'))
        lgb_valid = lgb.Dataset(data=x_valid.astype('float32'), label=y_valid.astype('float32'))
        
        self.lgb_model = lgb.train(self.params, lgb_train, valid_sets=lgb_valid, verbose_eval=1000)
    
    def predict(self, x):
        return self.lgb_model.predict(x.astype('float32'), num_iteration=self.lgb_model.best_iteration)

params = {'objective': 'regression',
          'metric': 'mae',
          'n_estimators': 5000, # 10000 is the best known value
          'early_stopping_rounds': 100,
          'num_leaves': 300,
          'max_depth': 14,
          'bagging_fraction': 0.9,
          'learning_rate': 0.05, # 0.03 is the best known value
          'bagging_seed': 0,
          'num_threads': 4,
          'colsample_bytree': 0.7}


print('Load train data')

df_train = reduce_mem_usage(pd.read_csv('../input/train_V2.csv', index_col='Id'))

df_train.drop(df_train[df_train['winPlacePerc'].isnull()].index, inplace=True)

gc.collect()

print('Fit')

# lgb_scaler = StandardScaler()
lgb_oof, lgb_trained_estimators = pipeline_fit(LightGBM(params), df_train, n_splits=3)

del df_train

gc.collect()

print('Load test data')

df_test = reduce_mem_usage(pd.read_csv('../input/test_V2.csv', index_col = 'Id'))

df_test_id = pd.DataFrame(index=df_test.index)

gc.collect()

print('Predict')

lgb_y = pipeline_predict(lgb_trained_estimators, df_test)

del df_test
del lgb_trained_estimators
# del lgb_scaler

del lgb_oof

gc.collect()




#
# Blending
#


print('Blending')

y = np.add(pytorch_y, lgb_y) / 2




#
# Submission
#


print('Save raw predictions')

df_submission = pd.DataFrame(index=df_test_id.index)
df_submission['winPlacePerc'] = y
df_submission.to_csv('solution_raw.csv', index_label='Id')

print('Adjust predictions')

df_test = pd.read_csv('../input/test_V2.csv')

df_submission = df_submission.merge(df_test[['Id', 'matchId', 'groupId', 'maxPlace', 'numGroups']], on='Id', how='left')

df_submission_group = df_submission.groupby(['matchId', 'groupId']).first().reset_index()

df_submission_group['rank'] = df_submission_group.groupby(['matchId'])['winPlacePerc'].rank()

df_submission_group = df_submission_group.merge(df_submission_group.groupby('matchId')['rank'].max().to_frame('max_rank').reset_index(), on='matchId', how='left')

df_submission_group['adjustedPerc'] = (df_submission_group['rank'] - 1) / (df_submission_group['numGroups'] - 1)

df_submission = df_submission.merge(df_submission_group[['adjustedPerc', 'matchId', 'groupId']], on=['matchId', 'groupId'], how='left')

df_submission['winPlacePerc'] = df_submission['adjustedPerc']

df_submission.loc[df_submission.maxPlace == 0, 'winPlacePerc'] = 0
df_submission.loc[df_submission.maxPlace == 1, 'winPlacePerc'] = 1

# Thanks to https://www.kaggle.com/anycode/simple-nn-baseline-4

t = df_submission.loc[df_submission.maxPlace > 1]
gap = 1.0 / (t.maxPlace.values - 1)
fixed_perc = np.around(t.winPlacePerc.values / gap) * gap
df_submission.loc[df_submission.maxPlace > 1, 'winPlacePerc'] = fixed_perc

df_submission.loc[(df_submission.maxPlace > 1) & (df_submission.numGroups == 1), 'winPlacePerc'] = 0

assert df_submission['winPlacePerc'].isnull().sum() == 0

print('Save adjusted predictions')

df_submission[['Id', 'winPlacePerc']].to_csv('solution_adjusted.csv', index=False)
