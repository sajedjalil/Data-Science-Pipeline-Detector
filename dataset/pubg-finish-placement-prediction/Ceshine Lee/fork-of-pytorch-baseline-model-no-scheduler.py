from subprocess import call
import os

FNULL = open(os.devnull, 'w')
call("pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.1.zip".split(" "), stdout=FNULL, stderr=FNULL)
call("pip install tensorboardX".split(" "), stdout=FNULL, stderr=FNULL)

import gc
import random
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
from timeit import default_timer

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.optim import Optimizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
import joblib

from helperbot.bot import BaseBot
from helperbot.lr_scheduler import TriangularLR
from helperbot.weight_decay import WeightDecayOptimizerWrapper

SEED = 12139

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
DEVICE = "cuda:0"


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
    return df


class Timer(object):
    """ A timer as a context manager
    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.
    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Adapted from: https://github.com/brouberol/contexttimer/blob/master/contexttimer/__init__.py

    Keyword arguments:
        output -- if True, print output after exiting context.
                  if callable, pass output to callable.
        format -- str.format string to be used for output; default "took {} seconds"
        prefix -- string to prepend (plus a space) to output
                  For convenience, if you only specify this, output defaults to True.
    """

    def __init__(self, prefix="", timer=default_timer,
                 output=None, fmt="took {:.2f} seconds"):
        self.timer = timer
        self.output = output
        self.fmt = fmt
        self.prefix = prefix
        self.end = None

    def __call__(self):
        """ Return the current time """
        return self.timer()

    def __enter__(self):
        """ Set the start time """
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

        if self.prefix and self.output is None:
            self.output = True

        if self.output:
            output = " ".join([self.prefix, self.fmt.format(self.elapsed)])
            if callable(self.output):
                self.output(output)
            else:
                print(output)
        gc.collect()

    def __str__(self):
        return '%.3f' % (self.elapsed)

    @property
    def elapsed(self):
        """ Return the current elapsed time since start
        If the `elapsed` property is called in the context manager scope,
        the elapsed time bewteen start and property access is returned.
        However, if it is accessed outside of the context manager scope,
        it returns the elapsed time bewteen entering and exiting the scope.
        The `elapsed` property can thus be accessed at different points within
        the context manager scope, to time different parts of the block.
        """
        if self.end is None:
            # if elapsed is called in the context manager scope
            return self() - self.start
        else:
            # if elapsed is called out of the context manager scope
            return self.end - self.start


def feature_engineering(input_dir="data/", is_train=True):
    # Credit: 1. https://www.kaggle.com/anycode/simple-nn-baseline-4
    #         2. https://www.kaggle.com/harshitsheoran/mlp-and-fe
    # When this function is used for the training data, load train_V2.csv :
    if is_train:
        print("processing train_V2.csv")
        df = pd.read_csv(input_dir + 'train_V2.csv')
        # Only take the samples with matches that have more than 1 player
        # there are matches with no players or just one player ( those samples could affect our model badly)
        df = df[df['maxPlace'] > 1]
    # When this function is used for the test data, load test_V2.csv :
    else:
        print("processing test_V2.csv")
        df = pd.read_csv(input_dir + 'test_V2.csv')
    df = reduce_mem_usage(df)
    gc.collect()

    # Make a new feature indecating the total distance a player cut :
    df['totalDistance'] = (
        df['rideDistance'] +
        df["walkDistance"] +
        df["swimDistance"]
    )
    df['headshotrate'] = df['kills'] / df['headshotKills']
    df['killStreakrate'] = df['killStreaks'] / df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"]+df["roadKills"]

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    df.fillna(0, inplace=True)

    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :
    df.loc[df.rankPoints < 0, 'rankPoints'] = 0

    target = 'winPlacePerc'
    # Get a list of the features to be used
    features = df.columns.tolist()

    # Remove some features from the features list :
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    features.remove("maxPlace")

    y = None

    # If we are processing the training data, process the target
    # (group the data by the match and the group then take the mean of the target)
    if is_train:
        with Timer("Calculating y:"):
            y = df.groupby(['matchId', 'groupId'])[
                target].first().values
            # Remove the target from the features list :
            features.remove(target)
    else:
        df_idx = df[["Id", "matchId", "groupId"]].copy()

    with Timer("Match level feature"):
        df_out = df.groupby(['matchId', 'groupId'])[
            ["maxPlace", "matchDuration"]].first().reset_index()

    df = df[features + ["matchId", "groupId"]].copy()
    gc.collect()

    with Timer("Mean features:"):
        # Make new features indicating the mean of the features ( grouped by match and group ) :
        agg = df.groupby(['matchId', 'groupId'])[
            features].agg('mean')
        # Put the new features into a rank form ( max value will have the highest rank)
        agg_rank = agg.groupby('matchId')[features].rank(
            pct=True).reset_index()
        agg_mean = agg.reset_index().groupby(
            'matchId')[features].mean()
        agg_mean.columns = [x + "_mean_mean" for x in agg_mean.columns]

    with Timer("Merging (mean):"):
        # Merge agg and agg_rank (that we got before) with df_out :
        df_out = df_out.merge(
            agg.reset_index(), how='left', on=['matchId', 'groupId'])
        df_out = df_out.merge(
            agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
        df_out = df_out.merge(
            agg_mean.reset_index(), how='left', on=['matchId'])
        df_out = reduce_mem_usage(df_out)

    with Timer("Max features:"):
        # Make new features indicating the max value of the features for each group ( grouped by match )
        agg = df.groupby(['matchId', 'groupId'])[features].agg('max')
        # Put the new features into a rank form ( max value will have the highest rank)
        agg_rank = agg.groupby('matchId')[features].rank(
            pct=True).reset_index()
        agg_mean = agg.groupby('matchId')[features].mean()
        agg_mean.columns = [x + "_max_mean" for x in agg_mean.columns]

    with Timer("Merging (max):"):
        # Merge the new (agg and agg_rank) with df_out :
        df_out = df_out.merge(
            agg.reset_index(), how='left', on=['matchId', 'groupId'])
        df_out = df_out.merge(agg_rank, suffixes=[
            "_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
        df_out = df_out.merge(
            agg_mean.reset_index(), how='left', on=['matchId'])
        df_out = reduce_mem_usage(df_out)

    with Timer("Min features:"):
        # Make new features indicating the minimum value of the features for each group ( grouped by match )
        agg = df.groupby(['matchId', 'groupId'])[features].agg('min')
        # Put the new features into a rank form ( max value will have the highest rank)
        agg_rank = agg.groupby('matchId')[features].rank(
            pct=True).reset_index()

    with Timer("Merging (min):"):
        # Merge the new (agg and agg_rank) with df_out :
        df_out = df_out.merge(agg.reset_index(), how='left', on=[
                              'matchId', 'groupId'])
        df_out = df_out.merge(agg_rank, suffixes=[
            "_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
        df_out = reduce_mem_usage(df_out)

    with Timer("Sum features:"):
        # Make new features indicating the minimum value of the features for each group ( grouped by match )
        agg = df.groupby(['matchId', 'groupId'])[features].agg('sum')
        # Put the new features into a rank form ( max value will have the highest rank)
        agg_rank = agg.groupby('matchId')[features].rank(
            pct=True).reset_index()

    with Timer("Merging (sum):"):
        # Merge the new (agg and agg_rank) with df_out :
        df_out = df_out.merge(agg.reset_index(), how='left', on=[
                              'matchId', 'groupId'])
        df_out = df_out.merge(agg_rank, suffixes=[
            "_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
        df_out = reduce_mem_usage(df_out)

    # Make new features indicating the number of players in each group ( grouped by match )
    with Timer("Group size:"):
        agg = df.groupby(['matchId', 'groupId']).size(
        ).reset_index(name='group_size')
        # Merge the group_size feature with df_out :
        df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    with Timer("Match mean feature"):
        # Make new features indicating the mean value of each features for each match :
        agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
        # Merge the new agg with df_out :
        df_out = df_out.merge(
            agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
        df_out = reduce_mem_usage(df_out)

    with Timer("Match median feature"):
        # Make new features indicating the mean value of each features for each match :
        agg = df.groupby(['matchId'])[features].agg('median').reset_index()
        # Merge the new agg with df_out :
        df_out = df_out.merge(
            agg, suffixes=["", "_match_median"], how='left', on=['matchId'])
        df_out = reduce_mem_usage(df_out)

    with Timer("Match size feature"):
        # Make new features indicating the number of groups in each match :
        agg = df.groupby(['matchId']).size().reset_index(name='match_size')
        # Merge the match_size feature with df_out :
        df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out[df_out == np.Inf] = np.NaN
    df_out[df_out == np.NINF] = np.NaN
    df_out.fillna(0, inplace=True)
    df_out = reduce_mem_usage(df_out)
    gc.collect()
    if is_train:
        # Drop matchId and groupId
        df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
        return df_out, y
    return df_out, df_idx

            
class PUBGBot(BaseBot):
    name = "PUBG"

    def __init__(self, model, train_loader, val_loader, *, optimizer,
                 avg_window=2000, log_dir="./data/cache/logs/",
                 log_level=logging.INFO, checkpoint_dir="./data/cache/model_cache/"):
        super().__init__(
            model, train_loader, val_loader,
            optimizer=optimizer, avg_window=avg_window,
            log_dir=log_dir, log_level=log_level, checkpoint_dir=checkpoint_dir,
            batch_idx=0, echo=False
        )
        self.criterion = torch.nn.L1Loss()
        self.loss_format = "%.8f"
    
    
def get_dataset(x, y):
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float()
    )


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_dataset(x, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


class MLPModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            weight_norm(nn.Linear(num_features, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 1)),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight_v)
                nn.init.kaiming_normal_(m.weight_g)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return torch.clamp(self.model(input_tensor), 0, 1)


def get_model(num_features):
    return MLPModel(num_features).to(DEVICE)
    

df_train, y_train = feature_engineering("../input/", is_train=True)
df_test, df_sub = feature_engineering("../input/", is_train=False)
df_test_cleaned = df_test[["matchId", "groupId", "maxPlace", "numGroups"]].copy()
df_test.drop(["matchId", "groupId"], axis=1, inplace=True)
scaler = preprocessing.MinMaxScaler(
    feature_range=(-1, 1), copy=False).fit(df_train.values)
x_train = scaler.transform(df_train.values).astype("float32")
x_test = scaler.transform(df_test.values).astype("float32")

joblib.dump([x_train, df_train.columns.tolist()],
            "x_train_dump.jl.gz", compress=3)
                
del df_train, df_test
gc.collect()

test_loader = get_dataloader(
    x_test, np.zeros(x_test.shape[0]), batch_size=1024, shuffle=False)
    
test_pred_list, val_losses = [], []
kf = KFold(n_splits=5, random_state=882)
for train_index, valid_index in kf.split(x_train):
    train_loader = get_dataloader(
        x_train[train_index], y_train[train_index],
        batch_size=256, shuffle=True
    )
    val_loader = get_dataloader(
        x_train[valid_index], y_train[valid_index],
        batch_size=1024, shuffle=False
    )
    
    model = get_model(x_train.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)
    wrapped_optimizer = WeightDecayOptimizerWrapper(
        optimizer, weight_decay=1e-3
    )
    batches_per_epoch = len(train_loader)
    bot = PUBGBot(
        model, train_loader, val_loader,
        optimizer=wrapped_optimizer, avg_window=int(batches_per_epoch / 10)
    )
    n_steps = batches_per_epoch * 20
    bot.train(
        n_steps,
        log_interval=int(batches_per_epoch / 10),
        snapshot_interval=int(batches_per_epoch / 10 * 5),
        early_stopping_cnt=10, scheduler=None)
    val_preds = bot.predict_avg(
        val_loader, k=2, is_test=True)[:, 0].cpu().numpy()
    val_losses.append(np.mean(np.abs(val_preds - y_train[valid_index])))
    print("Val loss: %.8f" % val_losses[-1])
    test_pred_list.append(bot.predict_avg(
        test_loader, k=2, is_test=True)[:, 0].cpu().numpy())
    bot.remove_checkpoints(keep=1)
    if len(test_pred_list) == 1:
        break
    
val_loss = np.mean(val_losses)
test_preds = np.mean(test_pred_list, axis=0)
print("Validation losses: %.10f +- %.10f" % (np.mean(val_losses), np.std(val_losses)))

df_test_cleaned["winPlacePerc"] = test_preds
df_test_cleaned.loc[
    df_test_cleaned.winPlacePerc < 0, "winPlacePerc"] = 0
df_test_cleaned.loc[
    df_test_cleaned.winPlacePerc > 1, "winPlacePerc"] = 1

df_sub = df_sub.merge(
    df_test_cleaned[["matchId", "groupId", "winPlacePerc"]], how="left",
    on=["matchId", "groupId"]
)
df_sub[['Id', 'winPlacePerc']].to_csv("submission_raw.csv",index=False)