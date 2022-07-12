# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import lightgbm as lgb
import time
import gc
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import os
print(os.listdir("../input"))

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2

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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def plot_feature_corr(data, features=13):
    corrmat = data.corr()
    cols = corrmat.nlargest(features, 'winPlacePerc').index  # nlargest : Return this many descending sorted values
    cm = np.corrcoef(data[cols].values.T)  # correlation
    sns.set(font_scale=1)
    size = features - 2
    f, ax = plt.subplots(figsize=(size, math.floor(size)))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': size}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()
    return


#cleaning the usless features and values
def clean_data(data):
    data.drop(['headshotKills','teamKills','roadKills','vehicleDestroys'], axis=1, inplace=True)
    data.drop(['rideDistance','swimDistance','matchDuration','walkDistance'], axis=1, inplace=True)
    #data.drop(['MaxPlace, rankPoints','killPoints','winPoints'], axis=1, inplace=True)
    data.drop(['DBNOs','assists','killStreaks'], axis=1, inplace=True)
    data.drop(['revives','longestKill','damageDealt'], axis=1, inplace=True)
    data.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
    data.drop(['numGroups', 'weaponsAcquired'], axis=1, inplace=True)
    #data.drop(['distance','players'], axis=1, inplace=True)
    data.drop(['boosts','heals'], axis=1, inplace=True)
    data.dropna()
    return data


def results_coping_submiting(sub_preds):
    df_test = pd.read_csv('../input/test_V2.csv')
    pred = sub_preds
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
    df_test['winPlacePerc'] = pred
    submission = df_test[['Id', 'winPlacePerc']]
    submission.to_csv('submission.csv', index=False)
    return

def lightbm_lba(train, test):
    train_index = round(int(train.shape[0] * 0.8))
    y_train = train['winPlacePerc']
    x_train = train.drop(['winPlacePerc'], axis=1)
    dev_X = x_train[:train_index]
    val_X = x_train[train_index:]
    dev_y = y_train[:train_index]
    val_y = y_train[train_index:]
    gc.collect()


    params = {"objective": "regression", "metric": "mae", 'n_estimators': 20000, 'early_stopping_rounds': 200,
                  "num_leaves": 31, "learning_rate": 0.05, "bagging_fraction": 0.7,
                  "bagging_seed": 0, "num_threads": 4, "colsample_bytree": 0.7
                  }

    lgtrain = lgb.Dataset(dev_X, label=dev_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
    pred_test_y = model.predict(test, num_iteration=model.best_iteration)
    return pred_test_y


def normalise_features_by_features(data):
    #data['teamKillPerRevive'] = data['teamKills'] / (data['revives'] + 1)
    data['rankPoints'] = np.where(data['rankPoints'] <= 0 ,0 , data['rankPoints'])
    data['rankPointsInvertedKillPlace'] = data['rankPoints'] / data['killPlace']
    data['winPerKillSocre'] = data['killPoints'] / (data['winPoints'] + 1)
    data['headshotsOfKills'] = data['headshotKills'] / (data['kills'] + 1)
    data['killsPerStreak'] = data['kills']/(data['killStreaks'] + 1)
    data['damageDealtPerWeapon'] = data['damageDealt'] / (data['weaponsAcquired'] + 1)
    data['lootPerHeadshot'] = data['weaponsAcquired']/(data['headshotKills'] + 1)
    data['lootPerRoadkill'] = data['weaponsAcquired']/(data['roadKills'] + 1)
    data['lootPerBoost'] = data['weaponsAcquired']/(data['heals'] + data['boosts'] + 1)
    data['KDBNOA'] = data['kills'] + data['DBNOs'] + data['assists'] + 1
    data['lootPErKDBNOA'] = data['weaponsAcquired']/(data['KDBNOA'])
    data['damagePerKDBNOA'] = data['damageDealt'] / (data['KDBNOA'])
    #data['timePerKill'] = data['matchDuration']/(data['kills'] + 1)
    #data['lootPerTime'] = data['weaponsAcquired'] / (data['matchDuration'] + 1)
    data['roadkillsPerVehicleDestroy'] = data['roadKills'] / (data['vehicleDestroys'] + 1)
    data['distance'] = data['walkDistance'] + data['rideDistance'] + data['swimDistance']
    data['distancePerKDBNOA'] = data['distance'] / (data['KDBNOA'])
    data['distancePerBoost'] = data['distance'] / (data['heals'] + data['boosts'] + 1)
    data['distancePerRoadKill'] = data['distance'] / (data['roadKills'] + 1)
    #data['distancePerTime'] = data['distance'] / (data['matchDuration'] + 1)
    data['distancePerHeadshot'] = data['distance'] / (data['headshotKills'] + 1)
    data['distancePerLoot'] = data['distance'] / (data['weaponsAcquired'] + 1)
    data['longestkillPerLoot'] = data['longestKill'] / (data['weaponsAcquired'] + 1)
    return data


def normalise_features_by_quantity(data):
    data['players'] = data.groupby('matchId')['matchId'].transform('count')
    data['KDBNOAPerPlayers'] = data['KDBNOA'] / (data['players'])
    data['timePerPlayers'] = data['matchDuration'] / (data['players'])
    data['rankPointsPerTeam'] = data['rankPoints']/(data['numGroups'])
    data['teamKillScore'] = data['teamKills']/(data['numGroups'])
    return data


#preparing the match-based percentile rank stats
def get_match_ranks_stats(data, features):
    features_rank = data.groupby('matchId')[features].rank(pct=True)
    for f in features:
        data[f+'Perc'] = features_rank[f]
    return data


#preparing team-based stats
def get_team_ranks_stats(data, features, comparison_metric):
    if comparison_metric == 'Median':
        agg = data.groupby(['matchId', 'groupId'])[features].median()
    elif comparison_metric == 'Mean':
        agg = data.groupby(['matchId', 'groupId'])[features].mean()
    elif comparison_metric == 'Min':
        agg = data.groupby(['matchId', 'groupId'])[features].min()
    elif comparison_metric == 'Max':
        agg = data.groupby(['matchId', 'groupId'])[features].max()
    data = data.merge(agg, suffixes=['', comparison_metric], how='left', on=['matchId', 'groupId'])
    return data

def plot_pca_feature_signicity(data):
    pca_data = PCA().fit_transform(data)
    sns.heatmap(np.log(pca_data.inverse_transform(np.eye(data.shape[1]))), cmap="hot", cbar=False)
    return

def plot_pca_feature_signicity(data):
    pca_trafo = PCA()
    pca_data = pca_trafo.fit_transform(data)
    sns.heatmap(pca_trafo.inverse_transform(np.eye(data.shape[1])), cmap="hot", cbar=False)
    plt.show()
    return


def main():
    train = pd.read_csv("../input/train_V2.csv")
    test = pd.read_csv("../input/test_V2.csv")
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    train = train.dropna(subset=['winPlacePerc', 'maxPlace'])
    train = train[train['maxPlace'] > 1]
    train = normalise_features_by_features(train)
    train = normalise_features_by_quantity(train)
    train = pd.get_dummies(train, columns=['matchType'], prefix=['matchType'])
    train = get_match_ranks_stats(train, ['killPlace', 'distance'])
    train = get_team_ranks_stats(train, ['distance','lootPErKDBNOA'], 'Mean')
    train = get_team_ranks_stats(train, ['KDBNOA','killPlace'], 'Median')
    train = get_team_ranks_stats(train, ['distance','KDBNOA','killPlace','lootPErKDBNOA'], 'Min')
    train = get_team_ranks_stats(train, ['distance', 'KDBNOA', 'killPlace','lootPErKDBNOA'], 'Max')
    train = clean_data(train)
    test = normalise_features_by_features(test)
    test = normalise_features_by_quantity(test)
    test = pd.get_dummies(test, columns=['matchType'], prefix=['matchType'])
    test = get_match_ranks_stats(test, ['killPlace', 'distance'])
    test = get_team_ranks_stats(test, ['distance','lootPErKDBNOA'], 'Mean')
    test = get_team_ranks_stats(test, ['KDBNOA','killPlace'], 'Median')
    test = get_team_ranks_stats(test, ['distance', 'KDBNOA', 'killPlace','lootPErKDBNOA'], 'Min')
    test = get_team_ranks_stats(test, ['distance', 'KDBNOA', 'killPlace','lootPErKDBNOA'], 'Max')
    test = clean_data(test)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    sub_preds = lightbm_lba(train, test)
    results_coping_submiting(sub_preds)
    return

if __name__ == "__main__":
    # execute only if run as a script
    main()
