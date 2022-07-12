import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def GDY5(directory):
    myseed = 321
    random.seed(myseed)
    np.random.seed(myseed)
    X_train = pd.read_json(directory + "train.json")
    index = list(range(X_train.shape[0]))
    random.shuffle(index)
    a = [np.nan]*len(X_train)
    b = [np.nan]*len(X_train)
    c = [np.nan]*len(X_train)

    for i in range(5):
        building_level = {}
        for j in X_train['manager_id'].values:
            building_level[j] = [0, 0, 0]
        test_index = index[int((i*X_train.shape[0])/5):int(((i+1) *
                           X_train.shape[0])/5)]
        train_index = list(set(index).difference(test_index))
        for j in train_index:
            temp = X_train.iloc[j]
            if temp['interest_level'] == 'low':
                building_level[temp['manager_id']][0] += 1
            if temp['interest_level'] == 'medium':
                building_level[temp['manager_id']][1] += 1
            if temp['interest_level'] == 'high':
                building_level[temp['manager_id']][2] += 1
        for j in test_index:
            temp = X_train.iloc[j]
            if sum(building_level[temp['manager_id']]) != 0:
                a[j] = \
                    (building_level[temp['manager_id']][0] *
                     1.0/sum(building_level[temp['manager_id']]))
                b[j] = (building_level[temp['manager_id']][1] *
                        1.0/sum(building_level[temp['manager_id']]))
                c[j] = (building_level[temp['manager_id']][2] *
                        1.0/sum(building_level[temp['manager_id']]))
    X_train['manager_level_low'] = a
    X_train['manager_level_medium'] = b
    X_train['manager_level_high'] = c
    interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
    X_train['interest_level'] = \
        X_train['interest_level'].apply(lambda x: interest_level_map[x])
    X_train[['manager_level_low',
             'manager_level_medium',
             'manager_level_high']] = \
            X_train[['manager_level_low',
                     'manager_level_medium',
                     'manager_level_high']].fillna(X_train[['manager_level_low',
                                                            'manager_level_medium',
                                                            'manager_level_high']].mean())
    print('GDY5', log_loss(X_train.interest_level,
                           X_train[['manager_level_low',
                                    'manager_level_medium',
                                    'manager_level_high']]))


def GPStats(data):
    p = (((data["mn"] * np.tanh((((((2.571430 > ((data["cnt"] + (data["mn"] + np.tanh(0.837500)))/2.0)).astype(float)) + ((data["cnt"] - np.tanh(np.tanh(data["mn"]))) / 2.0))/2.0) * 2.0)))) +
         (np.tanh(np.exp(-((((((0.271429 < data["mn"]).astype(float)) * 2.0) + (data["cnt"] + (2.482760 - ((data["cnt"] + (((1.714290 < (data["cnt"] / 2.0)).astype(float)) - data["mn"]))/2.0))))/2.0)*(((((0.271429 < data["mn"]).astype(float)) * 2.0) + (data["cnt"] + (2.482760 - ((data["cnt"] + (((1.714290 < (data["cnt"] / 2.0)).astype(float)) - data["mn"]))/2.0))))/2.0))))) +
         (np.exp(-((2.482760 * (((((data["mn"] * np.tanh(((data["mn"] > np.exp(-(np.tanh(data["mn"])*np.tanh(data["mn"])))).astype(float)))) + 2.230770) > (data["mn"] * data["cnt"])).astype(float)) + 0.837500))*(2.482760 * (((((data["mn"] * np.tanh(((data["mn"] > np.exp(-(np.tanh(data["mn"])*np.tanh(data["mn"])))).astype(float)))) + 2.230770) > (data["mn"] * data["cnt"])).astype(float)) + 0.837500))))) +
         ((-(((((data["st"] * 2.0) < 0.636620).astype(float)) * (((np.exp(-(((data["mn"] < 0.875000).astype(float))*((data["mn"] < 0.875000).astype(float)))) < data["mn"]).astype(float)) * (0.318310 / 2.0)))))) +
         (np.exp(-((3.141593 - ((np.exp(-((data["cnt"] + data["cnt"])*(data["cnt"] + data["cnt"]))) > ((2.230770 > ((data["cnt"] + np.tanh(((data["mn"] > (data["st"] * 0.636620)).astype(float))))/2.0)).astype(float))).astype(float)))*(3.141593 - ((np.exp(-((data["cnt"] + data["cnt"])*(data["cnt"] + data["cnt"]))) > ((2.230770 > ((data["cnt"] + np.tanh(((data["mn"] > (data["st"] * 0.636620)).astype(float))))/2.0)).astype(float))).astype(float)))))))
    return p.clip(.01, .99)


def GP(directory):
    train = pd.read_json(directory + "train.json")
    test = pd.read_json(directory + "test.json")
    interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
    train['interest_level'] = \
        train['interest_level'].apply(lambda x: interest_level_map[x])
    train = train.reset_index(drop=True)
    encoder = LabelEncoder()
    for f in ['manager_id', 'building_id']:
        encoder.fit(list(train[f]) + list(test[f]))
        train[f] = encoder.transform(train[f].ravel())
        test[f] = encoder.transform(test[f].ravel())
    train['low'] = (train.interest_level.values == 0).astype(int)
    train['medium'] = (train.interest_level.values == 1).astype(int)
    train['high'] = (train.interest_level.values == 2).astype(int)
    train['building_low'] = 0
    train['building_medium'] = 0
    train['building_high'] = 0
    train['manager_low'] = 0
    train['manager_medium'] = 0
    train['manager_high'] = 0
    cols = ['low', 'medium', 'high']
    splits = 5
    kf = StratifiedKFold(n_splits=splits, random_state=321, shuffle=True)
    for dev_index, val_index in kf.split(range(train.shape[0]),
                                         train.interest_level):
        for col in cols:
            stats = pd.DataFrame()
            stats['cnt'] = \
                np.log(train.loc[dev_index]
                       .groupby('building_id')[col].count())
            stats['mn'] = \
                train.loc[dev_index].groupby('building_id')[col].mean()
            stats['st'] = \
                train.loc[dev_index].groupby('building_id')[col].std()
            stats = stats.reset_index()
            stats.fillna(-1, inplace=True)
            btrain = train.loc[val_index][['building_id', col]].copy()
            btrain.columns = ['building_id', 'target']
            btrain = btrain.merge(stats, on='building_id', how='left')
            btrain.drop(['building_id'], inplace=True, axis=1)
            btrain.fillna(btrain.mean(), inplace=True)
            train.loc[val_index, 'building_' + str(col)] = \
                GPStats(btrain).values

    for dev_index, val_index in kf.split(range(train.shape[0]),
                                         train.interest_level):
        for col in cols:
            stats = pd.DataFrame()
            stats['cnt'] = \
                np.log(train.loc[dev_index].groupby('manager_id')[col].count())
            stats['mn'] = \
                train.loc[dev_index].groupby('manager_id')[col].mean()
            stats['st'] = \
                train.loc[dev_index].groupby('manager_id')[col].std()
            stats = stats.reset_index()
            stats.fillna(-1, inplace=True)
            btrain = train.loc[val_index][['manager_id', col]].copy()
            btrain.columns = ['manager_id', 'target']
            btrain = btrain.merge(stats, on='manager_id', how='left')
            btrain.drop(['manager_id'], inplace=True, axis=1)
            btrain.fillna(btrain.mean(), inplace=True)
            train.loc[val_index, 'manager_'+str(col)] = GPStats(btrain).values

    print('Building', log_loss(train.interest_level,
                               train[['building_low',
                                      'building_medium',
                                      'building_high']]))

    print('Manager', log_loss(train.interest_level,
                              train[['manager_low',
                                     'manager_medium',
                                     'manager_high']]))
    x1 = np.sqrt(train['building_low'] *
                 train['manager_low']).values.reshape(1, -1).T
    x2 = np.sqrt(train['building_medium'] *
                 train['manager_medium']).values.reshape(1, -1).T
    x3 = np.sqrt(train['building_high'] *
                 train['manager_high']).values.reshape(1, -1).T
    print('Geo', log_loss(train.interest_level, np.hstack([x1, x2, x3])))

    cols = ['low', 'medium', 'high']
    sub = pd.DataFrame()
    sub['listing_id'] = test['listing_id'].ravel()

    for col in cols:
        buildingstats = pd.DataFrame()
        buildingstats['cnt'] = \
            np.log(train.groupby('building_id')[col].count())
        buildingstats['mn'] = \
            train.groupby('building_id')[col].mean()
        buildingstats['st'] = \
            train.groupby('building_id')[col].std()
        buildingstats = buildingstats.reset_index()
        buildingstats.fillna(-1, inplace=True)
        x = test['building_id']
        x = x.to_frame().merge(buildingstats, on='building_id', how='left')
        x.fillna(x.mean(), inplace=True)
        managerstats = pd.DataFrame()
        managerstats['cnt'] = np.log(train.groupby('manager_id')[col].count())
        managerstats['mn'] = train.groupby('manager_id')[col].mean()
        managerstats['st'] = train.groupby('manager_id')[col].std()
        managerstats = managerstats.reset_index()
        managerstats.fillna(-1, inplace=True)
        y = test['manager_id']
        y = y.to_frame().merge(managerstats, on='manager_id', how='left')
        y.fillna(y.mean(), inplace=True)
        sub[col] = GPStats(x)*GPStats(y)
    sub[['low', 'medium', 'high']] = \
        sub[['low',
             'medium',
             'high']].div(sub[['low',
                               'medium',
                               'high']].sum(axis=1), axis=0)
    sub.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    directory = '../input/'
    GDY5(directory)
    GP(directory)
