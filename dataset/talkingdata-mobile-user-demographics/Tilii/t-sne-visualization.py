__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
__author__ = 'tilii: https://kaggle.com/tilii7'

# ZFTurbo defined first 3 features
# tilii added two new features and t-SNE clustering & visualization
# used some ideas from https://www.kaggle.com/cast42/santander-customer-satisfaction/t-sne-manifold-visualisation/code

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import manifold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def run_tsne(train, target):
    sss = StratifiedShuffleSplit(target, test_size=0.1)
    for train_index, test_index in sss:
        break

    X_train, X_valid = train[train_index], train[test_index]
    Y_train, Y_valid = target[train_index], target[test_index]

    train_norm = normalize(X_valid, axis=0)
    tsne = manifold.TSNE(n_components=3,
                         init='pca',
                         random_state=101,
                         method='barnes_hut',
                         n_iter=500,
                         verbose=2)
    train_tsne = tsne.fit_transform(train_norm)
    return (train_tsne, Y_valid)


def tsne_vis(tsne_data, tsne_groups):
    colors = cm.rainbow(np.linspace(0, 1, 12))
    labels = ['F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42', 'F43+', 'M22-',
              'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']

    plt.figure(figsize=(10, 10))
    for l, c, co, in zip(labels, colors, range(12)):
        plt.scatter(tsne_data[np.where(tsne_groups == co), 0],
                    tsne_data[np.where(tsne_groups == co), 1],
                    marker='o',
                    color=c,
                    linewidth='1',
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on 10% of train samples')
    plt.legend(loc='best')
    plt.savefig('rainbow-01.png')
    plt.show(block=False)

    plt.figure(figsize=(10, 10))
    for l, c, co, in zip(labels, colors, range(12)):
        plt.scatter(tsne_data[np.where(tsne_groups == co), 0],
                    tsne_data[np.where(tsne_groups == co), 2],
                    marker='o',
                    color=c,
                    linewidth='1',
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 3')
    plt.title('t-SNE on 10% of train samples')
    plt.legend(loc='best')
    plt.savefig('rainbow-02.png')
    plt.show(block=False)

    plt.figure(figsize=(10, 10))
    for l, c, co, in zip(labels, colors, range(12)):
        plt.scatter(tsne_data[np.where(tsne_groups == co), 1],
                    tsne_data[np.where(tsne_groups == co), 2],
                    marker='o',
                    color=c,
                    linewidth='1',
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 2')
    plt.ylabel('Dimension 3')
    plt.title('t-SNE on 10% of train samples')
    plt.legend(loc='best')
    plt.savefig('rainbow-03.png')
    plt.show(block=False)


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table


def read_train_test():
    # App events
    print('\nReading app events...')
    ape = pd.read_csv('../input/app_events.csv')
    ape['installed'] = ape.groupby(
        ['event_id'])['is_installed'].transform('sum')
    ape['active'] = ape.groupby(
        ['event_id'])['is_active'].transform('sum')
    ape.drop(['is_installed', 'is_active'], axis=1, inplace=True)
    ape.drop_duplicates('event_id', keep='first', inplace=True)
    ape.drop(['app_id'], axis=1)

    # Events
    print('Reading events...')
    events = pd.read_csv('../input/events.csv', dtype={'device_id': np.str})
    events['counts'] = events.groupby(
        ['device_id'])['event_id'].transform('count')

    print('Making events features...')
    # The idea here is to count the number of installed apps using the data
    # from app_events.csv above. Also to count the number of active apps.
    events = pd.merge(events, ape, how='left', on='event_id', left_index=True)

    # Below is the original events_small table
    # events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
    # And this is the new events_small table with two extra features
    events_small = events[['device_id', 'counts', 'installed',
                           'active']].drop_duplicates('device_id',
                                              keep='first')

    # Phone brand
    print('Reading phone brands...')
    pbd = pd.read_csv('../input/phone_brand_device_model.csv',
                      dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Train
    print('Reading train data...')
    train = pd.read_csv('../input/gender_age_train.csv',
                        dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    print('Merging features with train data...')
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train,
                     events_small,
                     how='left',
                     on='device_id',
                     left_index=True)
    train.fillna(-1, inplace=True)

    # Test
    print('Reading test data...')
    test = pd.read_csv('../input/gender_age_test.csv',
                       dtype={'device_id': np.str})
    print('Merging features with test data...\n')
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test,
                    events_small,
                    how='left',
                    on='device_id',
                    left_index=True)
    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')
    return train, test, features


train, test, features = read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}\n'.format(len(features), sorted(features)))
train_df = pd.DataFrame(data=train)
X = train_df.drop(['group', 'device_id'], axis=1).values
Y = train_df['group'].values
tsne_data, tsne_groups = run_tsne(X, Y)
tsne_vis(tsne_data, tsne_groups)
