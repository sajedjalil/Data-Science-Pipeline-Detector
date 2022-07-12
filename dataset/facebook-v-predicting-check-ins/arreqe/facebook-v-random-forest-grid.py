# coding: utf-8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# COMPETITION NAME (the 'Competition'): Facebook V: Predicting Check Ins
# COMPETITION SPONSOR: Facebook
# COMPETITION WEBSITE: https://www.kaggle.com/c/facebook-v-predicting-check-ins

"""
Partially based on several scripts:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts
"""

__author__ = 'Arman Zharmagambetov: armanform@gmail.com'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

x_threshold = 0.025
y_threshold = 0.0125

# area 10km by 10 km is divided into grids of size 20x40
grid_size = 10.0
x_step = 0.5
y_step = 0.25


def prepare_data(df):
    """
    Feature engineering
    """

    minute = df.time % 60
    df['hour'] = df['time'] // 60
    df.drop(['time'], axis=1, inplace=True)
    df['weekday'] = df['hour'] // 24
    df['month'] = df['weekday'] // 30
    df['year'] = (df['weekday'] // 365 + 1) * 10.0
    df['hour'] = ((df['hour'] % 24 + 1) + minute / 60.0) * 4.0
    df['weekday'] = (df['weekday'] % 7 + 1) * 3.0
    df['month'] = (df['month'] % 12 + 1) * 2.0
    df['accuracy'] = np.log10(df['accuracy']) * 10.0

    return df


def process_one_cell(df_train, df_test, th, x_min, y_min, x_max, y_max):
    """   
    Classification inside one grid cell.
    """

    x_min_th = x_min - x_threshold
    y_min_th = y_min - y_threshold
    x_max_th = x_max + x_threshold
    y_max_th = y_max + y_threshold

    # Working on df_train, getting few extra points outside this grid
    df_cell_train = df_train[(df_train['x'] >= x_min_th)
                             & (df_train['x'] <= x_max_th)
                             & (df_train['y'] >= y_min_th)
                             & (df_train['y'] <= y_max_th)]

    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    # Feature engineering on x and y for test
    df_cell_train.loc[:, 'x'] *= 500.0
    df_cell_train.loc[:, 'y'] *= 1000.0

    df_cell_train = df_cell_train.loc[mask]

    # Working on df_test
    df_cell_test = df_test[(df_test['x'] >= x_min_th) & (df_test['x'] <= x_max_th) &
                           (df_test['y'] >= y_min_th) & (df_test['y'] <= y_max_th)]
    row_ids = df_cell_test.index
    # Feature engineering on x and y for test
    df_cell_test.loc[:, 'x'] *= 500.0
    df_cell_test.loc[:, 'y'] *= 1000.0

    # Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values
    X_test = df_cell_test.values

    # Applying the classifier
    clf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1, min_samples_split=4,
                                 random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:, ::-1][:, :3])

    return pred_labels, row_ids


def process_grid(df_train, df_test, th):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    iterations_x = int(grid_size / x_step) # 20
    iterations_y = int(grid_size / y_step) # 40

    for i in range(iterations_x):
        print(i)
        x_min = x_step * i
        x_max = x_step * i + x_step
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == grid_size:
            x_max += 0.001

        for j in range(iterations_y):
            y_min = y_step * j
            y_max = y_step * j + y_step
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == grid_size:
                y_max += 0.001

            # Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_train, df_test, th, x_min, y_min, x_max, y_max)

            # Updating predictions
            preds[row_ids] = pred_labels

    print('Generating submission file')
    # Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    # Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

    # Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('submission_rf.csv', index=True, header=True, index_label='row_id')


def main():
    print('Loading data')
    df_train = pd.read_csv('../input/train.csv',
                           usecols=['row_id', 'x', 'y', 'accuracy', 'time', 'place_id'],
                           index_col=0)
    df_test = pd.read_csv('../input/test.csv',
                          usecols=['row_id', 'x', 'y', 'accuracy', 'time'],
                          index_col=0)

    print('Preparing train data')
    df_train = prepare_data(df_train)
    print(df_train.shape)
    # add data for periodic time that hit the boundary
    pd.options.mode.chained_assignment = None
    add_data = df_train[df_train.hour < 6]
    add_data.hour += 96
    df_train = df_train.append(add_data)

    add_data = df_train[df_train.hour > 98]
    add_data.hour -= 96
    df_train = df_train.append(add_data)

    print(df_train.shape)
    print('Preparing test data')
    df_test = prepare_data(df_test)

    # Solving classification problems inside each grid cell
    th = 8  # Keeping place_ids with more than th samples.
    process_grid(df_train, df_test, th)


if __name__ == '__main__':
    main()
