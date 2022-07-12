#!/usr/bin/env python3
# coding: utf-8
__author__ = ('Sandro Vega Pons : https://www.kaggle.com/svpons',
              'David : https://www.kaggle.com/overfit',
              'Ravi S. : https://www.kaggle.com/rshekhar2')

'''Partially based on grid_plus_classifier script:
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-knn/
https://www.kaggle.com/overfit/facebook-v-predicting-check-ins/grid-knn/
'''

'''
You will need to execute before using this script.
 !pip install git+https://github.com/fmfn/BayesianOptimization.git
'''



import numpy as np
import pandas as pd
import os.path, os
import datetime
import time
import sys
import functools
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from bayes_opt import BayesianOptimization
import argparse
import uuid
import json

size = 10.
x_step = 1.0
y_step = 0.5
x_border_augment = x_step * 0.2
y_border_augment = y_step * 0.2

uuid_string = str(uuid.uuid4())

def map_k_precision(truthvalues, predictions):
    '''
    This is a faster implementation of MAP@k valid for numpy arrays.
    It is only valid when there is one single truth value.

    m ~ number of observations
    k ~ MAP at k -- in this case k should equal 3

    truthvalues.shape = (m,)
    predictions.shape = (m, k)
    '''
    z = (predictions == truthvalues[:, None]).astype(np.float32)
    weights = 1./(np.arange(predictions.shape[1], dtype=np.float32) + 1.)
    z = z * weights[None, :]
    return float(np.mean(np.sum(z, axis=1)))


def prepare_data(dataframe,
                     w_hour=None,
                     w_log10acc=None,
                     w_weekday=None,
                     w_month=None,
                     w_year=None,):
    mintue = dataframe['time'] % 60
    dataframe['hour'] = dataframe['time']//60
    dataframe['weekday'] = dataframe['hour']//24
    dataframe['month'] = dataframe['weekday']//30
    dataframe['year'] = (dataframe['weekday']//365+1)*w_year
    dataframe['hour'] = ((dataframe['hour'] % 24+1)+mintue/60.0)*w_hour
    dataframe['weekday'] = (dataframe['weekday'] % 7+1)*w_weekday
    dataframe['month'] = (dataframe['month'] % 12+1)*w_month
    dataframe['log10acc'] = np.log10(dataframe['accuracy'].values) * w_log10acc
    dataframe.drop(['time', 'accuracy'], axis=1, inplace=True)

    return dataframe


def process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max,
                     th=None,
                     w_x=None,
                     w_y=None,
                     w_hour=None,
                     w_log10acc=None,
                     w_weekday=None,
                     w_month=None,
                     w_year=None,
                     n_neighbors=None):

    # Working on df_train
    df_cell_train = df_train[(df_train['x'] >= x_min-x_border_augment)
                             & (df_train['x'] < x_max+x_border_augment)
                             & (df_train['y'] >= y_min-y_border_augment)
                             & (df_train['y'] < y_max+y_border_augment)].copy()
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]
    df_cell_train = prepare_data(df_cell_train, 
                                 w_hour=w_hour,
                                 w_log10acc=w_log10acc,
                                 w_weekday=w_weekday,
                                 w_month=w_month,
                                 w_year=w_year,)

    # Working on df_test
    df_cell_test = df_test[(df_test['x'] >= x_min)
                           & (df_test['x'] < x_max)
                           & (df_test['y'] >= y_min)
                           & (df_test['y'] < y_max)].copy()
    row_ids = df_cell_test.index

    # Feature engineering on x and y
    df_cell_train.loc[:, 'x'] *= w_x
    df_cell_train.loc[:, 'y'] *= w_y
    df_cell_test.loc[:, 'x'] *= w_x
    df_cell_test.loc[:, 'y'] *= w_y
    df_cell_test = prepare_data(df_cell_test, 
                                w_hour=w_hour,
                                w_log10acc=w_log10acc,
                                w_weekday=w_weekday,
                                w_month=w_month,
                                w_year=w_year,)

    # Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values.astype(float)
    if 'place_id' in df_cell_test.columns:
        df_cell_test.drop(['place_id'], axis=1, inplace=True)

    X_test = df_cell_test.values.astype(float)

    # Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=int(round(n_neighbors)),
                               weights='distance',
                               metric='manhattan')
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(
        np.argsort(y_pred, axis=1)[:, ::-1][:, :3])

    return pred_labels, row_ids


def process_grid(df_train, df_test, outfilename,
                 write_validation=False,
                 # INITIAL GUESS PARAMS
                  th=5,
                  w_x=500,
                  w_y=1000,
                  w_hour=4,
                  w_log10acc=15,
                  w_weekday=3,
                  w_month=2,
                  w_year=10,
                  n_neighbors=25
                 ):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """

    VALIDATION = 'place_id' in df_test.columns
    if VALIDATION:
        df_test2 = df_test.copy()
        truthvalues = df_test2.place_id.values
        df_test2['pred1'] = -99
        df_test2['pred2'] = -99
        df_test2['pred3'] = -99

        df_test2.drop(['x', 'y', 'accuracy', 'time'], axis=1, inplace=True)

    if (not VALIDATION) or (write_validation):
        fh = open(outfilename, 'w', encoding='utf-8')
        fh.write('row_id,place_id\n')

    for i in range((int)(size/x_step)):
        start_time_row = time.time()
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == size:
            x_max = x_max + 0.001

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == size:
                y_max = y_max + 0.001

            # STOCHASTIC SKIP
            #if np.random.random() > 0.5:
            #    continue

            # Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(
                df_train, df_test, x_min, x_max, y_min, y_max,
                th=th, w_x=w_x, w_y=w_y, w_hour=w_hour,
                w_log10acc=w_log10acc, w_weekday=w_weekday,
                w_month=w_month, w_year=w_year, n_neighbors=n_neighbors)
            for id, labs in zip(row_ids, pred_labels):
                if VALIDATION and (not write_validation):
                    df_test2.loc[id].values[1:] = np.array(labs)
                else:
                    fh.write("{0},{1}\n".format(
                        id, ' '.join([str(x) for x in labs]))
                    )
    if VALIDATION:
        truthvalues = df_test2.place_id.values
        predictions = df_test2['pred1 pred2 pred3'.split()].as_matrix()

        #truthvalues = truthvalues[predictions[:, 0] != -99]
        #predictions = predictions[predictions[:, 0] != -99]

        return map_k_precision(truthvalues, predictions)


if __name__ == '__main__':
    """
    """
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    parser = argparse.ArgumentParser("kNeighbors prediction")
    parser.add_argument("--bayes", action='store_true', default=False,
      help="Perform Bayesian optimization of hyperparameters")
    parser.add_argument("--validation", "-v", action='store_true',
      default=False, help="Fit on train set, and calculate MAP@3 on validation set ")
    d = vars(parser.parse_args())

    print('Loading data ...')
    df_train = pd.read_csv('../input/train.csv',
                           usecols=[
                               'row_id', 'x', 'y', 'accuracy', 'time', 'place_id'],
                           index_col=0)
    df_train.sort_values('time', inplace=True)
    df_test = pd.read_csv('../input/test.csv',
                          usecols=['row_id', 'x', 'y', 'accuracy', 'time'],
                          index_col=0)

    if d['validation'] and d['bayes']:
      print("Validation and Bayes options are exclusive. Pick one.")
      sys.exit(9)


    VALIDATION = d['validation']
    BAYESIAN_OPTIMIZATION = d['bayes']

    if BAYESIAN_OPTIMIZATION:
        print(
            "Bayesian Optimization mode. Taking last 10% of training as test set.")
        ninety_percent_mark = int(df_train.shape[0]*0.9)
        df_test = df_train[ninety_percent_mark:]
        df_train = df_train[:ninety_percent_mark]
        outfilename = sub_file = os.path.join(
            'vali_{0}.csv'.format(str(now_time)))

        n_neighbors = int(os.environ['NEIGH'])


        f = functools.partial(process_grid, df_train=df_train, df_test=df_test,
                              outfilename=outfilename, n_neighbors=n_neighbors)
        bo = BayesianOptimization(f=f,
                                  pbounds={
                                      'th': (0, 4.1),
                                      'w_x': (100, 1000),
                                      # Fix w_y at 1000 as the most important feature
                                      #'w_y': (500, 2000), 
                                      "w_hour": (1, 10),
                                      "w_log10acc": (3, 30),
                                      "w_weekday": (1, 10),
                                      "w_month": (1, 10),
                                      "w_year": (2, 20),
                                      "n_neighbors": (1, 30)},
                                  verbose=True
                                  )


        bo.maximize(init_points=2, n_iter=1, acq="ei", xi=0.1)
        with open('knn_params/{}.json'.format(uuid_string), 'w') as fh:
            fh.write(json.dumps(bo.res, sort_keys=True, indent=4))

        for i in range(300):
            bo.maximize(n_iter=1, acq="ei", xi=0.0) # exploration points
            with open('knn_params/{}.json'.format(uuid_string), 'w') as fh:
                fh.write(json.dumps(bo.res, sort_keys=True, indent=4))

            bo.maximize(n_iter=1, acq="ei", xi=0.1) # exploitation points
            with open('knn_params/{}.json'.format(uuid_string), 'w') as fh:
                fh.write(json.dumps(bo.res, sort_keys=True, indent=4))



    elif VALIDATION:
        print("Validation Mode. Taking last 10% of training as test set.")
        ninety_percent_mark = int(df_train.shape[0]*0.9)
        df_test = df_train[ninety_percent_mark:]
        df_train = df_train[:ninety_percent_mark]
        outfilename = sub_file = os.path.join(
            'vali_{0}.csv'.format(str(now_time)))
        process_grid(df_train, df_test, outfilename, write_validation=True)
    else:
        print("Normal Mode. Generating test set predictions.")
        outfilename = sub_file = os.path.join(
            'pred_{0}.csv'.format(str(now_time)))
        process_grid(df_train, df_test, outfilename)
