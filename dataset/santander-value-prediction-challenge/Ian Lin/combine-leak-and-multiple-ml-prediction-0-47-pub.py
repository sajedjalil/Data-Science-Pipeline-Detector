"""Here I provide the row time series, and 2 of my ML prediction
(both of them are from single LightGBM model), and then combine them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from bistiming import SimpleTimer
import yaml


def combine_leak(row_time_series_path, extra_predictions):
    with open(row_time_series_path) as fp:
        row_time_series = yaml.safe_load(fp)['row_time_series']
    train_df = pd.read_csv(
        '../input/santander-value-prediction-challenge/train.csv', index_col='ID')
    test_df = pd.read_csv(
        '../input/santander-value-prediction-challenge/test.csv', index_col='ID')
    test_id_s = pd.Series(test_df.index)

    label_series_features = [
        'f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe',
        '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0',
        '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d',
        '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45',
        '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
    label_series_df = (train_df.loc[:, label_series_features]
                       .append(test_df.loc[:, label_series_features]))
    print('label_series_df.shape:', label_series_df.shape)

    leaky_label_dfs = []
    for time_series_i, time_series in enumerate(row_time_series):
        time_to_label_dict = {}
        index_to_label_dict = {}
        df = label_series_df.loc[time_series['row_ids']]
        time_offset = 0
        for distance, (row_idx, row_s) in zip([0] + time_series['distances'], df.iterrows()):
            time_offset += distance
            # build all labels using time as key
            for i, label_feature in enumerate(row_s):
                label_time = time_offset + i + 2
                if label_time in time_to_label_dict:
                    assert time_to_label_dict[label_time] == label_feature
                else:
                    time_to_label_dict[label_time] = label_feature

            # fill the label of corresponding rows
            if time_offset in time_to_label_dict:
                index_to_label_dict[row_idx] = time_to_label_dict[time_offset]

        # check
        not_match_idx = df.index[~df.index.isin(index_to_label_dict)].tolist()
        if time_series['distances'][0] != 1 and len(not_match_idx) > 1:
            raise ValueError('Not matched unexpectedly: %s' % not_match_idx)

        leaky_label_s = pd.Series(index_to_label_dict)
        leaky_label_dfs.append(
            pd.DataFrame({'leaky_label': leaky_label_s, 'from_series': time_series_i}))

    leaky_label_df = pd.concat(leaky_label_dfs)
    # remove 0 leak for the assumption "no label is 0"
    print("row index of 0 leak:", leaky_label_df.index[leaky_label_df['leaky_label'] == 0].tolist())
    leaky_label_df = leaky_label_df[leaky_label_df['leaky_label'] != 0]
    print("leaky_label_df.shape:", leaky_label_df.shape)

    # some analysis to make sure the result is reasonable
    train_leaky_label_df = leaky_label_df.join(train_df['target'], how='inner')
    train_time_series_idx = train_leaky_label_df['from_series'].unique()
    train_time_series = [row_time_series[idx] for idx in train_time_series_idx]
    test_leaky_label_df = leaky_label_df.loc[leaky_label_df.index.isin(test_id_s)]
    test_time_series_idx = test_leaky_label_df['from_series'].unique()
    test_time_series = [row_time_series[idx] for idx in test_time_series_idx]
    print("train_leaky_label_df.shape:", train_leaky_label_df.shape)
    print("test_leaky_label_df.shape:", test_leaky_label_df.shape)
    print('accuracy on training data:',
          (train_leaky_label_df['leaky_label'] == train_leaky_label_df['target']).mean())
    print('number of training series:', len(train_time_series))
    print('number of testing series:', len(test_time_series))
    print('intersection of train / test series:',
          len(set(test_leaky_label_df['from_series']) & set(train_leaky_label_df['from_series'])))

    # prepare test_pred with leak filled
    print("filling leak prediction")
    test_pred = test_leaky_label_df['leaky_label'].reindex(test_id_s)
    print("filled: {}, na left: {}".format(test_pred.notnull().sum(),
                                           test_pred.isnull().sum()))

    # fill extra prediction into test_pred
    test_pred = fill_extra_predictions(extra_predictions, test_pred, test_df, train_df,
                                       label_series_df)
    write_prediction(test_id_s, test_pred, 'test_pred.csv')


def fill_extra_predictions(extra_predictions, test_pred, test_df, train_df, label_series_df):
    if extra_predictions is not None:
        for extra_prediction in extra_predictions:
            print("filling prediction with:", extra_prediction)

            if extra_prediction == 'train_label_log1p_mean':
                extra_pred = np.expm1(np.log1p(train_df['target']).mean())
            elif extra_prediction == 'label_series_log1p_mean_each_row':
                extra_pred = np.expm1(
                    np.nanmean(
                        np.log1p(label_series_df.replace({0.: np.nan})),
                        axis=1,
                    )
                )
                extra_pred[np.isnan(extra_pred)] = 0.
                extra_pred = pd.Series(extra_pred, index=label_series_df.index)
            elif extra_prediction == 'log1p_mean_each_row':
                extra_pred = np.expm1(
                    np.nanmean(
                        np.log1p(test_df.replace({0.: np.nan})),
                        axis=1,
                    )
                )
                extra_pred = pd.Series(extra_pred, index=test_df.index)
            else:
                pred_path = Path(extra_prediction).expanduser()
                extra_pred = pd.read_csv(pred_path, index_col='ID')['target']

            n_na_before_fill = test_pred.isnull().sum()
            test_pred = test_pred.fillna(extra_pred)
            n_na_after_fill = test_pred.isnull().sum()
            if n_na_before_fill == n_na_after_fill:
                raise ValueError("No any na filled by {}!!".format(extra_prediction))

            print("filled: {}, na left: {}".format(n_na_before_fill - n_na_after_fill,
                                                   n_na_after_fill))
    # check if nan in test_pred
    if test_pred.isnull().any():
        raise ValueError("still {} na left in prediction".format(test_pred.isnull().sum()))
    return test_pred
    
def write_prediction(data_id, pred, path):
    path = Path(path)
    pred_df = pd.DataFrame({'target': pred}, index=data_id)
    with SimpleTimer(f"Writing prediction to {path}", end_in_new_line=False):
        pred_df.to_csv(path, float_format="%.10g")
        

# 24.yml is the best row time series I have found.
# test_pred_144.csv is for filling those rows that have all-zero
# in the 40 label-related columns.
# test_pred_179.csv is for the rest rows.
combine_leak('../input/svpc-additional-data/24.yml',
             ['../input/svpc-additional-data/test_pred_144.csv',
              '../input/svpc-additional-data/test_pred_179.csv'])