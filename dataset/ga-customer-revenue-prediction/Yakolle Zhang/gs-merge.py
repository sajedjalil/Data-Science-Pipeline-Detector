import os
import sys

import pandas as pd


def merge(pred_path, sub_data_dir, submit_id):
    print(submit_id)
    id_col = 'fullVisitorId'
    target_col = 'PredictedLogRevenue'
    submission = pd.read_csv(os.path.join(sub_data_dir, 'sample_submission_v2.csv'), dtype={id_col: 'str'},
                             index_col=id_col)
    print(f'sample_submission: {submission.shape}')
    submission[target_col] = 0.0047795
    submission.to_csv(f'{submit_id}_m0.csv', index=True, float_format='%.5f')

    pred = pd.read_csv(pred_path, dtype={id_col: 'str'}, index_col=id_col)
    submission[target_col] += pred[target_col]
    submission[target_col] /= 2
    submission.to_csv(f'{submit_id}_m05.csv', index=True, float_format='%.5f')


def run():
    # sys.stderr = sys.stdout = open(os.path.join('merge_log'), 'w')

    root_dir = '../input'
    sub_data_dir = os.path.join(root_dir, 'ga-customer-revenue-prediction')

    for submit_id in ['avg', 'blend', 'average']:
        pred_path = os.path.join(root_dir, 'gs-combine', f'{submit_id}.csv')
        merge(pred_path, sub_data_dir, submit_id)


if __name__ == '__main__':
    run()
