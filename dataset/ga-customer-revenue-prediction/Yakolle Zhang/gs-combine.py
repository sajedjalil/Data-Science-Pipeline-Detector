import os
import sys

import pandas as pd

pred_num = 5

 
def combine(pred_paths, sub_data_dir, submit_id):
    print(submit_id)
    id_col = 'fullVisitorId'
    target_col = 'PredictedLogRevenue'
    submission = pd.read_csv(os.path.join(sub_data_dir, 'sample_submission_v2.csv'), dtype={id_col: 'str'},
                             index_col=id_col)
    print(f'sample_submission: {submission.shape}')
    submission[target_col] = 0.0
    for pred_path in pred_paths:
        pred = pd.read_csv(pred_path, dtype={id_col: 'str'}, index_col=id_col)
        submission[target_col] += pred[target_col]
    submission[target_col] /= len(pred_paths)

    submission.to_csv(f'{submit_id}.csv', index=True, float_format='%.5f')


def run():
    # sys.stderr = sys.stdout = open(os.path.join('combine_log'), 'w')

    root_dir = '../input'
    sub_data_dir = os.path.join(root_dir, 'ga-customer-revenue-prediction')

    for submit_id in ['avg', 'blend', 'average']:
        pred_paths = [os.path.join(root_dir, f'gs-blend-{i+1}', f'{submit_id}_{i+1}.csv') for i in range(pred_num)]
        combine(pred_paths, sub_data_dir, submit_id)


if __name__ == '__main__':
    run()
