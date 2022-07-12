import numpy as np
import pandas as pd
import numba

@numba.jit
def smape_np(y_true, y_pred):
    tmp = (np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true)))
    tmp[np.isnan(tmp)] = 0.0
    return np.mean(tmp)


def cal_smape(ground_truth_file, prediction_file):
    print('Reading ground truth ...')
    ground_truth_df = pd.read_csv(ground_truth_file)
    print('Sorting ground truth ...')
    ground_truth_df = ground_truth_df.sort_values('Id')
    y_true = ground_truth_df['Visits'].values
    y_state = ground_truth_df['Usage'].values

    print('Reading predictions ...')
    predction_df = pd.read_csv(prediction_file)
    print('Sorting predictions ...')
    predction_df = predction_df.sort_values('Id')
    y_pred = predction_df['Visits'].values

    #remove unavailable items
    print('Removing unavailable items')
    available_cnt = len(y_true)
    del_idxes = []
    for k, state in enumerate(y_state):
        if state == 'Ignored':
            available_cnt -= 1
            del_idxes.append(k)
    print('Total items {}, available items {}'.format(len(y_true), available_cnt))

    y_true = np.delete(y_true, del_idxes)
    y_pred = np.delete(y_pred, del_idxes)

    smape = smape_np(y_true, y_pred)
    print(smape)

if __name__ == '__main__':
    #cal_smape('ground_truth/solution_11_01.csv', 'sub.csv')
    cal_smape('ground_truth/solution_11_01.csv', '/media/wb/backup/work/web_traffic_forecast/output/fusion.csv')
