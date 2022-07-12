import os
import time
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib

oof_seed = 1
pred_type_id = 8
round_table = {4: [1, 1, 1, 1, 1], 8: [1, 1, 1, 1, 1]}
round_nums = round_table[pred_type_id]
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]


@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def submit(pred, sub_data_dir, submit_id):
    with timer('submit'):
        submission = pd.read_csv(os.path.join(sub_data_dir, 'sample_submission_v2.csv'), dtype={'fullVisitorId': 'str'})
        submission['PredictedLogRevenue'] = pred
        submission.to_csv(f'{submit_id}.csv', index=False, float_format='%.5f')


def get_res_data(tv_round, round_num):
    res_data_dir = f'../input/gs-nn-uv-train-{oof_seed}-{pred_type_id}-{round_num}'
    model_id = f'rtnn_{oof_seed}_{pred_type_id}_{tv_round}'
    ty = joblib.load(os.path.join(res_data_dir, f'{model_id}_ty'))
    tp = joblib.load(os.path.join(res_data_dir, f'{model_id}_tp'))
    vy = joblib.load(os.path.join(res_data_dir, f'{model_id}_vy'))
    vp = joblib.load(os.path.join(res_data_dir, f'{model_id}_vp'))
    oy = joblib.load(os.path.join(res_data_dir, f'{model_id}_oy'))
    op = joblib.load(os.path.join(res_data_dir, f'{model_id}_op'))
    p = joblib.load(os.path.join(res_data_dir, f'{model_id}_p'))
    return ty, tp, vy, vp, oy, op, p


def ensemble(ensemble_model_id):
    oy, eop, ep = None, None, None
    for i, round_num in enumerate(round_nums):
        tv_round = i + 1
        ty, tp, vy, vp, oy, op, p = get_res_data(tv_round, round_num)
        t_score = rmse(ty, tp)
        v_score = rmse(vy, vp)
        o_score = rmse(oy, op)
        print(f'tv_round({tv_round}), round_num({round_num}): t_rmse={t_score}, v_rmse={v_score}, o_rmse={o_score}')

        if not i:
            eop, ep = op, p
        else:
            eop += op
            ep += p

    tv_num = len(round_nums)
    eop /= tv_num
    joblib.dump(eop, f'{ensemble_model_id}_op', compress=('gzip', 3))
    joblib.dump(oy, f'{ensemble_model_id}_oy', compress=('gzip', 3))
    print(f'o_rmse={rmse(oy, eop)}')
    ep /= tv_num
    joblib.dump(ep, f'{ensemble_model_id}_p', compress=('gzip', 3))
    return ep


def run():
    # sys.stderr = sys.stdout = open(os.path.join('nn_uv_ensemble_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-nn-src-data'
    print(os.listdir(data_dir))
    sub_data_dir = '../input/ga-customer-revenue-prediction'
    print(os.listdir(sub_data_dir))

    ensemble_model_id = f'nn_uv_ensemble_{oof_seed}_{pred_type_id}'
    p = ensemble(ensemble_model_id)

    submit(p, sub_data_dir, ensemble_model_id)


if __name__ == '__main__':
    run()
