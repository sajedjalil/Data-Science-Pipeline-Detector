import gc
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit, KFold

 
@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


def get_time_stamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def write_model(model, model_dir, model_file_name=None, model_id=None):
    model_id = type(model).__name__ if model_id is None else model_id
    model_file_name = f'{get_time_stamp()}_{model_id}' if model_file_name is None else model_file_name
    joblib.dump(model, os.path.join(model_dir, model_file_name))
    return model_file_name


# def kfold(data, n_splits=3, random_state=0):
#     test_size = data[0].shape[0] // n_splits + 1
#     tind, vind = next(ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + 1).split(data[0]))
#     splits = [(tind, vind)]
#     for i in range(2, n_splits):
#         tind1, vind1 = next(ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + i).split(tind))
#         tind1, vind1 = tind[tind1], tind[vind1]
#         splits.append((np.append(tind1, vind), vind1))
#         vind = np.append(vind, vind1)
#         tind = tind1
#     splits.append((vind, tind))
#
#     return splits

def kfold(data, n_splits=3, shuffle=True, random_state=0):
    return list(KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(data[0]))


# --------------------------------------------------lgb_indi_train-----------------------------------------------------
kfold_k = 5

oof_seed = 1
pred_type_id = 1
indi_type_id = 3
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def get_oof_data(data_dir):
    with timer('get train src data'):
        train_df = pd.read_pickle(os.path.join(data_dir, 'train_df'), compression='gzip')
        tr_x_text = joblib.load(os.path.join(data_dir, 'tr_x_text'))
        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        print(f'train_df: {train_df.shape}, tr_x_text: {tr_x_text.shape}')

    with timer('oof split'):
        iind, oind = next(ShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(train_df))
        in_df = train_df.take(iind)
        in_x_text = tr_x_text[iind]
        print(f'in_df: {in_df.shape}, in_x_text: {in_x_text.shape}')

        out_df = train_df.take(oind)
        out_x_text = tr_x_text[oind]
        print(f'out_df: {out_df.shape}, out_x_text: {out_x_text.shape}')
        del train_df, tr_x_text, iind, oind
        gc.collect()

    with timer('combine features'):
        iy = in_df.target.values.astype(np.bool).astype(np.int8)
        ix = in_df.drop(['fullVisitorId', 'target'], axis=1).values
        ix = combine_features([ix, in_x_text]) if 'plain' != text_type else ix
        del in_df, in_x_text
        gc.collect()

        oy = out_df.target.values.astype(np.bool).astype(np.int8)
        ox = out_df.drop(['fullVisitorId', 'target'], axis=1).values
        ox = combine_features([ox, out_x_text]) if 'plain' != text_type else ox
        del out_df, out_x_text
        gc.collect()
        print(f'ix: {ix.shape}, ox: {ox.shape}')
        print(f'iy(>0): {np.sum(iy)}, oy(>0): {np.sum(oy)}')

    return ix, iy, ox, oy


def get_test_data(data_dir):
    with timer('get test src data'):
        test_df = pd.read_pickle(os.path.join(data_dir, 'test_df'), compression='gzip')
        ts_x_text = joblib.load(os.path.join(data_dir, 'ts_x_text'))
        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        print(f'test_df: {test_df.shape}, ts_x_text: {ts_x_text.shape}')

    with timer('combine features'):
        ts_x = test_df.drop('fullVisitorId', axis=1).values
        ts_x = combine_features([ts_x, ts_x_text]) if 'plain' != text_type else ts_x
        print(f'ts_x: {ts_x.shape}')
        del test_df, ts_x_text
        gc.collect()

    return ts_x


def learn(data, model_dir='.'):
    indi_id = f'indi_{oof_seed}_{pred_type_id}_{indi_type_id}'
    ix, iy, ox, oy, ts_x = data

    with timer('train'):
        seed = oof_seed * 1000000 + pred_type_id * 10000 + indi_type_id
        fold_inds = kfold((ix, iy), kfold_k, True, seed)

        params = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'nthread': 4, 'scale_pos_weight': 10.0,
                  'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.8,
                  'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}  # , 'is_unbalance': True}
        params.update(dict([('learning_rate', 0.0025), ('num_leaves', 31), ('scale_pos_weight', 1.0), ('max_depth', 10),
                            ('min_data', 80), ('bagging_fraction', 0.7), ('feature_fraction', 0.5), ('lambda_l1', 0.8),
                            ('lambda_l2', 0.0)]))

        ip = np.zeros(iy.shape)
        ops = []
        ps = []
        t_aucs, v_aucs, o_aucs = [], [], []
        t_f1s, v_f1s, o_f1s = [], [], []
        for tind, vind in fold_inds:
            tx, ty, vx, vy = ix[tind], iy[tind], ix[vind], iy[vind]
            print(f'tx: {tx.shape}, vx: {vx.shape}')
            print(f'ty(>0): {np.sum(ty)}, vy(>0): {np.sum(vy)}')
            model = lgb.train(params, lgb.Dataset(tx, label=ty), 100000, [lgb.Dataset(vx, label=vy)],
                              early_stopping_rounds=200, verbose_eval=200)
            model_file_name = write_model(model, model_dir)
            print(f"{model_file_name}'s best iteration: {model.best_iteration}")

            tp, vp, op, p, t_auc, v_auc, o_auc, t_f1, v_f1, o_f1 = validation((tx, ty, vx, vy, ox, oy, ts_x), model)
            ip[vind] = vp
            ops.append(op)
            ps.append(p)
            t_aucs.append(t_auc)
            v_aucs.append(v_auc)
            o_aucs.append(o_auc)
            t_f1s.append(t_f1)
            v_f1s.append(v_f1)
            o_f1s.append(o_f1)
            del tx, ty, vx, vy
            gc.collect()

        print('----------------------------------------------------------------------------')
        print(f't_aucs: mean={np.mean(t_aucs)}, std={np.std(t_aucs)}, {t_aucs}')
        print(f'v_aucs: mean={np.mean(v_aucs)}, std={np.std(v_aucs)}, {v_aucs}')
        print(f'o_aucs: mean={np.mean(o_aucs)}, std={np.std(o_aucs)}, {o_aucs}')
        print(f't_f1s: mean={np.mean(t_f1s)}, std={np.std(t_f1s)}, {t_f1s}')
        print(f'v_f1s: mean={np.mean(v_f1s)}, std={np.std(v_f1s)}, {v_f1s}')
        print(f'o_f1s: mean={np.mean(o_f1s)}, std={np.std(o_f1s)}, {o_f1s}')
        print('----------------------------------------------------------------------------')

        joblib.dump(ip, f'{indi_id}_ip', compress=('gzip', 3))
        joblib.dump(np.mean(ops, axis=0), f'{indi_id}_op', compress=('gzip', 3))
        joblib.dump(np.mean(ps, axis=0), f'{indi_id}_p', compress=('gzip', 3))


def validation(data, model):
    tx, ty, vx, vy, ox, oy, ts_x = data

    with timer('validation'):
        tp = model.predict(tx)
        t_auc = metrics.roc_auc_score(ty, tp)
        t_f110 = metrics.f1_score(ty, tp > 0.10)
        t_f120 = metrics.f1_score(ty, tp > 0.20)
        t_f130 = metrics.f1_score(ty, tp > 0.30)
        t_f140 = metrics.f1_score(ty, tp > 0.40)
        t_f150 = metrics.f1_score(ty, tp > 0.50)
        t_f160 = metrics.f1_score(ty, tp > 0.60)
        t_f170 = metrics.f1_score(ty, tp > 0.70)
        t_f180 = metrics.f1_score(ty, tp > 0.80)
        t_f190 = metrics.f1_score(ty, tp > 0.90)

        vp = model.predict(vx)
        v_auc = metrics.roc_auc_score(vy, vp)
        v_f110 = metrics.f1_score(vy, vp > 0.10)
        v_f120 = metrics.f1_score(vy, vp > 0.20)
        v_f130 = metrics.f1_score(vy, vp > 0.30)
        v_f140 = metrics.f1_score(vy, vp > 0.40)
        v_f150 = metrics.f1_score(vy, vp > 0.50)
        v_f160 = metrics.f1_score(vy, vp > 0.60)
        v_f170 = metrics.f1_score(vy, vp > 0.70)
        v_f180 = metrics.f1_score(vy, vp > 0.80)
        v_f190 = metrics.f1_score(vy, vp > 0.90)

        op = model.predict(ox)
        o_auc = metrics.roc_auc_score(oy, op)
        o_f110 = metrics.f1_score(oy, op > 0.10)
        o_f120 = metrics.f1_score(oy, op > 0.20)
        o_f130 = metrics.f1_score(oy, op > 0.30)
        o_f140 = metrics.f1_score(oy, op > 0.40)
        o_f150 = metrics.f1_score(oy, op > 0.50)
        o_f160 = metrics.f1_score(oy, op > 0.60)
        o_f170 = metrics.f1_score(oy, op > 0.70)
        o_f180 = metrics.f1_score(oy, op > 0.80)
        o_f190 = metrics.f1_score(oy, op > 0.90)

        p = model.predict(ts_x)
        print(f't_auc: {t_auc}, v_auc: {v_auc}, o_auc: {o_auc}')
        print(f't_f110: {t_f110}, v_f110: {v_f110}, o_f110: {o_f110}')
        print(f't_f120: {t_f120}, v_f120: {v_f120}, o_f120: {o_f120}')
        print(f't_f130: {t_f130}, v_f130: {v_f130}, o_f130: {o_f130}')
        print(f't_f140: {t_f140}, v_f140: {v_f140}, o_f140: {o_f140}')
        print(f't_f150: {t_f150}, v_f150: {v_f150}, o_f150: {o_f150}')
        print(f't_f160: {t_f160}, v_f160: {v_f160}, o_f160: {o_f160}')
        print(f't_f170: {t_f170}, v_f170: {v_f170}, o_f170: {o_f170}')
        print(f't_f180: {t_f180}, v_f180: {v_f180}, o_f180: {o_f180}')
        print(f't_f190: {t_f190}, v_f190: {v_f190}, o_f190: {o_f190}')

    return tp, vp, op, p, t_auc, v_auc, o_auc, t_f150, v_f150, o_f150


def run():
    sys.stderr = sys.stdout = open(os.path.join('lgb_indi_train_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-src-data'
    print(os.listdir(data_dir))

    ix, iy, ox, oy = get_oof_data(data_dir)
    ts_x = get_test_data(data_dir)
    data = ix, iy, ox, oy, ts_x
    learn(data)


if __name__ == '__main__':
    run()
