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
from sklearn.model_selection import ShuffleSplit


# --------------------------------------------------util-----------------------------------------------------
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


# --------------------------------------------------lgb_uv_ensemble-----------------------------------------------------
oof_seed = 1
pred_type_id = 1
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]
pred_max_val = np.log1p(5.1e10)
need_indi = True


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def submit(pred, sub_data_dir, submit_id):
    with timer('submit'):
        submission = pd.read_csv(os.path.join(sub_data_dir, 'sample_submission_v2.csv'), dtype={'fullVisitorId': 'str'})
        submission['PredictedLogRevenue'] = pred
        submission.to_csv(f'{submit_id}.csv', index=False, float_format='%.5f')


def get_indi_pred_type_ids():
    return [1, 7, 13]


def get_indicators(root_dir='../input'):
    in_indi_df, out_indi_df, ts_indi_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with timer('load classification indi'):
        indi_type_id = 3
        for i in get_indi_pred_type_ids():
            indi_id = f'indi_{oof_seed}_{i}_{indi_type_id}'
            indi_dir = f'gs-{"lgb" if i<=6 else "nn"}-indi-train-{oof_seed}-{i}-{indi_type_id}'

            in_indi_df[f'zp_{i}_prob'] = joblib.load(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'))
            out_indi_df[f'zp_{i}_prob'] = joblib.load(os.path.join(root_dir, indi_dir, f'{indi_id}_op'))
            ts_indi_df[f'zp_{i}_prob'] = joblib.load(os.path.join(root_dir, indi_dir, f'{indi_id}_p'))
            print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
            gc.collect()

    cols = sorted(ts_indi_df.columns)
    return in_indi_df[cols], out_indi_df[cols], ts_indi_df[cols]


def get_oof_data(data_dir, in_indi_df, out_indi_df):
    with timer('get train src data'):
        train_df = pd.read_pickle(os.path.join(data_dir, 'train_df'), compression='gzip')
        tr_x_text = joblib.load(os.path.join(data_dir, 'tr_x_text'))
        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        print(f'train_df: {train_df.shape}, tr_x_text: {tr_x_text.shape}')

    with timer('log transform'):
        embed_cols = [col for col in train_df.columns if '_price_w_' in col or '_price_u_' in col
                      or 'Revenue_w_' in col or 'Revenue_u_' in col or 'target_u_' in col]
        print(f'embed_cols({len(embed_cols)}): {embed_cols}')
        for col in embed_cols:
            train_df[col] = np.log1p(train_df[col])
        gc.collect()

    with timer('oof split'):
        iind, oind = next(ShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(train_df))
        in_df = train_df.take(iind).reset_index(drop=True)
        in_x_text = tr_x_text[iind]
        print(f'in_df: {in_df.shape}, in_x_text: {in_x_text.shape}')

        out_df = train_df.take(oind).reset_index(drop=True)
        out_x_text = tr_x_text[oind]
        print(f'out_df: {out_df.shape}, out_x_text: {out_x_text.shape}')
        del train_df, tr_x_text, iind, oind
        gc.collect()

    if in_indi_df is not None:
        with timer('join indicators'):
            in_df = in_df.join(in_indi_df, how='inner')
            out_df = out_df.join(out_indi_df, how='inner')
            print(f'in_df: {in_df.shape}, out_df: {out_df.shape}')

    with timer('combine features'):
        iy = np.log1p(in_df.target.values)
        ix = in_df.drop(['fullVisitorId', 'target'], axis=1).values
        ix = combine_features([ix, in_x_text]) if 'plain' != text_type else ix
        del in_df, in_x_text
        gc.collect()

        oy = np.log1p(out_df.target.values)
        ox = out_df.drop(['fullVisitorId', 'target'], axis=1).values
        ox = combine_features([ox, out_x_text]) if 'plain' != text_type else ox
        del out_df, out_x_text
        gc.collect()
        print(f'ix: {ix.shape}, ox: {ox.shape}')
        print(f'iy(>0): {np.sum(iy>0)}, oy(>0): {np.sum(oy>0)}')

    return ix, iy, ox, oy


def get_test_data(data_dir, ts_indi_df):
    with timer('get test src data'):
        test_df = pd.read_pickle(os.path.join(data_dir, 'test_df'), compression='gzip')
        ts_x_text = joblib.load(os.path.join(data_dir, 'ts_x_text'))
        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        print(f'test_df: {test_df.shape}, ts_x_text: {ts_x_text.shape}')

    with timer('log transform'):
        embed_cols = [col for col in test_df.columns if '_price_w_' in col or '_price_u_' in col
                      or 'Revenue_w_' in col or 'Revenue_u_' in col or 'target_u_' in col]
        for col in embed_cols:
            test_df[col] = np.log1p(test_df[col])
        gc.collect()

    if ts_indi_df is not None:
        with timer('join indicators'):
            test_df = test_df.join(ts_indi_df, how='inner')
            print(f'test_df: {test_df.shape}')

    with timer('combine features'):
        ts_x = test_df.drop('fullVisitorId', axis=1).values
        ts_x = combine_features([ts_x, ts_x_text]) if 'plain' != text_type else ts_x
        print(f'ts_x: {ts_x.shape}')
        del test_df, ts_x_text
        gc.collect()

    return ts_x


def get_tv_data(x, y, tv_seed):
    with timer('tv split'):
        tind, vind = next(ShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=tv_seed).split(x))
        tx, ty, vx, vy = x[tind], y[tind], x[vind], y[vind]
        print(f'tx: {tx.shape}, vx: {vx.shape}')
        print(f'ty(>0): {np.sum(ty>0)}, vy(>0): {np.sum(vy>0)}')

    return tx, ty, vx, vy


def learn(data, tv_seed, model_dir='.'):
    ix, iy, ox, oy = data
    tx, ty, vx, vy = get_tv_data(ix, iy, tv_seed)

    with timer('train'):
        params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4,
                  'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.8,
                  'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
        params.update(dict([('learning_rate', 0.01), ('num_leaves', 16), ('max_depth', 10), ('min_data', 25),
                            ('bagging_fraction', 0.9), ('feature_fraction', 1.0), ('lambda_l1', 0.8),
                            ('lambda_l2', 0.4)]))

        model = lgb.train(params, lgb.Dataset(tx, label=ty), 100000, [lgb.Dataset(vx, label=vy)],
                          early_stopping_rounds=200, verbose_eval=200)
        model_file_name = write_model(model, model_dir)
        print(f"tv_seed: {tv_seed}, {model_file_name}'s best iteration: {model.best_iteration}")

    data = tx, ty, vx, vy, ox, oy
    op, t_rmse, v_rmse, o_rmse = validation(data, model, model_dir, model_file_name)

    return model, op, t_rmse, v_rmse, o_rmse


def validation(data, model, model_dir, model_file_name):
    tx, ty, vx, vy, ox, oy = data

    with timer('validation'):
        tp = np.clip(model.predict(tx), 0, pred_max_val)
        del tx
        gc.collect()
        t_rmse = rmse(ty, tp)
        joblib.dump(tp, os.path.join(model_dir, f'{model_file_name}_tp'), compress=('gzip', 3))
        joblib.dump(ty, os.path.join(model_dir, f'{model_file_name}_ty'), compress=('gzip', 3))

        vp = np.clip(model.predict(vx), 0, pred_max_val)
        del vx
        gc.collect()
        v_rmse = rmse(vy, vp)
        joblib.dump(vp, os.path.join(model_dir, f'{model_file_name}_vp'), compress=('gzip', 3))
        joblib.dump(vy, os.path.join(model_dir, f'{model_file_name}_vy'), compress=('gzip', 3))

        op = np.clip(model.predict(ox), 0, pred_max_val)
        del ox
        gc.collect()
        o_rmse = rmse(oy, op)
        joblib.dump(op, os.path.join(model_dir, f'{model_file_name}_op'), compress=('gzip', 3))
        joblib.dump(oy, os.path.join(model_dir, f'{model_file_name}_oy'), compress=('gzip', 3))

        print(f't_rmse: {t_rmse}, v_rmse: {v_rmse}, o_rmse: {o_rmse}')

        return op, t_rmse, v_rmse, o_rmse


def ensemble(data, ensemble_model_id, tv_num=5):
    ix, iy, ox, oy = data
    t_rmses, v_rmses, o_rmses = [], [], []
    ops = []
    num_rounds = []
    models = []
    for i in range(1, tv_num + 1):
        tv_seed = oof_seed * 1000000 + pred_type_id * 10000 + i * 100
        model, op, t_rmse, v_rmse, o_rmse = learn((ix, iy, ox, oy), tv_seed)
        models.append(model)
        t_rmses.append(t_rmse)
        v_rmses.append(v_rmse)
        o_rmses.append(o_rmse)
        num_rounds.append(model.best_iteration)
        ops.append(op)

    op = np.mean(ops, axis=0)
    joblib.dump(op, f'{ensemble_model_id}_op', compress=('gzip', 3))
    joblib.dump(oy, f'{ensemble_model_id}_oy', compress=('gzip', 3))
    o_rmse = rmse(oy, op)
    print(f'o_rmse: {o_rmse}')

    print(f'num_rounds: {num_rounds}')
    print(f't_rmses: mean={np.mean(t_rmses)}, std={np.std(t_rmses)},',
          f'min={np.min(t_rmses)}, max={np.max(t_rmses)}, {t_rmses}')
    print(f'v_rmses: mean={np.mean(v_rmses)}, std={np.std(v_rmses)},',
          f'min={np.min(v_rmses)}, max={np.max(v_rmses)}, {v_rmses}')
    print(f'o_rmses: mean={np.mean(o_rmses)}, std={np.std(o_rmses)},',
          f'min={np.min(o_rmses)}, max={np.max(o_rmses)}, {o_rmses}')

    return models


def run():
    sys.stderr = sys.stdout = open(os.path.join('lgb_uv_ensemble_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-src-data'
    print(os.listdir(data_dir))
    sub_data_dir = '../input/ga-customer-revenue-prediction'
    print(os.listdir(sub_data_dir))

    if need_indi:
        in_indi_df, out_indi_df, ts_indi_df = get_indicators()
    else:
        in_indi_df, out_indi_df, ts_indi_df = None, None, None
    ix, iy, ox, oy = get_oof_data(data_dir, in_indi_df, out_indi_df)
    del in_indi_df, out_indi_df
    gc.collect()
    data = ix, iy, ox, oy
    ensemble_model_id = f'lgb_uv_ensemble_{oof_seed}_{pred_type_id}'
    models = ensemble(data, ensemble_model_id)
    del ix, iy, ox, oy, data
    gc.collect()

    ts_x = get_test_data(data_dir, ts_indi_df)
    ps = []
    for model in models:
        ps.append(np.clip(model.predict(ts_x), 0, pred_max_val))
    p = np.mean(ps, axis=0)
    joblib.dump(p, f'{ensemble_model_id}_p', compress=('gzip', 3))

    submit(p, sub_data_dir, ensemble_model_id)


if __name__ == '__main__':
    run()
