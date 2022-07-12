import gc
import os
import random
import shutil
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import keras.backend as bk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Embedding, Flatten, Dense, BatchNormalization, Activation, Dropout, Lambda
from keras.layers import concatenate
from keras.optimizers import Adam
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# from ml2lm.calc.model.units.FMLayer import FMLayer
# from ml2lm.calc.model.units.SegTriangleLayer import SegTriangleLayer


sys.path.append('../input/nn-layers')
from FMLayer import FMLayer
from SegTriangleLayer import SegTriangleLayer
# --------------------------------------------------util-----------------------------------------------------
@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


# --------------------------------------------------cv_util-----------------------------------------------------
def kfold(data, n_splits=3, shuffle=True, random_state=0):
    return list(KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(data[0]))


def group_kfold(data, n_splits=3, shuffle=True, random_state=0):
    x, y, selector = data[0], data[1], data[2]
    col = selector[data[3]].reset_index(drop=True) if len(data) > 3 else pd.Series(selector)
    cnt = sorted(Counter(col).items(), key=lambda pair: (pair[1], pair[0]))

    part_size = len(cnt) // 3
    s1 = np.array([k for k, v in cnt[: part_size]])
    part1 = [(col.loc[col.isin(s1[tind])].index, col.loc[col.isin(s1[vind])].index) for tind, vind in
             KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(s1)]
    s2 = np.array([k for k, v in cnt[part_size: part_size * 2]])
    part2 = [(col.loc[col.isin(s2[tind])].index, col.loc[col.isin(s2[vind])].index) for tind, vind in
             KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(s2)]
    s3 = np.array([k for k, v in cnt[part_size * 2:]])
    part3 = [(col.loc[col.isin(s3[tind])].index, col.loc[col.isin(s3[vind])].index) for tind, vind in
             KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(s3)]

    return [(np.concatenate((tind, part2[i][0], part3[i][0]), axis=None),
             np.concatenate((vind, part2[i][1], part3[i][1]), axis=None)) for i, (tind, vind) in enumerate(part1)]


# --------------------------------------------------nn_util-----------------------------------------------------
def init_tensorflow():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)
    bk.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


def set_seed(_seed=10000):
    os.environ['PYTHONHASHSEED'] = str(_seed + 6)
    np.random.seed(_seed + 7)
    random.seed(_seed + 8)
    tf.set_random_seed(_seed + 9)


def read_weights(model, weights_path):
    model.load_weights(weights_path)
    return model


def get_seg_num(val_cnt, shrink_factor=0.5, max_seg_dim=None):
    seg_dim = max(2, int(np.sqrt(val_cnt * shrink_factor)))

    seg_dim = seg_dim if max_seg_dim is None else max_seg_dim
    return seg_dim


def get_seg_num_by_value(x, precision=4, shrink_factor=0.5):
    val_mean = np.mean(np.abs(x))
    cur_precision = np.round(np.log10(val_mean))
    x = (x * 10 ** (precision - cur_precision)).astype(np.int64)
    val_cnt = len(np.unique(x))
    return get_seg_num(val_cnt, shrink_factor=shrink_factor)


def add_dense(x, units, bn=True, activation='relu', dropout=0.2):
    x = Dense(units)(x)
    if bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)
    return x


def shrink(dim, shrink_factor):
    if dim > 10:
        return max(10, int(dim * shrink_factor))
    return dim


def get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=1.0):
    embeds = []
    for i, in_dim in enumerate(cat_in_dims):
        embed = Lambda(lambda cats: cats[:, i, None])(cat_input)
        embed = Embedding(in_dim, shrink(cat_out_dims[i], shrink_factor))(embed)
        embeds.append(Flatten()(embed))
    return embeds


def get_segments(seg_input, seg_out_dims, shrink_factor=1.0, seg_type=0, seg_input_val_range=(0, 1)):
    segments = []
    for i, out_dim in enumerate(seg_out_dims):
        segment = Lambda(lambda segs: segs[:, i, None])(seg_input)
        segment = SegTriangleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range)(segment)
        segments.append(segment)
    return segments


# --------------------------------------------------tnn-----------------------------------------------------
def get_simple_linear_output(flat, name=None):
    flat = add_dense(flat, 64, bn=False, dropout=0.05)
    return Dense(1, name=name)(flat)


def compile_default_mse_output(outputs, oh_input=None, cat_input=None, seg_input=None, num_input=None,
                               loss_weights=None):
    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='mse', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_simple_sigmoid_output(flat, name=None):
    flat = add_dense(flat, 64, bn=False, dropout=0.05)
    return Dense(1, activation='sigmoid', name=name)(flat)


def compile_default_bce_output(output, oh_input=None, cat_input=None, seg_input=None, num_input=None,
                               loss_weights=None):
    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)

    dnn = Model(inputs, output)
    dnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_tnn_block(block_no, get_output=get_simple_linear_output, oh_input=None, cat_input=None, seg_input=None,
                  num_input=None, pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                  num_segs=None, seg_type=0, seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=50,
                  shrink_factor=1.0, use_fm=False, seg_flag=True, add_seg_src=True, seg_num_flag=True,
                  x=None, get_extra_layers=None, embed_dropout=0.2, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2,
                  fm_activation=None, hidden_units=320, hidden_dropout=0.2):
    embeds = [Flatten()(Embedding(3, 2)(oh_input))] if oh_input is not None else []
    if cat_input is not None:
        embeds += get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=shrink_factor ** block_no)
    embeds = Dropout(embed_dropout)(concatenate(embeds)) if embeds else None

    segments = get_segments(seg_input, seg_out_dims, shrink_factor=shrink_factor ** block_no, seg_type=seg_type,
                            seg_input_val_range=seg_x_val_range) if seg_flag and seg_input is not None else[]
    segments += get_segments(num_input, num_segs, shrink_factor=shrink_factor ** block_no, seg_type=seg_type,
                             seg_input_val_range=seg_x_val_range) if seg_num_flag and num_input is not None else []

    if pre_output is not None:
        seg_y_dim = shrink(seg_y_dim, shrink_factor ** block_no)
        segment = SegTriangleLayer(seg_y_dim, input_val_range=seg_y_val_range)(pre_output)
        segments.append(segment)
    segments = Dropout(seg_dropout)(concatenate(segments)) if segments else None

    feats = [embeds] if embeds is not None else []
    if segments is not None:
        feats.append(segments)
    if seg_input is not None and (add_seg_src or not seg_flag):
        feats.append(seg_input)
    if num_input is not None:
        feats.append(num_input)
    if pre_output is not None:
        feats.append(pre_output)
    feats = concatenate(feats) if len(feats) > 1 else feats[0]

    extra_feats = get_extra_layers(x, feats) if get_extra_layers is not None else None

    if use_fm:
        fm = FMLayer(fm_dim, activation=fm_activation)(feats)
        fm = Dropout(fm_dropout)(fm)
        flat = concatenate([feats, fm])
    else:
        flat = feats

    flat = concatenate([flat, extra_feats]) if extra_feats is not None else flat

    flat = add_dense(flat, hidden_units, bn=True, dropout=hidden_dropout)
    tnn_block = get_output(flat, name=f'out{block_no}')
    return tnn_block


def get_tnn_model(x, get_output=get_simple_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                  cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), use_fm=False,
                  seg_flag=True, add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2,
                  seg_dropout=0.2, fm_dim=320, fm_dropout=0.2, fm_activation=None, hidden_units=320,
                  hidden_dropout=0.2):
    oh_input = Input(shape=[x['ohs'].shape[1]], name='ohs') if 'ohs' in x else None
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    tnn = get_tnn_block(0, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                        num_input=num_input, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                        seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type,
                        seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
                        seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers, embed_dropout=embed_dropout,
                        seg_dropout=seg_dropout, fm_dim=fm_dim, fm_dropout=fm_dropout, fm_activation=fm_activation,
                        hidden_units=hidden_units, hidden_dropout=hidden_dropout)
    tnn = compile_func(tnn, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input, num_input=num_input)
    return tnn


# --------------------------------------------------nn_uv_train-----------------------------------------------------
id_col = 'fullVisitorId'
scale_max_val = 20
tv_num = 5

oof_seed = 1
pred_type_id = 4
round_num = 6
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]
pred_max_val = np.log1p(7.8e10)

use_fm = pred_type_id >= 7
block_num = 1
shrink_factor = 1
seg_flag = True
add_seg_src = True
seg_num_flag = False
fm_dim = 320
hidden_units = 320

lr_patience = 10
stop_patience = 15
epochs = 100
batch_size = 1024

res_shrinkage = 0.1


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def combine_features(features, batch_num=5):
    cols = []
    _batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * _batch_size: (i + 1) * _batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def get_ab_split(x, y, selector, aind, bind):
    a_selector = selector.take(aind).reset_index(drop=True)
    ax = {col: val[aind] for col, val in x.items()}
    ay = y[aind]
    b_selector = selector.take(bind).reset_index(drop=True)
    bx = {col: val[bind] for col, val in x.items()}
    by = y[bind]
    return ax, ay, a_selector, bx, by, b_selector


def get_indi_pred_type_ids():
    return [1, 4, 10, 16] + [2, 8, 14] + [3, 6, 12, 18]


def get_indicators(root_dir='../input'):
    def agg_num_by_user(df, indi_pred_type_id):
        prefix = f'zr_{indi_pred_type_id}'
        df['pred'] = np.expm1(df['pred'])
        gdf = df.groupby(id_col)['pred'].agg([np.sum, np.mean, np.std]).rename(
            columns={'sum': f'{prefix}_sum', 'mean': f'{prefix}_mean', 'std': f'{prefix}_std'}).fillna(0)
        gdf[f'{prefix}_sum'] = np.log1p(gdf[f'{prefix}_sum'])
        gdf[f'{prefix}_mean'] = np.log1p(gdf[f'{prefix}_mean'])
        gdf[f'{prefix}_std'] = np.log1p(gdf[f'{prefix}_std'])
        return gdf

    def agg_indi_by_user(df, indi_pred_type_id):
        prefix = f'zc_{indi_pred_type_id}'
        gdf = df.groupby(id_col)['pred'].agg([np.mean, np.std]).rename(
            columns={'mean': f'{prefix}_mean', 'std': f'{prefix}_std'}).fillna(0)
        return gdf

    in_indi_df, out_indi_df, ts_indi_df = None, None, None
    with timer('load regression indi'):
        indi_type_id = 1
        for i in get_indi_pred_type_ids():
            indi_id = f'indi_{oof_seed}_{i}_{indi_type_id}'
            indi_dir = f'gs-{"lgb" if i<=6 else "nn"}-indi-train-{oof_seed}-{i}-{indi_type_id}'

            in_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'), compression='gzip')
            in_indi_df = agg_num_by_user(in_selector, i) if in_indi_df is None else in_indi_df.join(
                agg_num_by_user(in_selector, i), how='inner')
            out_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_op'), compression='gzip')
            out_indi_df = agg_num_by_user(out_selector, i) if out_indi_df is None else out_indi_df.join(
                agg_num_by_user(out_selector, i), how='inner')
            ts_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_p'), compression='gzip')
            ts_indi_df = agg_num_by_user(ts_selector, i) if ts_indi_df is None else ts_indi_df.join(
                agg_num_by_user(ts_selector, i), how='inner')
            print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
            del in_selector, out_selector, ts_selector
            gc.collect()

    with timer('load classification(session) indi'):
        indi_type_id = 2
        for i in get_indi_pred_type_ids():
            indi_id = f'indi_{oof_seed}_{i}_{indi_type_id}'
            indi_dir = f'gs-{"lgb" if i<=6 else "nn"}-indi-train-{oof_seed}-{i}-{indi_type_id}'

            in_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'), compression='gzip')
            in_indi_df = in_indi_df.join(agg_indi_by_user(in_selector, i), how='inner')
            out_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_op'), compression='gzip')
            out_indi_df = out_indi_df.join(agg_indi_by_user(out_selector, i), how='inner')
            ts_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_p'), compression='gzip')
            ts_indi_df = ts_indi_df.join(agg_indi_by_user(ts_selector, i), how='inner')
            print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
            del in_selector, out_selector, ts_selector
            gc.collect()

    with timer('load classification(user) indi'):
        indi_type_id = 3
        for i in get_indi_pred_type_ids():
            indi_id = f'indi_{oof_seed}_{i}_{indi_type_id}'
            indi_dir = f'gs-{"lgb" if i<=6 else "nn"}-indi-train-{oof_seed}-{i}-{indi_type_id}'

            in_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'), compression='gzip').rename(
                columns={'pred': f'zp_{i}_prob'}).set_index(id_col)
            in_indi_df = in_indi_df.join(in_selector, how='inner')
            out_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_op'), compression='gzip').rename(
                columns={'pred': f'zp_{i}_prob'}).set_index(id_col)
            out_indi_df = out_indi_df.join(out_selector, how='inner')
            ts_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_p'), compression='gzip').rename(
                columns={'pred': f'zp_{i}_prob'}).set_index(id_col)
            ts_indi_df = ts_indi_df.join(ts_selector, how='inner')
            print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
            del in_selector, out_selector, ts_selector
            gc.collect()

    with timer('scale & collect seg infos'):
        indi_cols = sorted(ts_indi_df.columns)
        tr_indi_df = in_indi_df[indi_cols].append(out_indi_df[indi_cols], sort=False)
        ts_indi_df = ts_indi_df[indi_cols]
        del in_indi_df, out_indi_df
        gc.collect()

        indi_seg_out_dims = []
        for col in indi_cols:
            indi_seg_out_dims.append(get_seg_num_by_value(tr_indi_df[col], precision=6))

        scaler = MinMaxScaler(feature_range=(0, scale_max_val))
        tr_indi_df[indi_cols] = scaler.fit_transform(tr_indi_df).astype(np.float32)
        ts_indi_df[indi_cols] = np.clip(scaler.transform(ts_indi_df), 0, scale_max_val).astype(np.float32)
        print(f'indi_cols({len(indi_cols)})')
        gc.collect()

    return tr_indi_df, ts_indi_df, indi_seg_out_dims


def get_oof_data(data_dir, tr_indi_df, indi_seg_out_dims):
    with timer('get train src data'):
        tr_cats, tr_segs, tr_nums, tr_x_text, tr_selector_u, y = joblib.load(os.path.join(data_dir, 'trd_u'))
        tr_selector = joblib.load(os.path.join(data_dir, 'trd'))[4]
        cat_in_dims, cat_out_dims, seg_out_dims = joblib.load(os.path.join(data_dir, 'md_u'))
        seg_out_dims += indi_seg_out_dims
        md = cat_in_dims, cat_out_dims, seg_out_dims
        print(f'tr_cats: {tr_cats.shape}, tr_segs: {tr_segs.shape}, tr_nums: {tr_nums.shape},',
              f'tr_x_text: {tr_x_text.shape}, y: {y.shape}, seg_out_dims({len(seg_out_dims)})')

        tr_indi_df = tr_selector_u.join(tr_indi_df, on=id_col, how='inner')
        tr_indis = tr_indi_df.drop(id_col, axis=1).values
        print(f'tr_indis: {tr_indis.shape}')
        tr_segs = np.hstack([tr_segs, tr_indis])
        print(f'tr_segs: {tr_segs.shape}')

        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        tr_nums = combine_features([tr_nums, tr_x_text]) if 'plain' != text_type else tr_nums
        print(f'tr_nums: {tr_nums.shape}')

        x = {'cats': tr_cats, 'segs': tr_segs, 'nums': tr_nums}
        del tr_cats, tr_segs, tr_nums, tr_x_text, tr_indi_df, tr_indis
        gc.collect()

    with timer('oof split'):
        iind, oind = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(
            tr_selector, groups=tr_selector[id_col]))
        ids = tr_selector.take(iind)[id_col]
        iind = tr_selector_u.loc[tr_selector_u[id_col].isin(ids)].index.values
        ids = tr_selector.take(oind)[id_col]
        oind = tr_selector_u.loc[tr_selector_u[id_col].isin(ids)].index.values
        tr_selector = tr_selector_u

        ix, iy, in_selector, ox, oy, out_selector = get_ab_split(x, y, tr_selector, iind, oind)
        print(f'iy: {iy.shape}, oy: {oy.shape}')
        del tr_selector, tr_selector_u, iind, oind, x, y
        gc.collect()

    return ix, iy, in_selector, ox, oy, out_selector, md


def get_test_data(data_dir, ts_indi_df):
    with timer('get test src data'):
        ts_cats, ts_segs, ts_nums, ts_x_text, ts_selector = joblib.load(os.path.join(data_dir, 'tsd_u'))
        print(f'ts_cats: {ts_cats.shape}, ts_segs: {ts_segs.shape}, ts_nums: {ts_nums.shape},',
              f'ts_x_text: {ts_x_text.shape}')

        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        ts_nums = combine_features([ts_nums, ts_x_text]) if 'plain' != text_type else ts_nums
        print(f'ts_nums: {ts_nums.shape}')

        ts_indi_df = ts_selector.join(ts_indi_df, on=id_col, how='inner')
        ts_indis = ts_indi_df.drop(id_col, axis=1).values
        print(f'ts_indis: {ts_indis.shape}')
        ts_segs = np.hstack([ts_segs, ts_indis])
        print(f'ts_segs: {ts_segs.shape}')

        ts_x = {'cats': ts_cats, 'segs': ts_segs, 'nums': ts_nums}
        del ts_cats, ts_segs, ts_nums, ts_x_text, ts_indi_df, ts_indis
        gc.collect()

    return ts_x, ts_selector


def get_tv_data(x, y, selector, tv_seed):
    tind, vind = next(ShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=tv_seed).split(selector))
    tx, ty, t_selector, vx, vy, v_selector = get_ab_split(x, y, selector, tind, vind)
    del selector, tind, vind, x, y
    gc.collect()
    return tx, ty, t_selector, vx, vy, v_selector


def get_last_res_data(tv_round):
    last_data_dir = f'../input/gs-nn-uv-train-{oof_seed}-{pred_type_id}-{round_num-1}'
    model_id = f'rtnn_{oof_seed}_{pred_type_id}_{tv_round}'
    ty = joblib.load(os.path.join(last_data_dir, f'{model_id}_ty'))
    tp = joblib.load(os.path.join(last_data_dir, f'{model_id}_tp'))
    vy = joblib.load(os.path.join(last_data_dir, f'{model_id}_vy'))
    vp = joblib.load(os.path.join(last_data_dir, f'{model_id}_vp'))
    oy = joblib.load(os.path.join(last_data_dir, f'{model_id}_oy'))
    op = joblib.load(os.path.join(last_data_dir, f'{model_id}_op'))
    return ty - res_shrinkage * tp, vy - res_shrinkage * vp, oy - res_shrinkage * op


def learn(data):
    ix, iy, in_selector, ox, oy, out_selector, (cat_in_dims, cat_out_dims, seg_out_dims), ts_x, ts_selector = data

    params = {'get_output': get_simple_linear_output, 'compile_func': compile_default_mse_output,
              'cat_in_dims': cat_in_dims, 'cat_out_dims': cat_out_dims, 'seg_out_dims': seg_out_dims,
              'seg_x_val_range': (0, scale_max_val), 'use_fm': use_fm, 'seg_flag': seg_flag,
              'add_seg_src': add_seg_src, 'seg_num_flag': seg_num_flag, 'fm_dim': fm_dim, 'fm_activation': None,
              'hidden_units': hidden_units}

    c = tv_num
    while not os.path.exists(f'rtnn_{oof_seed}_{pred_type_id}_{c}_p') and c > 0:
        c -= 1
    c += 1

    start_time = int(time.time())
    end_time = start_time + 6 * 3600 - 5 * 60
    last_time = start_time
    spend_times = []
    for i in range(c, tv_num + 1):
        tv_seed = oof_seed * 1000000 + pred_type_id * 10000 + 2 * i * 100
        model_id = f'rtnn_{oof_seed}_{pred_type_id}_{i}'
        tx, ty, t_selector, vx, vy, v_selector = get_tv_data(ix, iy, in_selector, tv_seed)
        if round_num > 1:
            ty, vy, oy = get_last_res_data(i)
            print(f'last res data: ty({ty.shape[0]}), vy({vy.shape[0]}), oy({oy.shape[0]})')
        print(f'---------------------------------{model_id}---------------------------------')
        print(f'ty: {ty.shape}, vy: {vy.shape}, t_selector: {t_selector.shape}, v_selector: {v_selector.shape}')
        gc.collect()

        checkpointer = ModelCheckpoint(model_id, monitor='val_loss', verbose=1, save_best_only=True,
                                       save_weights_only=True, period=1)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1, min_lr=1e-5,
                                         min_delta=3e-3)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=3e-3, patience=stop_patience, verbose=1)
        model = get_tnn_model(tx, **params)

        model.fit(tx, ty, epochs=epochs, batch_size=batch_size, validation_data=(vx, vy), verbose=2,
                  callbacks=[checkpointer, lr_scheduler, early_stopper])

        validation((tx, ty, vx, vy, ox, oy, ts_x, model), model_id)

        cur_time = int(time.time())
        spend_times.append(cur_time - last_time)
        if cur_time + 2 * np.max(spend_times) - np.min(spend_times) > end_time and i < tv_num:
            exit(0)
        last_time = cur_time

    print_scores()


def print_scores():
    for i in range(1, tv_num + 1):
        model_id = f'rtnn_{oof_seed}_{pred_type_id}_{i}'
        ty = joblib.load(f'{model_id}_ty')
        tp = joblib.load(f'{model_id}_tp')
        t_score = rmse(ty, tp)

        vy = joblib.load(f'{model_id}_vy')
        vp = joblib.load(f'{model_id}_vp')
        v_score = rmse(vy, vp)

        oy = joblib.load(f'{model_id}_oy')
        op = joblib.load(f'{model_id}_op')
        o_score = rmse(oy, op)
        print(f'{model_id}: t_rmse={t_score}, v_rmse={v_score}, o_rmse={o_score}')


def validation(data, model_id, _batch_size=10000):
    tx, ty, vx, vy, ox, oy, ts_x, model = data

    with timer('validation'):
        model = read_weights(model, model_id)

        tp = np.squeeze(model.predict(tx, batch_size=_batch_size))
        tp = np.clip(tp, 0, pred_max_val)
        t_score = rmse(ty, tp)
        joblib.dump(tp, f'{model_id}_tp', compress=('gzip', 3))
        joblib.dump(ty, f'{model_id}_ty', compress=('gzip', 3))

        vp = np.squeeze(model.predict(vx, batch_size=_batch_size))
        vp = np.clip(vp, 0, pred_max_val)
        v_score = rmse(vy, vp)
        joblib.dump(vp, f'{model_id}_vp', compress=('gzip', 3))
        joblib.dump(vy, f'{model_id}_vy', compress=('gzip', 3))

        op = np.squeeze(model.predict(ox, batch_size=_batch_size))
        op = np.clip(op, 0, pred_max_val)
        o_score = rmse(oy, op)
        joblib.dump(op, f'{model_id}_op', compress=('gzip', 3))
        joblib.dump(oy, f'{model_id}_oy', compress=('gzip', 3))

        p = np.squeeze(model.predict(ts_x, batch_size=_batch_size))
        p = np.clip(p, 0, pred_max_val)
        joblib.dump(p, f'{model_id}_p', compress=('gzip', 3))
        print(f'{model_id}: t_rmse={t_score}, v_rmse={v_score}, o_rmse={o_score}')


def run():
    # reboot_dir = f'../input/gs-nn-uv-train-{oof_seed}-{pred_type_id}-{round_num}'
    # file_names = os.listdir(reboot_dir)
    # print(file_names)
    # for file_name in file_names:
    #     file_path = os.path.join(reboot_dir, file_name)
    #     if os.path.isfile(file_path):
    #         shutil.copy(file_path, '.')

    sys.stderr = sys.stdout = open(os.path.join('nn_uv_train_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-nn-src-data'
    print(os.listdir(data_dir))

    tr_indi_df, ts_indi_df, indi_seg_out_dims = get_indicators()
    ix, iy, in_selector, ox, oy, out_selector, md = get_oof_data(data_dir, tr_indi_df, indi_seg_out_dims)
    ts_x, ts_selector = get_test_data(data_dir, ts_indi_df)
    del tr_indi_df, ts_indi_df
    gc.collect()
    data = ix, iy, in_selector, ox, oy, out_selector, md, ts_x, ts_selector

    # init_tensorflow()
    learn(data)


if __name__ == '__main__':
    run()
