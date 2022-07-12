import gc
import os
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

warnings.filterwarnings('ignore')


@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


def get_out_dim(vocab_size, scale=10, shrink_factor=0.5, max_out_dim=None):
    if vocab_size <= 10:
        out_dim = max(2, vocab_size)
    elif vocab_size <= 40:
        out_dim = max(10, int(shrink_factor * vocab_size // 2))
    else:
        out_dim = max(10, int(shrink_factor * 20), int(shrink_factor * vocab_size / np.log2(vocab_size / scale)))
    out_dim = max_out_dim if max_out_dim is not None and out_dim > max_out_dim else out_dim
    return out_dim


def get_seg_num(val_cnt, shrink_factor=0.5, max_seg_dim=None):
    seg_dim = max(2, int(np.sqrt(val_cnt * shrink_factor)))
    seg_dim = max_seg_dim if max_seg_dim is not None and seg_dim > max_seg_dim else seg_dim
    return seg_dim


def get_val_cnt(x, precision=4):
    val_mean = np.mean(np.abs(x))
    cur_precision = np.round(np.log10(val_mean))
    x = (x * 10 ** (precision - cur_precision)).astype(np.int64)
    val_cnt = len(np.unique(x))
    return val_cnt


# --------------------------------------------------get_nn_src_data-----------------------------------------------------
scale_max_val = 20
vocab_max_size = 1000


class ValueSqueezer:
    def __init__(self):
        self.value_mapper = {}

    def fit_transform(self, x):
        s = sorted(x.unique())
        self.value_mapper = dict([(ele, i + 1) for i, ele in enumerate(s)])
        return self.transform(x)

    def transform(self, x):
        unseen_value = len(self.value_mapper)
        return x.apply(lambda ele: unseen_value if ele not in self.value_mapper else self.value_mapper[ele]).astype(
            x.dtype)


def transform_data(train_df, test_df, tr_x_text, ts_x_text):
    with timer("collect cols' meta infos"):
        cols = sorted(list(test_df.columns))
        cat_cols = []
        seg_cat_cols = []
        seg_cols = []
        embed_cols = []
        for col in cols:
            if '_price_w_' in col or '_price_u_' in col or 'Revenue_w_' in col or 'Revenue_u_' in col \
                    or 'target_u_' in col:
                embed_cols.append(col)
            elif col[-1].isdigit():
                if train_df[col].nunique() > vocab_max_size:
                    seg_cat_cols.append(col)
                else:
                    cat_cols.append(col)
            else:
                seg_cols.append(col)

        print(f'cat_cols({len(cat_cols)}): {cat_cols}')
        print(f'seg_cat_cols({len(seg_cat_cols)}): {seg_cat_cols}')
        print(f'seg_cols({len(seg_cols)}): {seg_cols}')
        print(f'embed_cols({len(embed_cols)}): {embed_cols}')

    with timer('squeeze and collect cat infos'):
        cat_in_dims = []
        cat_out_dims = []
        for col in cat_cols:
            squeezer = ValueSqueezer()
            train_df[col] = squeezer.fit_transform(train_df[col])
            test_df[col] = squeezer.transform(test_df[col])

            max_val = np.max(train_df[col])
            cat_in_dims.append(max_val + 1)
            cat_out_dims.append(get_out_dim(max_val))
            gc.collect()
        tr_cats = train_df[cat_cols].values
        ts_cats = test_df[cat_cols].values

    with timer('collect segment infos'):
        seg_out_dims = []
        for col in seg_cat_cols:
            val_cnt = get_val_cnt(train_df[col], precision=6)
            seg_out_dims.append(get_out_dim(val_cnt, max_out_dim=200))

            scaler = MinMaxScaler(feature_range=(0, scale_max_val))
            train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1)).reshape(-1)
            test_df[col] = np.clip(scaler.transform(test_df[col].values.reshape(-1, 1)).reshape(-1), 0, scale_max_val)
            gc.collect()

        for col in seg_cols:
            val_cnt = get_val_cnt(train_df[col], precision=5)
            seg_out_dims.append(get_seg_num(val_cnt, max_seg_dim=100))

            scaler = MinMaxScaler(feature_range=(0, scale_max_val))
            train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1)).reshape(-1)
            test_df[col] = np.clip(scaler.transform(test_df[col].values.reshape(-1, 1)).reshape(-1), 0, scale_max_val)
            gc.collect()

    with timer('log transform'):
        for col in np.intersect1d(embed_cols, train_df.columns):
            train_df[col] = np.log1p(train_df[col])
            test_df[col] = np.log1p(test_df[col])
            gc.collect()

            val_cnt = get_val_cnt(train_df[col], precision=6)
            seg_out_dims.append(get_seg_num(val_cnt))

            scaler = MinMaxScaler(feature_range=(0, scale_max_val))
            train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1)).reshape(-1)
            test_df[col] = np.clip(scaler.transform(test_df[col].values.reshape(-1, 1)).reshape(-1), 0, scale_max_val)
            gc.collect()
        tr_segs = train_df[seg_cat_cols + seg_cols + embed_cols].values.astype(np.float32)
        ts_segs = test_df[seg_cat_cols + seg_cols + embed_cols].values.astype(np.float32)

    with timer('scale num cols'):
        num_cols = cat_cols
        for col in num_cols:
            scaler = MinMaxScaler(feature_range=(0, scale_max_val))
            train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1)).reshape(-1)
            test_df[col] = np.clip(scaler.transform(test_df[col].values.reshape(-1, 1)).reshape(-1), 0, scale_max_val)
            gc.collect()
        tr_nums = train_df[num_cols].values.astype(np.float32)
        ts_nums = test_df[num_cols].values.astype(np.float32)

    with timer('scale text cols'):
        scaler = MaxAbsScaler()
        x_text = vstack([tr_x_text, ts_x_text])
        scaler.fit(x_text)
        del x_text
        gc.collect()

        tr_x_text = scaler.transform(tr_x_text).astype(np.float32) * scale_max_val
        ts_x_text = scaler.transform(ts_x_text).astype(np.float32) * scale_max_val
        gc.collect()

    print(f'cat_in_dims({len(cat_in_dims)}): {cat_in_dims}')
    print(f'cat_out_dims({len(cat_out_dims)}): {cat_out_dims}')
    print(f'seg_out_dims({len(seg_out_dims)}): {seg_out_dims}')

    print(f'cat_in_dim_dict: {dict(zip(cat_cols,cat_in_dims))}')
    print(f'cat_out_dim_dict: {dict(zip(cat_cols,cat_out_dims))}')
    print(f'seg_cat_out_dim_dict: {dict(zip(seg_cat_cols,seg_out_dims[:len(seg_cat_cols)]))}')
    print(f'seg_out_dim_dict: {dict(zip(seg_cols,seg_out_dims[len(seg_cat_cols):len(seg_cat_cols)+len(seg_cols)]))}')
    print(f'embed_out_dim_dict: {dict(zip(embed_cols,seg_out_dims[len(seg_cat_cols)+len(seg_cols):]))}')

    return (tr_cats, tr_segs, tr_nums, tr_x_text, np.log1p(train_df.target.values)), (
        ts_cats, ts_segs, ts_nums, ts_x_text), (cat_in_dims, cat_out_dims, seg_out_dims)


def process_data(data_dir):
    with timer('get data'):
        train_df = pd.read_pickle(os.path.join(data_dir, 'train_df'), compression='gzip').drop('fullVisitorId', axis=1)
        gc.collect()
        test_df = pd.read_pickle(os.path.join(data_dir, 'test_df'), compression='gzip').drop('fullVisitorId', axis=1)
        gc.collect()
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')
        tr_x_text = joblib.load(os.path.join(data_dir, 'tr_x_text'))
        ts_x_text = joblib.load(os.path.join(data_dir, 'ts_x_text'))
        print(f'tr_x_text: {tr_x_text.shape}, ts_x_text: {ts_x_text.shape}')

    trd, tsd, md = transform_data(train_df, test_df, tr_x_text, ts_x_text)
    del train_df, test_df, tr_x_text, ts_x_text
    gc.collect()

    with timer('save data'):
        joblib.dump(trd, 'trd', compress=('gzip', 3))
        joblib.dump(tsd, 'tsd', compress=('gzip', 3))
        joblib.dump(md, 'md', compress=('gzip', 3))

        del trd, tsd, md
        gc.collect()

  
def run():
    sys.stderr = sys.stdout = open(os.path.join('get_nn_src_data_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-src-data'
    print(os.listdir(data_dir))

    process_data(data_dir)


if __name__ == '__main__':
    run()
