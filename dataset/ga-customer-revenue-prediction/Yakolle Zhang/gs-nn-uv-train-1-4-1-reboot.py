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
from keras import Input, Model, activations, initializers, regularizers, constraints
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.constraints import Constraint
from keras.engine import Layer, InputSpec
from keras.initializers import Constant, RandomUniform
from keras.layers import Dense, Embedding, Flatten, BatchNormalization, Activation, Dropout, Lambda, concatenate, Add
from keras.optimizers import Adam
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GroupShuffleSplit, KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------activations-----------------------------------------------------
def seu(x):
    return bk.sigmoid(x) * x


def make_lseu(c_val):
    def _lseu(x):
        x1 = bk.sigmoid(x) * x
        x2 = c_val + bk.log(1 + bk.relu(x - c_val))
        return bk.minimum(x1, x2)

    return _lseu


def lseu(x):
    return make_lseu(19.0)(x)


class MinMaxValue(Constraint):
    def __init__(self, min_value=1e-3, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return bk.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


class PLSEU(Layer):
    def __init__(self, alpha_initializer=Constant(4.4), alpha_regularizer=None, alpha_constraint=MinMaxValue(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(PLSEU, self).__init__(**kwargs)
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.alpha = None

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.alpha = self.add_weight(shape=(input_dim,), name='alpha', initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer, constraint=self.alpha_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, mask=None):
        x1 = bk.sigmoid(inputs / self.alpha) * inputs
        x2 = self.alpha * (self.alpha + bk.log(1 + bk.relu(inputs / self.alpha - self.alpha)))
        return bk.minimum(x1, x2)

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint)
        }
        base_config = super(PLSEU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def make_plseu(alpha):
    def _plseu(x):
        return PLSEU(alpha_initializer=Constant(alpha))(x)

    return _plseu


def plseu(x):
    return make_plseu(4.4)(x)


# --------------------------------------------------FMLayer-----------------------------------------------------
class FMLayer(Layer):
    def __init__(self, factor_rank, activation='softsign', use_bias=False, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)
        self.factor_rank = factor_rank
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.kernel = None
        self.bias = None
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.factor_rank),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.factor_rank,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        output = (bk.square(bk.dot(inputs, self.kernel)) - bk.dot(bk.square(inputs), bk.square(self.kernel))) / 2
        if self.use_bias:
            output = bk.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.factor_rank
        return tuple(output_shape)

    def get_config(self):
        config = {
            'factor_rank': self.factor_rank,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# --------------------------------------------------SegTriangleLayer-----------------------------------------------------
class SegTriangleLayer(Layer):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SegTriangleLayer, self).__init__(**kwargs)
        self.seg_num = seg_num

        self.left_pos = None
        self.middle_pos = None
        self.right_pos = None
        self.middle_seg_width = None

        self.input_val_range = input_val_range
        self.seg_func = seg_func

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert 1 == input_shape[-1]

        seg_width = (self.input_val_range[1] - self.input_val_range[0]) / self.seg_num
        left_pos = self.input_val_range[0] + seg_width
        right_pos = self.input_val_range[1] - seg_width

        self.left_pos = self.add_weight(shape=(1,), initializer=Constant(value=left_pos), name='left_pos')
        if self.seg_num > 2:
            self.middle_pos = self.add_weight(shape=(self.seg_num - 2,), name='middle_pos',
                                              initializer=RandomUniform(minval=left_pos, maxval=right_pos - seg_width))
        else:
            self.middle_pos = None
        self.right_pos = self.add_weight(shape=(1,), initializer=Constant(value=right_pos), name='right_pos')

        if self.seg_num > 2:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,), initializer=Constant(value=seg_width),
                                                    name='middle_seg_width')
        else:
            self.middle_seg_width = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: 1})
        self.built = True

    def call(self, inputs, **kwargs):
        left_out = self.left_pos - inputs
        middle_out = None if self.middle_pos is None else -bk.abs(inputs - self.middle_pos) + self.middle_seg_width
        right_out = inputs - self.right_pos

        if self.middle_pos is not None:
            output = bk.concatenate([left_out, middle_out, right_out])
        else:
            output = bk.concatenate([left_out, right_out])
        return self.seg_func(output)

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.seg_num
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'input_val_range': self.input_val_range,
            'seg_func': self.seg_func
        }
        base_config = super(SegTriangleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


def add_dense(x, units, bn=True, activation=seu, dropout=0.2):
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


def get_segments(seg_input, seg_out_dims, shrink_factor=1.0, seg_type=0, seg_func=seu, seg_input_val_range=(0, 1)):
    segments = []
    for i, out_dim in enumerate(seg_out_dims):
        segment = Lambda(lambda segs: segs[:, i, None])(seg_input)
        segment = SegTriangleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                   seg_func=seg_func)(segment)
        segments.append(segment)
    return segments


# --------------------------------------------------tnn-----------------------------------------------------
def get_simple_linear_output(flat, name=None, unit_activation=seu):
    flat = add_dense(flat, 64, bn=False, activation=unit_activation, dropout=0.05)
    return Dense(1, name=name)(flat)


def compile_default_mse_output(outputs, oh_input=None, cat_input=None, seg_input=None, num_input=None,
                               other_inputs=None, loss_weights=None):
    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='mse', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_simple_sigmoid_output(flat, name=None, unit_activation=seu):
    flat = add_dense(flat, 64, bn=False, activation=unit_activation, dropout=0.05)
    return Dense(1, activation='sigmoid', name=name)(flat)


def compile_default_bce_output(outputs, oh_input=None, cat_input=None, seg_input=None, num_input=None,
                               other_inputs=None, loss_weights=None):
    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_tnn_block(block_no, get_output=get_simple_linear_output, oh_input=None, cat_input=None, seg_input=None,
                  num_input=None, pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                  num_segs=None, seg_type=0, seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=50,
                  shrink_factor=1.0, use_fm=False, seg_flag=True, add_seg_src=True, seg_num_flag=True, x=None,
                  get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2,
                  fm_activation=None, hidden_units=320, hidden_activation=seu, hidden_dropout=0.2):
    embeds = [Flatten()(Embedding(3, 2)(oh_input))] if oh_input is not None else []
    if cat_input is not None:
        embeds += get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=shrink_factor ** block_no)
    embeds = Dropout(embed_dropout)(concatenate(embeds)) if embeds else None

    segments = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func,
                            seg_x_val_range) if seg_flag and seg_input is not None else[]
    segments += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func,
                             seg_x_val_range) if seg_num_flag and num_input is not None else []

    if pre_output is not None:
        seg_y_dim = shrink(seg_y_dim, shrink_factor ** block_no)
        segment = SegTriangleLayer(seg_y_dim, input_val_range=seg_y_val_range, seg_func=seg_func)(pre_output)
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

    flat = add_dense(flat, hidden_units, bn=True, activation=hidden_activation, dropout=hidden_dropout)
    tnn_block = get_output(flat, name=f'out{block_no}', unit_activation=hidden_activation)
    return tnn_block


def get_tnn_model(x, get_output=get_simple_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                  cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), use_fm=False,
                  seg_flag=True, add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2,
                  seg_func=seu, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2, fm_activation=None, hidden_units=320,
                  hidden_activation=seu, hidden_dropout=0.2):
    oh_input = Input(shape=[x['ohs'].shape[1]], name='ohs') if 'ohs' in x else None
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    tnn = get_tnn_block(0, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                        num_input=num_input, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                        seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type,
                        seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
                        seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers, embed_dropout=embed_dropout,
                        seg_func=seg_func, seg_dropout=seg_dropout, fm_dim=fm_dim, fm_dropout=fm_dropout,
                        fm_activation=fm_activation, hidden_units=hidden_units, hidden_activation=hidden_activation,
                        hidden_dropout=hidden_dropout)
    tnn = compile_func(tnn, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input, num_input=num_input)
    return tnn


# --------------------------------------------------nn_uv_train-----------------------------------------------------
scale_max_val = 20
tv_num = 5

oof_seed = 1
pred_type_id = 4
round_num = 1
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]
pred_max_val = np.log1p(5.1e10)
need_indi = True
 
use_fm = pred_type_id >= 7
block_num = 1
shrink_factor = 1
seg_flag = True
add_seg_src = True
seg_num_flag = False
fm_dim = 320
hidden_units = 320

seg_func = plseu
hidden_activation = plseu

lr_patience = 10
stop_patience = 15
epochs = 100
batch_size = 1024

res_shrinkage = 0.1

spend_hours = 7 if use_fm else 6


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def combine_features(features, batch_num=5):
    cols = []
    _batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * _batch_size: (i + 1) * _batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def get_ab_split(x, y, aind, bind):
    ax = {col: val[aind] for col, val in x.items()}
    ay = y[aind]
    bx = {col: val[bind] for col, val in x.items()}
    by = y[bind]
    return ax, ay, bx, by


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

    with timer('scale & collect seg infos'):
        indi_cols = sorted(ts_indi_df.columns)
        in_indi_df, out_indi_df, ts_indi_df = in_indi_df[indi_cols], out_indi_df[indi_cols], ts_indi_df[indi_cols]
        tr_indi_df = in_indi_df.append(out_indi_df, sort=False)
        gc.collect()

        indi_seg_out_dims = []
        for col in indi_cols:
            indi_seg_out_dims.append(get_seg_num_by_value(tr_indi_df[col], precision=5))

        scaler = MinMaxScaler(feature_range=(0, scale_max_val)).fit(tr_indi_df)
        in_indi_df[indi_cols] = scaler.transform(in_indi_df).astype(np.float32)
        out_indi_df[indi_cols] = scaler.transform(out_indi_df).astype(np.float32)
        ts_indi_df[indi_cols] = np.clip(scaler.transform(ts_indi_df), 0, scale_max_val).astype(np.float32)
        print(f'indi_cols({len(indi_cols)})')
        gc.collect()

    return in_indi_df, out_indi_df, ts_indi_df, indi_seg_out_dims


def get_oof_data(data_dir, in_indi_df, out_indi_df, indi_seg_out_dims):
    with timer('get train src data'):
        tr_cats, tr_segs, tr_nums, tr_x_text, y = joblib.load(os.path.join(data_dir, 'trd'))
        cat_in_dims, cat_out_dims, seg_out_dims = joblib.load(os.path.join(data_dir, 'md'))
        seg_out_dims += indi_seg_out_dims
        print(f'tr_cats: {tr_cats.shape}, tr_segs: {tr_segs.shape}, tr_nums: {tr_nums.shape},',
              f'tr_x_text: {tr_x_text.shape}, y: {y.shape}')

        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        tr_nums = combine_features([tr_nums, tr_x_text]) if 'plain' != text_type else tr_nums
        print(f'tr_nums: {tr_nums.shape}')

        x = {'cats': tr_cats, 'segs': tr_segs, 'nums': tr_nums}
        del tr_cats, tr_segs, tr_nums, tr_x_text
        gc.collect()

    with timer('oof split'):
        iind, oind = next(ShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(y))

        ix, iy, ox, oy = get_ab_split(x, y, iind, oind)
        print(f'iy: {iy.shape}, oy: {oy.shape}')
        print(f'iy(>0): {np.sum(iy>0)}, oy(>0): {np.sum(oy>0)}')
        del iind, oind, x, y
        gc.collect()

    if in_indi_df is not None:
        with timer('join indicators'):
            ix['segs'] = np.hstack([ix['segs'], in_indi_df.values])
            print(f"in_indi_df: {in_indi_df.shape}, ix_segs: {ix['segs'].shape}")
            ox['segs'] = np.hstack([ox['segs'], out_indi_df.values])
            print(f"out_indi_df: {out_indi_df.shape}, ox_segs: {ox['segs'].shape}")

    return ix, iy, ox, oy, (cat_in_dims, cat_out_dims, seg_out_dims)


def get_test_data(data_dir, ts_indi_df):
    with timer('get test src data'):
        ts_cats, ts_segs, ts_nums, ts_x_text = joblib.load(os.path.join(data_dir, 'tsd'))
        print(f'ts_cats: {ts_cats.shape}, ts_segs: {ts_segs.shape}, ts_nums: {ts_nums.shape},',
              f'ts_x_text: {ts_x_text.shape}')

        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        ts_nums = combine_features([ts_nums, ts_x_text]) if 'plain' != text_type else ts_nums
        print(f'ts_nums: {ts_nums.shape}')

        ts_x = {'cats': ts_cats, 'segs': ts_segs, 'nums': ts_nums}
        del ts_cats, ts_segs, ts_nums, ts_x_text
        gc.collect()

    if ts_indi_df is not None:
        with timer('join indicators'):
            ts_x['segs'] = np.hstack([ts_x['segs'], ts_indi_df.values])
            print(f"ts_indi_df: {ts_indi_df.shape}, ts_x_segs: {ts_x['segs'].shape}")

    return ts_x


def get_tv_data(x, y, tv_seed):
    tind, vind = next(ShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=tv_seed).split(y))
    tx, ty, vx, vy = get_ab_split(x, y, tind, vind)
    del tind, vind, x, y
    gc.collect()
    return tx, ty, vx, vy


def get_last_res_data(tv_round):
    last_data_dir = f'../input/gs-nn-uv-train-{oof_seed}-{pred_type_id}-{round_num-1}'
    model_id = f'rtnn_{oof_seed}_{pred_type_id}_{tv_round}'
    stp = joblib.load(os.path.join(last_data_dir, f'{model_id}_stp'))
    svp = joblib.load(os.path.join(last_data_dir, f'{model_id}_svp'))
    sop = joblib.load(os.path.join(last_data_dir, f'{model_id}_sop'))
    sp = joblib.load(os.path.join(last_data_dir, f'{model_id}_sp'))
    return stp, svp, sop, sp


def learn(data):
    ix, iy, ox, oy, (cat_in_dims, cat_out_dims, seg_out_dims), ts_x = data

    params = {'get_output': get_simple_linear_output, 'compile_func': compile_default_mse_output,
              'cat_in_dims': cat_in_dims, 'cat_out_dims': cat_out_dims, 'seg_out_dims': seg_out_dims,
              'seg_x_val_range': (0, scale_max_val), 'use_fm': use_fm, 'seg_flag': seg_flag,
              'add_seg_src': add_seg_src, 'seg_num_flag': seg_num_flag, 'fm_dim': fm_dim, 'fm_activation': None,
              'hidden_units': hidden_units, 'seg_func': seg_func, 'hidden_activation': hidden_activation}

    c = tv_num
    while not os.path.exists(f'rtnn_{oof_seed}_{pred_type_id}_{c}_p') and c > 0:
        c -= 1
    c += 1

    start_time = int(time.time())
    end_time = start_time + spend_hours * 3600 - 5 * 60
    last_time = start_time
    spend_times = []
    for i in range(c, tv_num + 1):
        tv_seed = oof_seed * 1000000 + pred_type_id * 10000 + 2 * i * 100
        model_id = f'rtnn_{oof_seed}_{pred_type_id}_{i}'
        tx, ty, vx, vy = get_tv_data(ix, iy, tv_seed)
        stp, svp, sop, sp = None, None, None, None
        if round_num > 1:
            stp, svp, sop, sp = get_last_res_data(i)
            print(f'last res data: stp({stp.shape[0]}), svp({svp.shape[0]}), sop({sop.shape[0]}), sp({sp.shape[0]})')
        print(f'---------------------------------{model_id}---------------------------------')
        print(f'ty: {ty.shape}, vy: {vy.shape}')
        print(f'ty(>0): {np.sum(ty>0)}, vy(>0): {np.sum(vy>0)}')
        gc.collect()

        checkpointer = ModelCheckpoint(model_id, monitor='val_loss', verbose=1, save_best_only=True,
                                       save_weights_only=True, period=1)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1, min_lr=1e-5,
                                         min_delta=3e-3)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=3e-3, patience=stop_patience, verbose=1)
        model = get_tnn_model(tx, **params)

        if stp is not None:
            tx['lp'], vx['lp'], ox['lp'], ts_x['lp'] = stp, svp, sop, sp

            lp_input = Input(shape=[1], name='lp')
            inputs = model.inputs + [lp_input]
            output = Add(name='output')([model.output, Lambda(lambda x: res_shrinkage * x, name='shlp')(lp_input)])
            rtnn = Model(inputs, output)
            rtnn.compile(loss=model.loss, optimizer=model.optimizer)
            model = rtnn

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
        joblib.dump(tp + (1 - res_shrinkage) * tx['lp'] if 'lp' in tx else tp, f'{model_id}_stp', compress=('gzip', 1))

        vp = np.squeeze(model.predict(vx, batch_size=_batch_size))
        vp = np.clip(vp, 0, pred_max_val)
        v_score = rmse(vy, vp)
        joblib.dump(vp, f'{model_id}_vp', compress=('gzip', 3))
        joblib.dump(vy, f'{model_id}_vy', compress=('gzip', 3))
        joblib.dump(vp + (1 - res_shrinkage) * vx['lp'] if 'lp' in vx else vp, f'{model_id}_svp', compress=('gzip', 1))

        op = np.squeeze(model.predict(ox, batch_size=_batch_size))
        op = np.clip(op, 0, pred_max_val)
        o_score = rmse(oy, op)
        joblib.dump(op, f'{model_id}_op', compress=('gzip', 3))
        joblib.dump(oy, f'{model_id}_oy', compress=('gzip', 3))
        joblib.dump(op + (1 - res_shrinkage) * ox['lp'] if 'lp' in ox else op, f'{model_id}_sop', compress=('gzip', 1))

        p = np.squeeze(model.predict(ts_x, batch_size=_batch_size))
        p = np.clip(p, 0, pred_max_val)
        joblib.dump(p, f'{model_id}_p', compress=('gzip', 3))
        joblib.dump(p + (1 - res_shrinkage) * ts_x['lp'] if 'lp' in ts_x else p, f'{model_id}_sp', compress=('gzip', 1))
        print(f'{model_id}: t_rmse={t_score}, v_rmse={v_score}, o_rmse={o_score}')


def run():
    reboot_dir = f'../input/gs-nn-uv-train-{oof_seed}-{pred_type_id}-{round_num}'
    file_names = os.listdir(reboot_dir)
    print(file_names)
    for file_name in file_names:
        file_path = os.path.join(reboot_dir, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, '.')

    sys.stderr = sys.stdout = open(os.path.join('nn_uv_train_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-nn-src-data'
    print(os.listdir(data_dir))

    if need_indi:
        in_indi_df, out_indi_df, ts_indi_df, indi_seg_out_dims = get_indicators()
    else:
        in_indi_df, out_indi_df, ts_indi_df, indi_seg_out_dims = None, None, None, []
    ix, iy, ox, oy, md = get_oof_data(data_dir, in_indi_df, out_indi_df, indi_seg_out_dims)
    ts_x = get_test_data(data_dir, ts_indi_df)
    del in_indi_df, out_indi_df, ts_indi_df, indi_seg_out_dims
    gc.collect()
    data = ix, iy, ox, oy, md, ts_x
 
    # init_tensorflow()
    learn(data)


if __name__ == '__main__':
    run()
