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
from keras.layers import Dense, Embedding, Flatten, BatchNormalization, Activation, Dropout, Lambda, concatenate
from keras.optimizers import Adam
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GroupShuffleSplit, KFold


# --------------------------------------------------activations-----------------------------------------------------
def seu(x):
    return bk.sigmoid(x) * x


def make_lseu(c_val):
    def _lseu(x):
        x1 = bk.sigmoid(x) * x
        x2 = c_val + bk.log(1 + bk.relu(x - c_val))
        return bk.minimum(x1, x2)

    return _lseu


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


def get_simple_sigmoid_output(flat, name=None, unit_activation=seu):
    flat = add_dense(flat, 64, bn=False, activation=unit_activation, dropout=0.05)
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


# --------------------------------------------------nn_indi_train-----------------------------------------------------
id_col = 'fullVisitorId'
scale_max_val = 20
kfold_k = 5
repeat_times = 3
  
oof_seed = 1
pred_type_id = 18
indi_type_id = 2
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]
user_mode = indi_type_id > 2
pred_max_val = np.log1p(7.8e10) if user_mode else np.log1p(2.4e10)
need_embeded_target = (pred_type_id - 1) // len(text_types) % 2 == 1 and not user_mode

use_fm = pred_type_id >= 13
block_num = 1
shrink_factor = 1
seg_flag = True
add_seg_src = True
seg_num_flag = False
fm_dim = 320
hidden_units = 320

lseu = make_lseu(0.9) if indi_type_id > 1 else make_lseu(19.0)
plseu = make_plseu(0.65) if indi_type_id > 1 else make_plseu(4.4)

seg_func = seu
hidden_activation = seu

lr_patience = 10
stop_patience = 15
epochs = 100
batch_size = 1024


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def auc(y, p):
    if np.sum(y) in [0, y.shape[0]]:
        return metrics.accuracy_score(y, p > 0.5)
    return metrics.roc_auc_score(y, p)


def keras_auc(target, pred):
    return tf.py_func(lambda ys, ps: np.array(auc(ys.flatten(), ps.flatten()), dtype=np.float32), [target, pred],
                      tf.float32)


def combine_features(features, batch_num=5):
    cols = []
    _batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * _batch_size: (i + 1) * _batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def refine_embeded_target(segs, md=None):
    if not need_embeded_target:
        segs = segs[:, :-3]
        if md is not None:
            md = md[0], md[1], md[2][:-3]
    return segs, md


def get_ab_split(x, y, selector, aind, bind):
    a_selector = selector.take(aind).reset_index(drop=True)
    ax = {col: val[aind] for col, val in x.items()}
    ay = y[aind]
    b_selector = selector.take(bind).reset_index(drop=True)
    bx = {col: val[bind] for col, val in x.items()}
    by = y[bind]
    return ax, ay, a_selector, bx, by, b_selector


def get_oof_data(data_dir):
    with timer('get train src data'):
        tr_cats, tr_segs, tr_nums, tr_x_text, tr_selector, y = joblib.load(os.path.join(data_dir, 'trd'))
        tr_cats_u, tr_segs_u, tr_nums_u, tr_x_text_u, tr_selector_u, y_u = joblib.load(os.path.join(data_dir, 'trd_u'))
        md = joblib.load(os.path.join(data_dir, 'md_u')) if user_mode else joblib.load(os.path.join(data_dir, 'md'))

        if not user_mode:
            tr_segs, md = refine_embeded_target(tr_segs, md)
        if user_mode:
            tr_cats, tr_segs, tr_nums, tr_x_text, y = tr_cats_u, tr_segs_u, tr_nums_u, tr_x_text_u, y_u
        y = y if 1 == indi_type_id else y.astype(np.bool).astype(np.int8)
        print(f'tr_cats: {tr_cats.shape}, tr_segs: {tr_segs.shape}, tr_nums: {tr_nums.shape},',
              f'tr_x_text: {tr_x_text.shape}, y: {y.shape}')

        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        tr_nums = combine_features([tr_nums, tr_x_text]) if 'plain' != text_type else tr_nums
        print(f'tr_nums: {tr_nums.shape}')

        if indi_type_id > 1:
            tr_segs /= scale_max_val
            tr_nums /= scale_max_val

        x = {'cats': tr_cats, 'segs': tr_segs, 'nums': tr_nums}
        del tr_cats, tr_segs, tr_nums, tr_x_text, tr_cats_u, tr_segs_u, tr_nums_u, tr_x_text_u, y_u
        gc.collect()

    with timer('oof split'):
        iind, oind = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(
            tr_selector, groups=tr_selector[id_col]))
        if user_mode:
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


def get_test_data(data_dir):
    with timer('get test src data'):
        ts_cats, ts_segs, ts_nums, ts_x_text, ts_selector = joblib.load(os.path.join(data_dir, 'tsd_u')) \
            if user_mode else joblib.load(os.path.join(data_dir, 'tsd'))

        if not user_mode:
            ts_segs, _ = refine_embeded_target(ts_segs)
        print(f'ts_cats: {ts_cats.shape}, ts_segs: {ts_segs.shape}, ts_nums: {ts_nums.shape},',
              f'ts_x_text: {ts_x_text.shape}')

        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        ts_nums = combine_features([ts_nums, ts_x_text]) if 'plain' != text_type else ts_nums
        print(f'ts_nums: {ts_nums.shape}')

        if indi_type_id > 1:
            ts_segs /= scale_max_val
            ts_nums /= scale_max_val

        ts_x = {'cats': ts_cats, 'segs': ts_segs, 'nums': ts_nums}
        del ts_cats, ts_segs, ts_nums, ts_x_text
        gc.collect()

    return ts_x, ts_selector


def learn(data):
    indi_id = f'indi_{oof_seed}_{pred_type_id}_{indi_type_id}'
    ix, iy, in_selector, ox, oy, out_selector, (cat_in_dims, cat_out_dims, seg_out_dims), ts_x, ts_selector = data

    measure_func = rmse
    measure_name = measure_func.__name__
    min_delta = 3e-3
    params = {'get_output': get_simple_linear_output, 'compile_func': compile_default_mse_output,
              'cat_in_dims': cat_in_dims, 'cat_out_dims': cat_out_dims, 'seg_out_dims': seg_out_dims,
              'seg_x_val_range': (0, scale_max_val), 'use_fm': use_fm, 'seg_flag': seg_flag,
              'add_seg_src': add_seg_src, 'seg_num_flag': seg_num_flag, 'fm_dim': fm_dim, 'fm_activation': None,
              'hidden_units': hidden_units, 'seg_func': seg_func, 'hidden_activation': hidden_activation}
    if indi_type_id > 1:
        params.update({'get_output': get_simple_sigmoid_output, 'compile_func': compile_default_bce_output,
                       'seg_x_val_range': (0, 1), 'fm_activation': 'softsign'})
        measure_func = auc
        measure_name = measure_func.__name__
        min_delta = 2e-4

    ipss = [np.zeros(iy.shape) for i in range(repeat_times)]
    opss, pss = [[] for i in range(repeat_times)], [[] for i in range(repeat_times)]
    t_scoress, v_scoress = [[] for i in range(repeat_times)], [[] for i in range(repeat_times)]
    o_scoress = [[] for i in range(repeat_times)]
    if os.path.exists('predss'):
        ipss, opss, pss = joblib.load('predss')
        t_scoress, v_scoress, o_scoress = joblib.load('scoress')
    c = 0
    for ops in opss:
        if len(ops) < kfold_k:
            break
        c += 1

    start_time = int(time.time())
    end_time = start_time + 6 * 3600 - 5 * 60
    last_time = start_time
    spend_times = []
    for i in range(c, repeat_times):
        print(f'---------------------------------round({i})---------------------------------')
        seed = oof_seed * 1000000 + pred_type_id * 10000 + (2 * indi_type_id - 1) * 100 + (i + 1) * 10
        set_seed(seed)
        ip, ops, ps = ipss[i], opss[i], pss[i]
        t_scores, v_scores, o_scores = t_scoress[i], v_scoress[i], o_scoress[i]
        k = len(ops)
        fold_inds = (kfold((iy,), kfold_k, True, seed) if user_mode else group_kfold(
            (ix, iy, in_selector, id_col), kfold_k, True, seed))[k:]

        for tind, vind in fold_inds:
            model_id = f'tnn_{seed}_{k}'
            tx, ty, t_selector, vx, vy, v_selector = get_ab_split(ix, iy, in_selector, tind, vind)
            print(f'---------------------------------{model_id}---------------------------------')
            print(f'ty: {ty.shape}, vy: {vy.shape}')
            gc.collect()

            checkpointer = ModelCheckpoint(model_id, monitor='val_loss', verbose=1, save_best_only=True,
                                           save_weights_only=True, period=1)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1,
                                             min_lr=1e-5, min_delta=min_delta)
            early_stopper = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=stop_patience,
                                          verbose=1)
            model = get_tnn_model(tx, **params)

            model.fit(tx, ty, epochs=epochs, batch_size=batch_size, validation_data=(vx, vy), verbose=2,
                      callbacks=[checkpointer, lr_scheduler, early_stopper])

            tp, vp, op, p, t_score, v_score, o_score = validation((tx, ty, vx, vy, ox, oy, ts_x, model), model_id,
                                                                  measure_func)

            k += 1
            ip[vind] = vp
            ops.append(op)
            ps.append(p)
            t_scores.append(t_score)
            v_scores.append(v_score)
            o_scores.append(o_score)
            joblib.dump((ipss, opss, pss), 'predss')
            joblib.dump((t_scoress, v_scoress, o_scoress), 'scoress')
            del tx, ty, vx, vy
            gc.collect()

            cur_time = int(time.time())
            spend_times.append(cur_time - last_time)
            if cur_time + 2 * np.max(spend_times) - np.min(spend_times) > end_time and (
                                i * kfold_k + k < repeat_times * kfold_k):
                exit(0)
            last_time = cur_time
        print(f't_{measure_name}s: mean={np.mean(t_scores)}, std={np.std(t_scores)}, {t_scores}')
        print(f'v_{measure_name}s: mean={np.mean(v_scores)}, std={np.std(v_scores)}, {v_scores}')
        print(f'o_{measure_name}s: mean={np.mean(o_scores)}, std={np.std(o_scores)}, {o_scores}')

    print_scores(t_scoress, v_scoress, o_scoress, measure_name)
    in_selector['pred'] = np.mean(ipss, axis=0)
    in_selector.to_pickle(f'{indi_id}_ip', compression='gzip')
    out_selector['pred'] = np.mean(opss, axis=(0, 1))
    out_selector.to_pickle(f'{indi_id}_op', compression='gzip')
    ts_selector['pred'] = np.mean(pss, axis=(0, 1))
    ts_selector.to_pickle(f'{indi_id}_p', compression='gzip')


def print_scores(t_scoress, v_scoress, o_scoress, measure_name):
    tms, vms, oms = [], [], []
    tss, vss, oss = [], [], []
    for i in range(len(t_scoress)):
        print(f'---------------------------------round({i})---------------------------------')
        t_scores, v_scores, o_scores = t_scoress[i], v_scoress[i], o_scoress[i]
        tm, vm, om = np.mean(t_scores), np.mean(v_scores), np.mean(o_scores)
        tstd, vstd, ostd = np.std(t_scores), np.std(v_scores), np.std(o_scores)
        print(f't_{measure_name}s: mean={tm}, std={tstd}, {t_scores}')
        print(f'v_{measure_name}s: mean={vm}, std={vstd}, {v_scores}')
        print(f'o_{measure_name}s: mean={om}, std={ostd}, {o_scores}')
        tms.append(tm)
        vms.append(vm)
        oms.append(om)
        tss.append(tstd)
        vss.append(vstd)
        oss.append(ostd)
    print(f't_{measure_name}_means: mean={np.mean(tms)}, std={np.std(tms)}, {tms}')
    print(f't_{measure_name}_stds: mean={np.mean(tss)}, std={np.std(tss)}, {tss}')
    print(f'v_{measure_name}_means: mean={np.mean(vms)}, std={np.std(vms)}, {vms}')
    print(f'v_{measure_name}_stds: mean={np.mean(vss)}, std={np.std(vss)}, {vss}')
    print(f'o_{measure_name}_means: mean={np.mean(oms)}, std={np.std(oms)}, {oms}')
    print(f'o_{measure_name}_stds: mean={np.mean(oss)}, std={np.std(oss)}, {oss}')


def validation(data, model_id, measure_func, _batch_size=50000):
    tx, ty, vx, vy, ox, oy, ts_x, model = data
    measure_name = measure_func.__name__

    with timer('validation'):
        model = read_weights(model, model_id)

        tp = np.squeeze(model.predict(tx, batch_size=_batch_size))
        tp = np.clip(tp, 0, pred_max_val) if 1 == indi_type_id else tp
        t_score = measure_func(ty, tp)
        vp = np.squeeze(model.predict(vx, batch_size=_batch_size))
        vp = np.clip(vp, 0, pred_max_val) if 1 == indi_type_id else vp
        v_score = measure_func(vy, vp)
        op = np.squeeze(model.predict(ox, batch_size=_batch_size))
        op = np.clip(op, 0, pred_max_val) if 1 == indi_type_id else op
        o_score = measure_func(oy, op)
        p = np.squeeze(model.predict(ts_x, batch_size=_batch_size))
        p = np.clip(p, 0, pred_max_val) if 1 == indi_type_id else p
        print(f'{model_id}: t_{measure_name}={t_score}, v_{measure_name}={v_score}, o_{measure_name}={o_score}')

    return tp, vp, op, p, t_score, v_score, o_score

 
def run():
    reboot_dir = f'../input/gs-nn-indi-train-{oof_seed}-{pred_type_id}-{indi_type_id}-reboot'
    file_names = os.listdir(reboot_dir)
    print(file_names)
    for file_name in file_names:
        file_path = os.path.join(reboot_dir, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, '.')
 
    sys.stderr = sys.stdout = open(os.path.join('nn_indi_train_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-nn-src-data'
    print(os.listdir(data_dir))

    ix, iy, in_selector, ox, oy, out_selector, md = get_oof_data(data_dir)
    ts_x, ts_selector = get_test_data(data_dir)
    data = ix, iy, in_selector, ox, oy, out_selector, md, ts_x, ts_selector

    # init_tensorflow()
    learn(data)


if __name__ == '__main__':
    run()
