import gc
import os
import random
import string
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

import keras.backend as bk
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors as wv
from keras import Input, Model, activations, initializers, regularizers, constraints
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.constraints import Constraint
from keras.engine import Layer, InputSpec
from keras.initializers import Constant, RandomUniform
from keras.layers import Conv1D, GlobalMaxPool1D, CuDNNLSTM, SpatialDropout1D, CuDNNGRU
from keras.layers import Dense, Embedding, Flatten, BatchNormalization, Activation, Dropout, Lambda, concatenate
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import KFold, ShuffleSplit
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


# --------------------------------------------------callbacks-----------------------------------------------------
def calc_score(logs, monitor='loss', monitor_op=np.less):
    score = None
    t_score = logs.get(monitor)
    v_score = logs.get(f'val_{monitor}')
    if t_score is not None and v_score is not None:
        score = v_score ** 2 + (1 if monitor_op == np.less else -1) * (t_score - v_score) ** 2
    return score


class CheckpointDecorator(ModelCheckpoint):
    def __init__(self, filepath, monitor='loss', calc_score_func=calc_score, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        super(CheckpointDecorator, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode,
                                                  period)
        self.calc_score_func = calc_score_func

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = self.calc_score_func(logs, self.monitor, self.monitor_op)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class LRAnnealingByLoss(Callback):
    def __init__(self, loss_lr_pairs, tol=0.0, verbose=0):
        super(LRAnnealingByLoss, self).__init__()
        self.loss_lr_pairs = loss_lr_pairs[::-1]
        self.tol = tol
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        lr = self.loss_lr_pairs.pop()[1]
        bk.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        cur_loss = logs.get('loss')
        if cur_loss is not None and self.loss_lr_pairs:
            loss, lr = self.loss_lr_pairs[-1]
            if cur_loss - loss <= self.tol:
                bk.set_value(self.model.optimizer.lr, lr)
                self.loss_lr_pairs.pop()
                if self.verbose > 0:
                    print('\nEpoch %05d: LRAnnealingByLoss setting learning '
                          'rate to %s.' % (epoch + 1, lr))


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


# --------------------------------------------------nn_util-----------------------------------------------------
def init_tensorflow():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)
    bk.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


def set_seed(_seed=10000):
    os.environ['PYTHONHASHSEED'] = str(_seed + 6)
    np.random.seed(_seed + 7)
    random.seed(_seed + 8)
    tf.set_random_seed(_seed + 9)


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


def calc_val_cnt(x, precision=4):
    val_mean = np.mean(np.abs(x))
    cur_precision = np.round(np.log10(val_mean))
    x = (x * 10 ** (precision - cur_precision)).astype(np.int64)
    val_cnt = len(np.unique(x))
    return val_cnt


def get_seg_num_by_value(x, precision=4, shrink_factor=0.5):
    val_cnt = calc_val_cnt(x, precision)
    return get_seg_num(val_cnt, shrink_factor=shrink_factor)


def read_weights(model, weights_path):
    model.load_weights(weights_path)
    return model


def add_dense(x, units, bn=True, activation=seu, dropout=0.2):
    x = Dense(units)(x)
    if bn:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    if dropout > 0:
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
def get_linear_output(flat, name=None):
    return Dense(1, name=name)(flat)


def compile_default_mse_output(outputs, cat_input=None, seg_input=None, num_input=None, other_inputs=None,
                               loss_weights=None):
    inputs = [cat_input] if cat_input is not None else []
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='mse', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_sigmoid_output(flat, name=None):
    return Dense(1, activation='sigmoid', name=name)(flat)


def get_default_dense_layers(feats, extra_feats, hidden_units=(320, 64), hidden_activation=seu,
                             hidden_dropouts=(0.2, 0.05)):
    flat = concatenate([feats, extra_feats]) if feats is not None and extra_feats is not None else \
        feats if feats is not None else extra_feats
    if hidden_units:
        hidden_layer_num = len(hidden_units)
        for i in range(hidden_layer_num):
            flat = add_dense(flat, hidden_units[i], bn=i < hidden_layer_num - 1 or 1 == hidden_layer_num,
                             activation=hidden_activation, dropout=hidden_dropouts[i])
    return flat


def compile_default_bce_output(outputs, cat_input=None, seg_input=None, num_input=None, other_inputs=None,
                               loss_weights=None):
    inputs = [cat_input] if cat_input is not None else []
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_tnn_block(block_no, get_output=get_linear_output, cat_input=None, seg_input=None, num_input=None,
                  pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0,
                  seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=50, shrink_factor=1.0, use_fm=False,
                  seg_flag=True, add_seg_src=True, seg_num_flag=True, x=None, extra_inputs=None, get_extra_layers=None,
                  embed_dropout=0.2, seg_func=seu, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2, fm_activation=None,
                  get_last_layers=get_default_dense_layers, hidden_units=(320, 64), hidden_activation=seu,
                  hidden_dropouts=(0.2, 0.05)):
    embeds = get_embeds(cat_input, cat_in_dims, cat_out_dims,
                        shrink_factor=shrink_factor ** block_no) if cat_input is not None else []
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
    feats = concatenate(feats) if len(feats) > 1 else feats[0] if feats else None

    extra_feats, extra_inputs = get_extra_layers(x, feats, extra_inputs) if get_extra_layers is not None else (
        None, extra_inputs)

    if use_fm and feats is not None:
        fm = FMLayer(fm_dim, activation=fm_activation)(feats)
        fm = Dropout(fm_dropout)(fm)
        feats = concatenate([feats, fm])

    flat = get_last_layers(feats, extra_feats, hidden_units=hidden_units, hidden_activation=hidden_activation,
                           hidden_dropouts=hidden_dropouts)
    tnn_block = get_output(flat, name=f'out{block_no}')
    return tnn_block, extra_inputs


def get_tnn_model(x, get_output=get_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                  cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), use_fm=False,
                  seg_flag=True, add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2,
                  seg_func=seu, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2, fm_activation=None,
                  get_last_layers=get_default_dense_layers, hidden_units=(320, 64), hidden_activation=seu,
                  hidden_dropouts=(0.2, 0.05)):
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    tnn, extra_inputs = get_tnn_block(
        0, get_output=get_output, cat_input=cat_input, seg_input=seg_input, num_input=num_input,
        cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims, seg_out_dims=seg_out_dims, num_segs=num_segs,
        seg_type=seg_type, seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
        seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers, embed_dropout=embed_dropout,
        seg_func=seg_func, seg_dropout=seg_dropout, fm_dim=fm_dim, fm_dropout=fm_dropout, fm_activation=fm_activation,
        get_last_layers=get_last_layers, hidden_units=hidden_units, hidden_activation=hidden_activation,
        hidden_dropouts=hidden_dropouts)
    tnn = compile_func(tnn, cat_input=cat_input, seg_input=seg_input, num_input=num_input, other_inputs=extra_inputs)
    return tnn


# --------------------------------------------------tnn_submit-----------------------------------------------------
last_best_p_th = 0.5

oof_seed = 2
pred_type_id = 2

vocab_size = None
embed_dim = 300
seq_len = 56
word_index = {}
embed_weights = None

use_fm = False
block_num = 1
shrink_factor = 1
seg_flag = True
add_seg_src = True
seg_num_flag = False
fm_dim = 320
hidden_units = (64,)
hidden_dropouts = (0.05,)

lseu = make_lseu(0.9)
plseu = make_plseu(0.65)

seg_func = bk.relu
hidden_activation = 'relu'

lr_patience = 3
stop_patience = 10
epochs = 100
batch_size = 1024

eid = 0


def f1(y, p, detail=False):
    global last_best_p_th

    left = 0.1
    right = 0.9
    pace = 0.01
    k = 5
    tol_times = 2 * k

    f1_pairs = []
    best_score = 0

    p_th = last_best_p_th
    sink_times = 0
    while p_th <= right:
        f1_score = metrics.f1_score(y, p > p_th)
        f1_pairs.append((round(p_th, 10), f1_score))
        if f1_score < best_score:
            sink_times += 1
        else:
            best_score = f1_score
            sink_times = 0
        if sink_times > tol_times:
            break
        p_th += pace

    p_th = last_best_p_th - pace
    sink_times = 0
    while p_th >= left:
        f1_score = metrics.f1_score(y, p > p_th)
        f1_pairs.append((round(p_th, 10), f1_score))
        if f1_score < best_score:
            sink_times += 1
        else:
            best_score = f1_score
            sink_times = 0
        if sink_times > tol_times:
            break
        p_th -= pace

    f1_pairs = sorted(f1_pairs, key=lambda pair: (pair[1], pair[0]), reverse=True)[:k]
    last_best_p_th = f1_pairs[0][0]
    f1s = [f1_score for p_th, f1_score in f1_pairs]
    f1_mean = np.mean(f1s)
    if detail:
        print(f'last_best_p_th={last_best_p_th}, f1_mean={f1_mean}, f1_std={np.std(f1s)}, {f1_pairs}')
    return f1_mean


def keras_f1(target, pred):
    return tf.py_func(lambda ys, ps: np.array(f1(ys.flatten(), ps.flatten()), dtype=np.float32), [target, pred],
                      tf.float32)


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def get_ab_split(x, y, aind, bind):
    ax = {col: val[aind] for col, val in x.items()}
    ay = y[aind]
    bx = {col: val[bind] for col, val in x.items()}
    by = y[bind]
    return ax, ay, bx, by


def get_data(data_dir='../input'):
    with timer('load data'):
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col='qid')
        submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}, submission: {submission.shape}')
        test_df = submission.join(test_df, on='qid', how='inner').drop('prediction', axis=1)
        print(f'test_df: {test_df.shape}')
        gc.collect()

        train_df = train_df.fillna('the')
        test_df = test_df.fillna('the')
        gc.collect()

    with timer('encode text'):
        def encode_text(col):
            def count_chars(txt):
                _len = 0
                digit_cnt, number_cnt = 0, 0
                lower_cnt, upper_cnt, letter_cnt, word_cnt = 0, 0, 0, 0
                char_cnt, term_cnt = 0, 0
                conj_cnt, blank_cnt, punc_cnt = 0, 0, 0
                sign_cnt, marks_cnt = 0, 0

                flag = 10
                for ch in txt:
                    _len += 1
                    if ch in string.ascii_lowercase:
                        lower_cnt += 1
                        letter_cnt += 1
                        char_cnt += 1
                        if flag:
                            word_cnt += 1
                            if flag > 2:
                                term_cnt += 1
                            flag = 0
                    elif ch in string.ascii_uppercase:
                        upper_cnt += 1
                        letter_cnt += 1
                        char_cnt += 1
                        if flag:
                            word_cnt += 1
                            if flag > 2:
                                term_cnt += 1
                            flag = 0
                    elif ch in string.digits:
                        digit_cnt += 1
                        char_cnt += 1
                        if 1 != flag:
                            number_cnt += 1
                            if flag > 2:
                                term_cnt += 1
                            flag = 1
                    elif '_' == ch:
                        conj_cnt += 1
                        char_cnt += 1
                        if flag > 2:
                            term_cnt += 1
                        flag = 2
                    elif ch in string.whitespace:
                        blank_cnt += 1
                        flag = 3
                    elif ch in string.punctuation:
                        punc_cnt += 1
                        flag = 4
                    else:
                        sign_cnt += 1
                        if flag != 5:
                            marks_cnt += 1
                            flag = 5

                return (_len, digit_cnt, number_cnt, digit_cnt / (1 + number_cnt), lower_cnt, upper_cnt, letter_cnt,
                        word_cnt, letter_cnt / (1 + word_cnt), char_cnt, term_cnt, char_cnt / (1 + term_cnt), conj_cnt,
                        blank_cnt, punc_cnt, sign_cnt, marks_cnt, sign_cnt / (1 + marks_cnt))

            return np.array(list(col.apply(count_chars)), dtype=np.uint16)

        tr_cnts = encode_text(train_df.question_text)
        ts_cnts = encode_text(test_df.question_text)
        gc.collect()
        print(f'tr_cnts: {tr_cnts.shape}, ts_cnts: {ts_cnts.shape}')

    with timer('collect segment infos'):
        seg_out_dims = []
        for i in range(tr_cnts.shape[1]):
            seg_out_dims.append(get_seg_num_by_value(tr_cnts[:, i]))

        scaler = MinMaxScaler(feature_range=(0, 1))
        tr_cnts = scaler.fit_transform(tr_cnts)
        ts_cnts = np.clip(scaler.transform(ts_cnts), 0, 1)
        gc.collect()
        print(f'seg_out_dims({len(seg_out_dims)}): {seg_out_dims}')

    with timer('reserve punctuation'):
        def reserve_puncts(text):
            puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%',
                      '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→',
                      '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–',
                      '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓',
                      '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯',
                      '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
                      '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
            punct_dic = {punct: f' {punct} ' for punct in puncts}
            punct_dic.update({'\t': ' ', '\n': ' ', '\r': ' ', '\u200b': ''})
            return text.translate(str.maketrans(punct_dic))

        train_df['question_text'] = train_df.question_text.apply(reserve_puncts)
        test_df['question_text'] = test_df.question_text.apply(reserve_puncts)
        gc.collect()

    with timer('to text sequence'):
        tkr = Tokenizer(num_words=vocab_size, filters='')
        s = train_df.question_text.append(test_df.question_text, ignore_index=True)
        tkr.fit_on_texts(s)
        word_index.update(tkr.word_index)

        nn_tr_x = tkr.texts_to_sequences(train_df.question_text)
        nn_ts_x = tkr.texts_to_sequences(test_df.question_text)
        nn_tr_x = pad_sequences(nn_tr_x, maxlen=seq_len, truncating='post', padding='post')
        nn_ts_x = pad_sequences(nn_ts_x, maxlen=seq_len, truncating='post', padding='post')
        gc.collect()
        print(f'nn_tr_x: {nn_tr_x.shape}, nn_ts_x: {nn_ts_x.shape}')

    with timer('to tnn data'):
        tr_x = {'segs': tr_cnts, 'text': nn_tr_x}
        gc.collect()
        ts_x = {'segs': ts_cnts, 'text': nn_ts_x}
        gc.collect()

    y = train_df.target.values.copy()
    del train_df, test_df
    gc.collect()

    return tr_x, y, seg_out_dims, ts_x, submission


def load_embed_dic(embed_id, embed_root_dir='../input/embeddings'):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    file_path_dic = {
        'glove': os.path.join(embed_root_dir, 'glove.840B.300d', 'glove.840B.300d.txt'),
        'wiki': os.path.join(embed_root_dir, 'wiki-news-300d-1M', 'wiki-news-300d-1M.vec'),
        'para': os.path.join(embed_root_dir, 'paragram_300_sl999', 'paragram_300_sl999.txt'),
        'google': os.path.join(embed_root_dir, 'GoogleNews-vectors-negative300', 'GoogleNews-vectors-negative300.bin')
    }
    if 'wiki' == embed_id:
        embed_dic = dict(get_coefs(*line.split(' ')) for line in open(
            file_path_dic[embed_id], encoding='utf8', errors='ignore') if len(line) > 100)
    elif 'google' == embed_id:
        embed_dic = wv.load_word2vec_format(file_path_dic[embed_id], binary=True)
    else:
        embed_dic = dict(get_coefs(*line.split(' ')) for line in open(
            file_path_dic[embed_id], encoding='utf8', errors='ignore'))

    return embed_dic


def get_embed_weights(embed_id, embed_root_dir='../input/embeddings'):
    with timer('load embed'):
        embed_dic = load_embed_dic(embed_id, embed_root_dir)

    with timer('init embed'):
        def is_in_vocab(_word):
            if 'google' == embed_id:
                return _word in embed_dic.vocab
            return _word in embed_dic

        _weights = np.full((len(word_index), embed_dim), np.nan)
        for word, idx in word_index.items():
            if idx < _weights.shape[0] and is_in_vocab(word):
                _weights[idx] = embed_dic[word].copy()
        nan_embed = embed_dic['it'].copy()

        del embed_dic
        gc.collect()

    return _weights, nan_embed


def get_init_embed_weight():
    global embed_weights

    if embed_weights is None:
        _weight, nan_embed = get_embed_weights('glove')
        embed_weights = _weight
        embed_weights[np.isnan(embed_weights).any(axis=1)] = nan_embed
    return embed_weights


def get_nn_layer(x, feats, extra_inputs):
    if extra_inputs:
        text_input = extra_inputs[0]
    else:
        text_input = Input(shape=(seq_len,), name='text')
        extra_inputs = [text_input]

    _weights = get_init_embed_weight()
    embed0 = Embedding(_weights.shape[0], _weights.shape[1], weights=[_weights], trainable=False)(text_input)
    embed0 = SpatialDropout1D(0.2)(embed0)

    rnn0 = CuDNNGRU(64, return_sequences=True)(embed0)
    rnn1 = CuDNNGRU(64, return_sequences=True, go_backwards=True)(rnn0)
    # rnn1 = CuDNNLSTM(64, return_sequences=True, go_backwards=True)(rnn0)
    rnn0 = GlobalMaxPool1D()(rnn0)
    rnn1 = GlobalMaxPool1D()(rnn1)
    feats = [rnn0, rnn1]

    pool0 = GlobalMaxPool1D()(embed0)
    cnn_feats = [pool0]
    if eid % 2:
        cnn = Conv1D(80, kernel_size=1, padding='same', activation='tanh')(embed0)
        pool = Dropout(0.2)(GlobalMaxPool1D()(cnn))
        cnn_feats.append(pool)
    cnn = Conv1D(80, kernel_size=2, padding='same', activation='tanh')(embed0)
    pool = Dropout(0.2)(GlobalMaxPool1D()(cnn))
    cnn_feats.append(pool)
    cnn = Conv1D(80, kernel_size=3, padding='same', activation='tanh')(embed0)
    pool = Dropout(0.2)(GlobalMaxPool1D()(cnn))
    cnn_feats.append(pool)
    if eid // 2 == 1:
        feats.append(add_dense(concatenate(cnn_feats), 64, bn=True, activation='relu', dropout=0.05))
    elif eid // 2 == 2:
        feats.append(add_dense(concatenate(cnn_feats), 64, bn=False, activation='relu', dropout=0.05))
    else:
        feats.extend(cnn_feats)

    feats = concatenate(feats) if len(feats) > 1 else feats[0]
    return feats, extra_inputs


def get_dense_layers(feats, extra_feats, hidden_units=(320, 64), hidden_activation=seu, hidden_dropouts=(0.2, 0.05)):
    flat = []
    if hidden_units and feats is not None:
        hidden_layer_num = len(hidden_units)
        for i in range(hidden_layer_num):
            feats = add_dense(feats, hidden_units[i], bn=i < hidden_layer_num - 1, activation=hidden_activation,
                              dropout=hidden_dropouts[i])
        flat.append(feats)
    if extra_feats is not None:
        flat.append(extra_feats)

    flat = concatenate(flat) if len(flat) > 1 else flat[0]
    return flat


class EVPerEpoch(Callback):
    def __init__(self, vx, vy, ts_x, ps, p_ths, pred_batch_size=5000):
        super(EVPerEpoch, self).__init__()
        self.vx = vx
        self.vy = vy
        self.ts_x = ts_x
        self.ps = ps
        self.p_ths = p_ths
        self.pred_batch_size = pred_batch_size

    def on_epoch_end(self, epoch, logs=None):
        vp = np.squeeze(self.model.predict(self.vx, batch_size=self.pred_batch_size))
        f1(self.vy, vp, False)
        self.p_ths.append(last_best_p_th)
        p = np.squeeze(self.model.predict(self.ts_x, batch_size=self.pred_batch_size))
        self.ps.append(p)


def run_nn(tr_x, tr_y, seg_out_dims, pred_batch_size=5000):
    global last_best_p_th, eid

    iind, oind = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=oof_seed * 10000).split(tr_y))
    tr_x, tr_y, ts_x, ts_y = get_ab_split(tr_x, tr_y, iind, oind)
    print(f'tr_y: {tr_y.shape}, ts_y: {ts_y.shape}')
    print(f'tr_y(>0): {np.sum(tr_y)}, ts_y(>0): {np.sum(ts_y)}')
    gc.collect()

    k = 4
    d_seed = oof_seed * 10000 + pred_type_id * 1000
    fold_inds = list(KFold(n_splits=k, shuffle=True, random_state=d_seed).split(tr_y))

    min_delta = 2e-4
    params = {'get_output': get_sigmoid_output, 'compile_func': compile_default_bce_output,
              'seg_out_dims': seg_out_dims, 'seg_x_val_range': (0, 1), 'use_fm': use_fm, 'seg_flag': seg_flag,
              'add_seg_src': add_seg_src, 'seg_num_flag': seg_num_flag, 'fm_dim': fm_dim, 'fm_activation': 'softsign',
              'hidden_units': hidden_units, 'hidden_dropouts': hidden_dropouts, 'seg_func': seg_func,
              'hidden_activation': hidden_activation, 'get_extra_layers': get_nn_layer,
              'get_last_layers': get_dense_layers}
    epoch_list = [20] * k
    eid_list = [2] * k
    class_weights = [2] * k

    with timer('train'):
        pss, p_thss = [[] for i in range(4)], [[] for i in range(4)]
        pss1, p_thss1 = [[] for i in range(k)], [[] for i in range(k)]
        for i, (tind, vind) in enumerate(fold_inds):
            seed = d_seed + (i + 1) * 100
            model_id = f'tnn_{seed}'
            tx, ty, vx, vy = get_ab_split(tr_x, tr_y, tind, vind)
            print(f'ty: {ty.shape}, vy: {vy.shape}')
            print(f'ty(>0): {np.sum(ty)}, vy(>0): {np.sum(vy)}')
            gc.collect()

            checkpointer1 = ModelCheckpoint(f'{model_id}_best', verbose=1, save_best_only=True, save_weights_only=True)
            checkpointer2 = CheckpointDecorator(f'{model_id}_steady', verbose=1, save_best_only=True,
                                                save_weights_only=True)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1,
                                             min_lr=1e-5, min_delta=min_delta)
            early_stopper = EarlyStopping(min_delta=min_delta, patience=stop_patience, verbose=1)
            ev = EVPerEpoch(vx, vy, ts_x, pss1[i], p_thss1[i], pred_batch_size)

            eid = eid_list[i]
            model = get_tnn_model(tx, **params)
            # model.summary()

            set_seed(seed)
            model.fit(tx, ty, epochs=epoch_list[i], batch_size=batch_size, validation_data=(vx, vy), verbose=2,
                      class_weight={0: 1, 1: class_weights[i]},
                      callbacks=[ev, checkpointer1, checkpointer2, lr_scheduler, early_stopper])

            tags = ['best', 'steady']
            v_aucs, v_recalls = [], []
            vp_ths, ps = [], []

            with timer('validation & predict'):
                for tag in tags:
                    model = read_weights(model, f'{model_id}_{tag}')

                    vp = np.squeeze(model.predict(vx, batch_size=pred_batch_size))
                    v_auc = metrics.roc_auc_score(vy, vp)
                    f1(vy, vp, False)
                    vp_ths.append(last_best_p_th)
                    tn, fp, fn, tp = list(metrics.confusion_matrix(vy, vp > last_best_p_th).ravel())
                    v_aucs.append(v_auc)
                    v_recalls.append(tp / (tp + fn))

                    p = np.squeeze(model.predict(ts_x, batch_size=pred_batch_size))
                    ps.append(p)

                pss[0].append(ps[0])
                pss[1].append(ps[1])
                p_thss[0].append(vp_ths[0])
                p_thss[1].append(vp_ths[1])

                idx = np.argmax(v_aucs)
                pss[2].append(ps[idx])
                p_thss[2].append(vp_ths[idx])

                idx = np.argmax(v_recalls)
                pss[3].append(ps[idx])
                p_thss[3].append(vp_ths[idx])

        joblib.dump(pss, 'pss', compress=('gzip', 3))
        joblib.dump(ts_y, 'ts_y', compress=('gzip', 3))
        joblib.dump(p_thss, 'p_thss', compress=('gzip', 3))
        joblib.dump(pss1, 'pss1', compress=('gzip', 3))
        joblib.dump(p_thss1, 'p_thss1', compress=('gzip', 3))

    with timer('merge'):
        def blend(_ps, _method):
            if 'gm' == _method:
                return np.exp(np.mean(np.log(_ps), axis=0))
            if 'hm' == _method:
                return 1 / np.mean(1 / np.array(_ps), axis=0)
            if 'am' == _method:
                return 1 / (1 + np.exp(-np.mean(-np.log(1 / np.array(_ps) - 1), axis=0)))
            return np.mean(_ps, axis=0)

        pss, p_thss = pss[2:], p_thss[2:]
        measures = ['a', 'r']
        methods = ['m', 'am']
        for i, measure in enumerate(measures):
            ps, pths = pss[i], p_thss[i]
            for method in methods:
                tag = f'{measure}-{method}'
                ts_p = blend(ps, method)
                v_auc = metrics.roc_auc_score(ts_y, ts_p)
                last_best_p_th = blend(pths, method)
                ts_p = (ts_p > last_best_p_th).astype(np.uint8)
                v_f1 = metrics.f1_score(ts_y, ts_p)
                tn, fp, fn, tp = list(metrics.confusion_matrix(ts_y, ts_p).ravel())
                print(f'last_best_p_th={last_best_p_th}, v_cm({tag}): tn,fp,fn,tp={tn,fp,fn,tp}')
                print(f'v_auc({tag}): {v_auc}, v_f1: {v_f1}, v_precision: {tp/(tp+fp)}, v_recall: {tp/(tp+fn)}')
                print(f'--------------------------------------{tag}--------------------------------------')
        min_epochs = min([len(ps1) for ps1 in pss1])
        for i in range(min_epochs):
            ps, pths = [ps1[i] for ps1 in pss1], [pths1[i] for pths1 in p_thss1]
            for method in methods:
                tag = f'{i}{method}'
                ts_p = blend(ps, method)
                v_auc = metrics.roc_auc_score(ts_y, ts_p)
                last_best_p_th = blend(pths, method)
                ts_p = (ts_p > last_best_p_th).astype(np.uint8)
                v_f1 = metrics.f1_score(ts_y, ts_p)
                tn, fp, fn, tp = list(metrics.confusion_matrix(ts_y, ts_p).ravel())
                print(f'last_best_p_th={last_best_p_th}, v_cm({tag}): tn,fp,fn,tp={tn,fp,fn,tp}')
                print(f'v_auc({tag}): {v_auc}, v_f1: {v_f1}, v_precision: {tp/(tp+fp)}, v_recall: {tp/(tp+fn)}')
                print(f'--------------------------------------{tag}--------------------------------------')


def run():
    sys.stderr = sys.stdout = open(os.path.join('log.txt'), 'w')

    tr_x, y, seg_out_dims, ts_x, submission = get_data()
    run_nn(tr_x, y, seg_out_dims)


if __name__ == '__main__':
    run()
