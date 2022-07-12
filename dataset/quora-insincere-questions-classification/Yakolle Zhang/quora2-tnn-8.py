import gc
import os
import random
import re
import string
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors as wv
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn.init import uniform_, constant_, xavier_uniform_, zeros_
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


# --------------------------------------------------activations-----------------------------------------------------
def seu(x):
    return torch.sigmoid(x) * x


def make_lseu(c_val):
    def _lseu(x):
        x1 = torch.sigmoid(x) * x
        x2 = c_val + torch.log(1 + torch.relu(x - c_val))
        return torch.min(x1, x2)

    return _lseu


class PLSEU(nn.Module):
    def __init__(self, input_dim, alpha_init_val=4.4, alpha_min_value=1e-3):
        super(PLSEU, self).__init__()
        self.alpha_min_value = alpha_min_value
        self.alpha = constant_(nn.Parameter(torch.Tensor(input_dim)), alpha_init_val)

    def forward(self, x):
        self.alpha.clamp_(min=self.alpha_min_value)

        x1 = torch.sigmoid(x / self.alpha) * x
        x2 = self.alpha * (self.alpha + torch.log(1 + torch.relu(x / self.alpha - self.alpha)))
        return torch.min(x1, x2)


def make_plseu(alpha):
    def _plseu(x):
        return PLSEU(x.size(-1), alpha_init_val=alpha)(x)

    return _plseu


# --------------------------------------------------callbacks-----------------------------------------------------
def calc_score(logs, monitor='loss', monitor_op=np.less):
    score = None
    t_score = logs.get(monitor)
    v_score = logs.get(f'val_{monitor}')
    if t_score is not None and v_score is not None:
        score = v_score ** 2 + (1 if monitor_op == np.less else -1) * (t_score - v_score) ** 2
    return score


def get_score(logs, monitor='val_loss', monitor_op=np.less):
    return logs.get(monitor)


class CheckpointDecorator(object):
    def __init__(self, model, filepath, monitor='loss', calc_score_func=calc_score, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.model = model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(f'ModelCheckpoint mode {mode} is unknown, fallback to auto mode.', RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

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
                    warnings.warn(f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)


class CyclicLR(object):
    def __init__(self, optimizer, init_base_lr=1e-3, init_max_lr=6e-3, half_step_size=500, gamma=1.0,
                 get_lr_func=None,
                 factor=0.5, min_lr=1e-5, tol_rate=0.1, eps=1e-8):
        super(CyclicLR, self).__init__()

        self.optimizer = optimizer
        self.base_lr = init_base_lr
        self.max_lr = init_max_lr
        self.half_step_size = half_step_size
        self.gamma = gamma

        self.steps = 0
        self.shrink_factor = gamma

        self.get_lr_func = get_lr_func
        self.factor = factor
        self.min_lr = min_lr
        self.tol_rate = tol_rate
        self.eps = eps

        self.lr_loss_pairs = []

    def _clr(self):
        delta_lr = (self.max_lr - self.base_lr) / self.half_step_size
        step_offset = abs(self.half_step_size - self.steps % (2 * self.half_step_size))
        return self.max_lr - delta_lr * step_offset

    def _set_lr(self, new_lr, detail=False):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            if abs(old_lr - new_lr) > self.eps:
                param_group['lr'] = new_lr
                if not i and detail:
                    print('Reducing learning rate to %s.' % new_lr)

    def on_train_begin(self, logs=None):
        if self.steps:
            self._set_lr(self._clr())
        else:
            self._set_lr(self.base_lr)

    def on_batch_end(self, batch, logs=None):
        self.steps += 1
        if not self.steps % (2 * self.half_step_size):
            self.max_lr = self.base_lr + (self.max_lr - self.base_lr) * self.gamma

        self._set_lr(self._clr())

    def on_epoch_end(self, epoch, logs=None):
        pass


# --------------------------------------------------FMLayer-----------------------------------------------------
class FMLayer(nn.Module):
    def __init__(self, input_dim, factor_rank, activation=nn.Softsign, use_bias=False,
                 kernel_initializer=xavier_uniform_, bias_initializer=zeros_):
        super(FMLayer, self).__init__()
        self.activation = activation

        self.kernel = nn.Parameter(torch.Tensor(input_dim, factor_rank))
        kernel_initializer(self.kernel)

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(factor_rank))
            bias_initializer(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        prod = torch.mm(inputs, self.kernel)
        output = (prod * prod - torch.mm(inputs * inputs, self.kernel * self.kernel)) / 2
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


# --------------------------------------------------SegTriangleLayer-----------------------------------------------------
class SegTriangleLayer(nn.Module):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu):
        super(SegTriangleLayer, self).__init__()
        self.seg_func = seg_func

        seg_width = (input_val_range[1] - input_val_range[0]) / seg_num
        left_pos = input_val_range[0] + seg_width
        right_pos = input_val_range[1] - seg_width

        self.left_pos = constant_(nn.Parameter(torch.Tensor(1)), left_pos)
        if seg_num > 2:
            self.middle_pos = uniform_(nn.Parameter(torch.Tensor(seg_num - 2)), left_pos, right_pos - seg_width)
        else:
            self.middle_pos = None
        self.right_pos = constant_(nn.Parameter(torch.Tensor(1)), right_pos)

        if seg_num > 2:
            self.middle_seg_width = constant_(nn.Parameter(torch.Tensor(seg_num - 2)), seg_width)
        else:
            self.middle_seg_width = None

    def forward(self, inputs):
        left_out = self.left_pos - inputs
        middle_out = None if self.middle_pos is None else -torch.abs(inputs - self.middle_pos) + self.middle_seg_width
        right_out = inputs - self.right_pos

        if self.middle_pos is not None:
            output = torch.cat([left_out, middle_out, right_out], -1)
        else:
            output = torch.cat([left_out, right_out], -1)
        return self.seg_func(output)


# --------------------------------------------------util-----------------------------------------------------
@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


# --------------------------------------------------nn_util-----------------------------------------------------
def set_seed(_seed=10000):
    os.environ['PYTHONHASHSEED'] = str(_seed + 5)
    np.random.seed(_seed + 6)
    random.seed(_seed + 7)
    torch.manual_seed(_seed + 8)
    torch.cuda.manual_seed(_seed + 9)
    torch.backends.cudnn.deterministic = True


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
    model.load_state_dict(torch.load(weights_path))
    return model


def shrink(dim, shrink_factor):
    if dim > 10:
        return max(10, int(dim * shrink_factor))
    return dim


class DenseLayer(nn.Module):
    def __init__(self, input_dim, units, bn=True, activation=seu, dropout=0.2):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_dim, units)
        self.bn = nn.BatchNorm1d(units) if bn else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, inputs):
        output = self.linear(inputs)
        if self.bn is not None:
            output = self.bn(output)
        if self.activation is not None:
            output = self.activation(output)
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class Cats(nn.Module):
    def __init__(self, cat_in_dims, cat_out_dims, shrink_factor=1.0):
        super(Cats, self).__init__()

        self.embeds = []
        for i, in_dim in enumerate(cat_in_dims):
            self.embeds.append(nn.Embedding(in_dim, shrink(cat_out_dims[i], shrink_factor)))

    def forward(self, inputs):
        embeds = []
        for i, embed in enumerate(self.embeds):
            embed = embed(inputs[:, i, None])
            embeds.append(torch.flatten(embed))
        return embeds

    def _apply(self, fn):
        for embed in self.embeds:
            embed._apply(fn)
        return self


class Segs(nn.Module):
    def __init__(self, seg_out_dims, shrink_factor=1.0, seg_func=seu, seg_input_val_range=(0, 1)):
        super(Segs, self).__init__()

        self.segments = []
        for out_dim in seg_out_dims:
            self.segments.append(SegTriangleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                                  seg_func=seg_func))

    def forward(self, inputs):
        segments = []
        for i, segment in enumerate(self.segments):
            segments.append(segment(inputs[:, i, None]))
        return segments

    def _apply(self, fn):
        for seg in self.segments:
            seg._apply(fn)
        return self


# --------------------------------------------------Tokenizer-----------------------------------------------------
def texts_to_word_sequence(texts, token_pattern=r'(?u)\b\w\w+\b', lower=True):
    token_pattern = re.compile(token_pattern)
    token_table = []
    for text in texts:
        if lower:
            text = text.lower()
        token_table.append(token_pattern.findall(text))
    return token_table


class Tokenizer(object):
    def __init__(self, num_words=None, token_pattern=r'(?u)\w+|[^\w\s]', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
                 lower=True, reserve_filter_index=True, reserve_oov_index=True):
        self.num_words = num_words
        self.token_pattern = token_pattern
        self.filters = filters
        self.lower = lower
        self.reserve_filter_index = reserve_filter_index
        self.reserve_oov_index = reserve_oov_index
        self.word_counts = OrderedDict()
        self.word_index = None

    def fit_on_texts(self, texts):
        token_table = texts_to_word_sequence(texts, self.token_pattern, self.lower)

        for seq in token_table:
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
        if self.filters is not None:
            for ch in self.filters:
                del self.word_counts[ch]

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        start_index = 1 + self.reserve_oov_index + (self.filters is not None and self.reserve_filter_index)
        self.word_index = dict(zip(sorted_voc, list(range(start_index, len(sorted_voc) + start_index))))
        if self.filters is not None:
            if self.reserve_filter_index:
                for ch in self.filters:
                    self.word_index[ch] = 1 + self.reserve_oov_index
            else:
                for ch in self.filters:
                    self.word_index[ch] = 0

        return token_table

    def to_sequences(self, token_table):
        res = []
        num_words = self.num_words
        for seq in token_table:
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None and (not num_words or i < num_words):
                    if i > 0:
                        vect.append(i)
                elif self.reserve_oov_index:
                    vect.append(1)
            res.append(vect)
        return res

    def fit_transform(self, texts):
        token_table = self.fit_on_texts(texts)
        return self.to_sequences(token_table)

    def fit(self, texts):
        return self.fit_on_texts(texts)

    def transform(self, texts):
        token_table = texts_to_word_sequence(texts, self.token_pattern, self.lower)
        return self.to_sequences(token_table)


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    num_samples = len(sequences)
    if maxlen is None:
        maxlen = 0
        for s in sequences:
            s_len = len(s)
            if s_len > maxlen:
                maxlen = s_len

    x = (np.ones((num_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


# --------------------------------------------------tnn_submit-----------------------------------------------------
last_best_p_th = 0.5

oof_seed = 3
run_seed = 1
pred_type_id = 3

vocab_size = None
embed_dim = 300
seq_len = 56
word_index = {}
embed_weights = None

hidden_units = (64,)
hidden_dropouts = (0.05,)

lseu = make_lseu(0.9)
plseu = make_plseu(0.65)

seg_func = torch.relu
hidden_activation = torch.relu

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

    with timer('to text sequence'):
        tkr = Tokenizer(num_words=vocab_size, filters='\u200b', reserve_filter_index=False, reserve_oov_index=False)
        s = train_df.question_text.append(test_df.question_text, ignore_index=True)
        nn_x = tkr.fit_transform(s)
        word_index.update(tkr.word_index)

        nn_tr_x = nn_x[:train_df.shape[0]]
        nn_ts_x = nn_x[train_df.shape[0]:]
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


def to_torch_dataloader(x, y=None, bs=batch_size, shuffle=False):
    segs, text = x['segs'], x['text']
    segs = torch.tensor(segs, dtype=torch.float32).cuda()
    text = torch.tensor(text, dtype=torch.int64).cuda()

    if y is not None:
        y = torch.tensor(y[:, None], dtype=torch.float32).cuda()
        data = TensorDataset(segs, text, y)
        data_loader = DataLoader(data, batch_size=bs, shuffle=shuffle)
    else:
        data = TensorDataset(segs, text)
        data_loader = DataLoader(data, batch_size=bs, shuffle=shuffle)

    return data_loader


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
        nan_embed = embed_dic['the'].copy()

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


class TNN(nn.Module):
    def __init__(self, seg_out_dims):
        super(TNN, self).__init__()
        out_dim = 0

        self.segs = Segs(seg_out_dims, seg_func=seg_func)
        self.seg_hiddens = []
        hidden_layer_num = len(hidden_units)
        hidden_input_dim = sum(seg_out_dims)
        for i in range(hidden_layer_num):
            self.seg_hiddens.append(DenseLayer(hidden_input_dim, hidden_units[i], bn=i < hidden_layer_num - 1,
                                               activation=hidden_activation, dropout=hidden_dropouts[i]))
            hidden_input_dim = hidden_units[i]
        out_dim += hidden_input_dim

        _embed_weights = get_init_embed_weight()
        _embed_dim = _embed_weights.shape[1]
        self.embed0 = nn.Embedding(_embed_weights.shape[0], _embed_dim)
        self.embed0.weight = nn.Parameter(torch.tensor(_embed_weights, dtype=torch.float32))
        self.embed0.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)

        rnn_units = 64
        self.rnn0 = nn.GRU(_embed_dim, rnn_units, batch_first=True, bidirectional=True)
        self.rnn1 = nn.GRU(rnn_units * 2, rnn_units, batch_first=True, bidirectional=True)
        out_dim += rnn_units * 4

        cnn_units = 80
        self.cnn_parts = []
        kernel_sizes = [1, 2, 3] if eid % 2 else [2, 3]
        for kernel_size in kernel_sizes:
            cnn = nn.Conv1d(_embed_dim, cnn_units, kernel_size, padding=kernel_size - 1)
            self.cnn_parts.append([cnn, nn.Dropout(0.2)])
        self.cnn_hidden = None
        cnn_input_dim = cnn_units * len(kernel_sizes) + _embed_dim
        cnn_out_dim = 64
        if eid // 2 == 1:
            self.cnn_hidden = DenseLayer(cnn_input_dim, cnn_out_dim, bn=True, activation=torch.relu, dropout=0.05)
        elif eid // 2 == 2:
            self.cnn_hidden = DenseLayer(cnn_input_dim, cnn_out_dim, bn=False, activation=torch.relu, dropout=0.05)
        else:
            cnn_out_dim = cnn_input_dim
        out_dim += cnn_out_dim

        self.output = nn.Linear(out_dim, 1)

    def forward(self, inputs):
        num_input, text_input = inputs
        feats = torch.cat(self.segs(num_input), -1)
        for hidden in self.seg_hiddens:
            feats = hidden(feats)
        feats = [feats]

        embed0 = self.embed0(text_input)
        embed0 = self.embedding_dropout(embed0.permute(0, 2, 1))

        rnn0 = self.rnn0(embed0.permute(0, 2, 1))[0]
        rnn1 = self.rnn1(rnn0)[0]
        rnn0 = torch.max(rnn0, 1)[0]
        rnn1 = torch.max(rnn1, 1)[0]
        feats.extend([rnn0, rnn1])

        cnn_feats = [torch.max(embed0, -1)[0]]
        for cnn, cnn_dropout in self.cnn_parts:
            cnn_feats.append(cnn_dropout(torch.max(cnn(embed0), -1)[0]))
        if self.cnn_hidden is not None:
            cnn_feats = self.cnn_hidden(torch.cat(cnn_feats, -1))
            feats.append(cnn_feats)
        else:
            feats.extend(cnn_feats)

        output = self.output(torch.cat(feats, -1))
        return output

    def _apply(self, fn):
        for hidden in self.seg_hiddens:
            hidden._apply(fn)
        for cnn, dropout in self.cnn_parts:
            cnn._apply(fn)
            dropout._apply(fn)
        return super(TNN, self)._apply(fn)


def predict(model, data_loader, loss_func=None):
    model.eval()
    ps = []
    if loss_func is not None:
        total_loss = 0.0
        for b_segs, b_text, by in data_loader:
            bp = model((b_segs, b_text)).detach()
            loss = loss_func(bp, by)
            total_loss += loss.item()
            bp = np.squeeze(bp.cpu().numpy())
            ps.append(1 / (1 + np.exp(-bp)))
        p = np.hstack(ps)
        total_loss /= p.shape[0]
        return p, total_loss
    else:
        for b_segs, b_text in data_loader:
            bp = model((b_segs, b_text)).detach()
            bp = np.squeeze(bp.cpu().numpy())
            ps.append(1 / (1 + np.exp(-bp)))
        p = np.hstack(ps)
        return p


def run_nn(tr_x, y, seg_out_dims, ts_x, pred_batch_size=5000):
    global last_best_p_th, eid

    ts_loader = to_torch_dataloader(ts_x, bs=pred_batch_size)

    k = 5
    d_seed = oof_seed * 10000 + pred_type_id * 1000
    fold_inds = list(KFold(n_splits=k, shuffle=True, random_state=d_seed).split(y))

    epoch_list = [8] * k
    eid_list = [2] * k
    class_weights = [2] * k

    with timer('train'):
        ps, vps, vp_ths = [], [], []
        for i, (tind, vind) in enumerate(fold_inds):
            seed = d_seed + (i + 1) * 100 + run_seed * 10
            set_seed(seed)
            tx, ty, vx, vy = get_ab_split(tr_x, y, tind, vind)
            print(f'ty: {ty.shape}, vy: {vy.shape}')
            print(f'ty(>0): {np.sum(ty)}, vy(>0): {np.sum(vy)}')
            gc.collect()
            t_loader = to_torch_dataloader(tx, ty, bs=batch_size, shuffle=True)
            v_loader = to_torch_dataloader(vx, vy, bs=pred_batch_size)

            eid = eid_list[i]
            model = TNN(seg_out_dims).cuda()
            # print(model)
            loss_func = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.Tensor([class_weights[i]]).cuda())
            optimizer = Adam(model.parameters())

            lr_scheduler = CyclicLR(optimizer, 1e-3, 5e-3, half_step_size=150, gamma=0.9)

            vp = None
            lr_scheduler.on_train_begin()
            for epoch in range(epoch_list[i]):
                start_time = time.time()
                model.train()
                t_loss = 0.
                batch = 0
                for b_segs, b_text, by in t_loader:
                    bp = model((b_segs, b_text))
                    loss = loss_func(bp, by)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()

                    batch += 1
                    lr_scheduler.on_batch_end(batch)
                t_loss /= ty.shape[0]

                vp, v_loss = predict(model, v_loader, loss_func)
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epoch_list[i], t_loss, v_loss, time.time() - start_time))

            with timer('validation & predict'):
                vps.append(vp)
                v_auc = metrics.roc_auc_score(vy, vp)
                v_f1 = f1(vy, vp, True)
                vp_ths.append(last_best_p_th)
                tn, fp, fn, tp = list(metrics.confusion_matrix(vy, vp > last_best_p_th).ravel())
                print(f'v_cm: tn,fp,fn,tp={tn,fp,fn,tp}')
                print(f'v_auc: {v_auc}, v_f1: {v_f1}, v_precision: {tp/(tp+fp)}, v_recall: {tp/(tp+fn)}')
                print()

                p = predict(model, ts_loader)
                ps.append(p)

        joblib.dump(ps, 'ps', compress=('gzip', 3))
        joblib.dump(vps, 'vps', compress=('gzip', 3))
        joblib.dump(vp_ths, 'vp_ths', compress=('gzip', 3))
    return np.mean(ps, axis=0), np.mean(vp_ths)


def run():
    sys.stderr = sys.stdout = open(os.path.join('log.txt'), 'w')

    tr_x, y, seg_out_dims, ts_x, submission = get_data()
    tnn_p, tnn_p_th = run_nn(tr_x, y, seg_out_dims, ts_x)

    submission['prediction'] = (tnn_p > tnn_p_th).astype(np.uint8)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()
