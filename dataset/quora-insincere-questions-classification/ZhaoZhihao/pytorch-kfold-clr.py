#!/usr/bin/env python
# coding: utf-8

from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from tqdm import tqdm
import math
import random
from datetime import datetime
import time
import gc
import os

from sklearn import model_selection, preprocessing, metrics, linear_model
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.utils import shuffle
from sklearn.externals import joblib

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, TensorDataset, DataLoader

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

start_time = datetime.now()
tqdm.pandas()
pd.set_option('max_colwidth', 200)

embed_size = 300  # word vector 大小
max_features = 100000  # 要使用多少个 unique words (即 embedding vector num rows)
maxlen = 70  # 问题中要使用的 max words

num_folds = 6
batch_size = 256
epochs = 3
batch_size_pred = 1024
base_lr=0.0005
max_lr=0.002
step_size=2300

gpu_device_count = '0'
ckpt_path = '0best_model'

seed = 2018
scoring = 'f1'
n_jobs = -1
verbose = 0
print_every = 10
plot_every = 5

def set_seed(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.set_random_seed(seed)
set_seed()

import re
import string
from unidecode import unidecode
from gensim.models import KeyedVectors
import gensim
from wordcloud import STOPWORDS

others = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\u2061', '\x10', '\x7f', '\x9d', '\xad']
def clean_other(sentence):
    sentence = str(sentence)
    for other in others:
        sentence = sentence.replace(other, '')
    sentence = sentence.replace('\xa0', ' ')
    return sentence

symbols = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '´', '«', '？', '،', '✔', '。', '‛', '„', ]
def clean_symbol(sentence):
    sentence = str(sentence)
    for symbol in symbols:
        sentence = sentence.replace(symbol, f' {symbol} ')
    return sentence

'''获得F1分数的最佳阈值-faster'''
def threshold_search(y_test, y_score, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    thresholds = np.append(thresholds, 1.001)
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_threshold = thresholds[np.argmax(F)]
    if plot:
        plt.plot(thresholds, F, '-b')
        plt.plot([best_threshold], [best_score], '*r')
        plt.plt.savefig('F1_threshold_search.png')
    search_result = {'best_threshold': best_threshold, 'best_score': best_score}
    return search_result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''save and load model'''
def save(model, info, ckpt_path='best_model'):
    torch.save(model, ckpt_path+'.m')
    torch.save(info, ckpt_path+'.info')
    
def load(ckpt_path='best_model'):
    model = torch.load(ckpt_path+'.m')
    info = torch.load(ckpt_path+'.info')
    return model, info


'''pytorch-Attention'''
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


'''pytorch-Cyclical Learning Rate'''
class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


def loaddata_and_process():
    train_df = pd.read_csv('../input/train.csv', nrows=None)
    test_df = pd.read_csv('../input/test.csv', nrows=None)
    # train-(1306122, 3), pos-0-1225312 : neg-1-80810 = 15.16 : 1
    # test-(56370, 2)
    
    train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: clean_symbol(x))
    test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: clean_symbol(x))
    
    train_X = train_df['question_text'].values
    test_X = test_df['question_text'].values
    
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X)+list(test_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    
    train_y = train_df['target'].values
    np.random.seed(seed)
    train_idx = np.random.permutation(len(train_X))
    train_X = train_X[train_idx]
    train_y = train_y[train_idx]
    
    return train_X, train_y, test_X, test_df, tokenizer.word_index
    
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def embed_matrix(embeddings_index, max_features, vocab):
    embeddings_index_arr = np.stack(embeddings_index.values())
    embed_size = embeddings_index_arr.shape[1]
    max_features = min(max_features, len(vocab))
    embedding_matrix = np.random.normal(loc=embeddings_index_arr.mean(), scale=embeddings_index_arr.std(), size=(max_features, embed_size))
    
    for word, cnt in tqdm(vocab.items()):
        if cnt >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[cnt, :embed_size] = embedding_vector
                
    return embedding_matrix

def load_glove(max_features, vocab):
    fname = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*f.split(" ")) for f in open(fname))  # (2196016, 300)

    embedding_matrix = embed_matrix(embeddings_index, max_features, vocab)

    return embedding_matrix

def load_paragram(max_features, vocab):
    fname = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    embeddings_index = dict(
        get_coefs(*f.split(" ")) for f in open(fname, encoding='utf-8', errors='ignore'))  # (1703756, 300)

    embedding_matrix = embed_matrix(embeddings_index, max_features, vocab)

    return embedding_matrix
    
# *Baseline Model*
'''gru_atten-512 3 gp√'''
class gru_atten_Model(nn.Module):
    def __init__(self, embedding_matrix):
        super(gru_atten_Model, self).__init__()
        dropout_rate = 0.1
        lstm_hidden_size = 96
        gru_hidden_size = 96
        embedding_matrix = embedding_matrix
        embedding_trainable = False
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = embedding_trainable
        
        self.embedding_dropout = nn.Dropout2d(dropout_rate)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(input_size=lstm_hidden_size * 2, hidden_size=gru_hidden_size, bidirectional=True, batch_first=True)
        self.lstm_atten = Attention(lstm_hidden_size * 2, maxlen)
        self.gru_atten = Attention(gru_hidden_size * 2, maxlen)
        self.linear = nn.Linear(768, 64)
        self.bn = nn.BatchNorm1d(64, momentum=0.01)
        self.out = nn.Linear(64, 1)
    
    def forward(self, x):
        embeds = self.embedding(x)
        embeds = torch.squeeze(self.embedding_dropout(torch.unsqueeze(embeds, 0)))
        
        lstm_out, _ = self.lstm(embeds)
        gru_out, _ = self.gru(lstm_out)
        lstm_atten = self.lstm_atten(lstm_out)
        gru_atten = self.gru_atten(gru_out)
        avg_pool = torch.mean(gru_out, 1)
        max_pool, _ = torch.max(gru_out, 1)
        conc = torch.cat((lstm_atten, gru_atten, avg_pool, max_pool), 1)
        conc = F.relu(self.linear(conc))
        conc = self.bn(conc)
        out = self.out(conc)
        
        return out

def kfold_train_val_pred(train_X, train_y, test_X, num_folds, batch_size, epochs, clip=True):
    kfold = list(model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed).split(train_X, train_y))
    
    y_score = np.zeros((train_X.shape[0],))
    y_pred = np.zeros((test_X.shape[0],))
    
    set_seed()
    
    test_X_cuda = torch.tensor(test_X, dtype=torch.long).cuda()
    test = TensorDataset(test_X_cuda)
    test_loader = DataLoader(test, batch_size=batch_size_pred, shuffle=False)
    
    avg_losses = []
    avg_val_losses = []
    for i, (train_index, val_index) in enumerate(kfold):
        X_train_cuda = torch.tensor(train_X[train_index], dtype=torch.long).cuda()
        y_train_cuda = torch.tensor(train_y[train_index, np.newaxis], dtype=torch.float32).cuda()
        X_val_cuda = torch.tensor(train_X[val_index], dtype=torch.long).cuda()
        y_val_cuda = torch.tensor(train_y[val_index, np.newaxis], dtype=torch.float32).cuda()

        model = gru_atten_Model(embedding_matrix_gp)
        model.cuda()

        loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    #     optimizer = optim.Adam(model.parameters())
    #     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        train = TensorDataset(X_train_cuda, y_train_cuda)
        val = TensorDataset(X_val_cuda, y_val_cuda)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size_pred, shuffle=False)
        
        print('{0} fold, train {1}, val {2}'.format(i+1, len(train_index), len(val_index)))
        min_val_loss = 1000.
        model_path = ckpt_path+'_kf'+str(i+1)
        for epoch in range(epochs):
            epoch_start_time = time.time()

            scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode='exp_range', gamma=0.99994)

            model.train()
            avg_loss = 0.
            for step, (x_batch, y_batch) in enumerate(train_loader):
                if scheduler:
                    scheduler.batch_step()
                y_batch_train_score = model(x_batch)
                loss = loss_function(y_batch_train_score, y_batch)
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                
            model.eval()
            y_score_kf = np.zeros((X_val_cuda.size(0),))
            avg_val_loss = 0.
            for step, (x_batch, y_batch) in enumerate(val_loader):
                y_batch_val_score = model(x_batch).detach()
                loss = loss_function(y_batch_val_score, y_batch)
                avg_val_loss += loss.item() / len(val_loader)
                y_score_kf[step * batch_size_pred:(step + 1) * batch_size_pred] = sigmoid(y_batch_val_score.cpu().numpy())[:, 0]

            epoch_spend_time = time.time() - epoch_start_time
            print('Epoch [{}/{}] - time: {:.2f}s - loss: {:.4f} - val_loss: {:.4f}'.format(epoch+1, epochs, epoch_spend_time, avg_loss, avg_val_loss))
            
        avg_losses.append(avg_loss)
        avg_val_losses.append(avg_val_loss)
        
        y_pred_kf = np.zeros((test_X.shape[0],))
        for step, (x_batch, ) in enumerate(test_loader):
            y_batch_pred = model(x_batch).detach()
            y_pred_kf[step * batch_size_pred:(step + 1) * batch_size_pred] = sigmoid(y_batch_pred.cpu().numpy())[:, 0]

        y_score[val_index] = y_score_kf
        y_pred += y_pred_kf / len(kfold)
    
    print('Avg - loss: {:.4f} - val_loss: {:.4f} \t '.format(np.average(avg_losses),np.average(avg_val_losses)))
    search_result = threshold_search(train_y, y_score)
    print("f1_score at best_threshold {0} is {1}".format(search_result['best_threshold'], search_result['best_score']))
    
    return y_score, y_pred, search_result

train_X, train_y, test_X, test_df, vocab = loaddata_and_process()
print(train_X.shape, train_y.shape, test_X.shape, len(vocab))

embedding_matrix_g = load_glove(max_features, vocab)
embedding_matrix_p = load_paragram(max_features, vocab)
embedding_matrix_gp = np.mean([embedding_matrix_g, embedding_matrix_p], axis=0)

'''gru_atten_gp√'''
print(">>> gru_atten_gp:")
y_score, y_pred, search_result = kfold_train_val_pred(train_X, train_y, test_X, num_folds, batch_size, epochs)
    
y_pred = (y_pred > search_result['best_threshold']).astype(int)
result_df = pd.DataFrame({"qid": test_df["qid"].values})
result_df['prediction'] = y_pred
result_df.to_csv("submission.csv", index=False)

end_time = datetime.now()
print(">>>Total runtime %s." % (end_time - start_time))