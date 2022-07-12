#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-

import gc
from keras.preprocessing import text
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn import metrics
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
print(os.listdir("../input"))
import sys
package_dir = "../input/pytorchpretrainedbert/ppbert/pytorch-pretrained-BERT/pytorch-pretrained-BERT"
sys.path.append(package_dir)
from pytorch_pretrained_bert import BertModel, BertTokenizer

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
BERT_PATH = '../input/bert-inference-2/bert-inference-2'
CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = '../input/embeddings/glove.twitter.27B/glove.840B.300d.txt'
LSTM_UNITS = 128
GRU_UNITS = 128
MAX_LEN = 222
n_epochs = 1
batch_size = 16
num = 68
NUM_MODELS = 1


def convert_lines(train, valid, test, max_seq_length,tokenizer):
    max_seq_length -=2
    train_tokens = []
    valid_tokens = []
    test_tokens = []
    longer = 0
    for text in train:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        train_tokens.append(one_token)
    for text in valid:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        valid_tokens.append(one_token)
    for text in test:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        test_tokens.append(one_token)
    return np.array(train_tokens),np.array(valid_tokens),np.array(test_tokens)

class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule, dim_capsule, routings, kernel_size=(4, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)
        self.T_epsilon = 1e-7
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size  # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(128, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 128即batch_size

    def forward(self, x):

        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + self.T_epsilon)
        return x / scale


class Dense_Layer(nn.Module):
    def __init__(self, num_capsule, dim_capsule, num_classes):
        super(Dense_Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_capsule * dim_capsule, num_classes),  # num_capsule*dim_capsule -> num_classes
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self,num_aux_targets):
        super(NeuralNet, self).__init__()
        self.capsule1 = Caps_Layer(768, num_capsule= 68, dim_capsule=10, \
                                   routings=3 )
        self.capsule2 = Caps_Layer(10, num_capsule= 20, dim_capsule=10, \
                                   routings=3 )
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = SpatialDropout(0.3)
        self.dense1 = Dense_Layer(num_capsule= 20, dim_capsule=10, num_classes=1)
        # self.dense2 = Dense_Layer(num_capsule= 20, dim_capsule=10, num_classes=num_aux_targets)


    def forward(self, x_train):
        x_train = x_train.cuda()
        _, pooled = self.bert(x_train, output_all_encoded_layers=False)
        pooled = pooled.view(-1, 1, 768)
        pooled = self.dropout(pooled)
        caps1 = self.capsule1(pooled)
        caps2 = self.capsule2(caps1)
        # aux_result = self.dense2(caps2)
        result =  self.dense1(caps2)
        # out = torch.cat([result, aux_result], 1)
        return result.cpu()

class NeuralNet1(nn.Module):
    def __init__(self,num_aux_targets):
        super(NeuralNet1, self).__init__()
        self.capsule1 = Caps_Layer(768, num_capsule= 168, dim_capsule=10, \
                                   routings=3 )
        self.capsule2 = Caps_Layer(10, num_capsule= 20, dim_capsule=10, \
                                   routings=3 )
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = SpatialDropout(0.3)
        self.dense1 = Dense_Layer(num_capsule= 20, dim_capsule=10, num_classes=1)
        # self.dense2 = Dense_Layer(num_capsule= 20, dim_capsule=10, num_classes=num_aux_targets)


    def forward(self, x_train):
        x_train = x_train.cuda()
        _, pooled = self.bert(x_train, output_all_encoded_layers=False)
        pooled = pooled.view(-1, 1, 768)
        pooled = self.dropout(pooled)
        caps1 = self.capsule1(pooled)
        caps2 = self.capsule2(caps1)
        # aux_result = self.dense2(caps2)
        result =  self.dense1(caps2)
        # out = torch.cat([result, aux_result], 1)
        return result.cpu()
        
class NeuralNet2(nn.Module):
    def __init__(self,num_aux_targets):
        super(NeuralNet2, self).__init__()
        self.capsule = Caps_Layer(768, num_capsule= 68, dim_capsule=10, \
                                   routings=3 )
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = SpatialDropout(0.3)
        self.dense1 = Dense_Layer(num_capsule= num, dim_capsule=10, num_classes=1)
        self.dense2 = Dense_Layer(num_capsule= num, dim_capsule=10, num_classes=num_aux_targets)


    def forward(self, x_train):
        x_train = x_train.cuda()
        _, pooled = self.bert(x_train, output_all_encoded_layers=False)
        pooled = pooled.view(-1, 1, 768)
        caps1 = self.capsule(pooled)
        aux_result = self.dense2(caps1)
        result =  self.dense1(caps1)
        out = torch.cat([result, aux_result], 1)
        return out.cpu()       
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(data):
    '''
        Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
        '''
    punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
             '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '\n'
                                                                                                   '®', '`', '<', '→',
             '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
             '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
             '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
             '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
             'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
             '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
    contraction_mapping = {
        "Trump's": 'trump is', "'cause": 'because', ',cause': 'because', ';cause': 'because', "ain't": 'am not',
        'ain,t': 'am not',
        'ain;t': 'am not', 'ain´t': 'am not', 'ain’t': 'am not', "aren't": 'are not',
        'aren,t': 'are not', 'aren;t': 'are not', 'aren´t': 'are not', 'aren’t': 'are not', "can't": 'cannot',
        "can't've": 'cannot have', 'can,t': 'cannot', 'can,t,ve': 'cannot have',
        'can;t': 'cannot', 'can;t;ve': 'cannot have',
        'can´t': 'cannot', 'can´t´ve': 'cannot have', 'can’t': 'cannot', 'can’t’ve': 'cannot have',
        "could've": 'could have', 'could,ve': 'could have', 'could;ve': 'could have', "couldn't": 'could not',
        "couldn't've": 'could not have', 'couldn,t': 'could not', 'couldn,t,ve': 'could not have',
        'couldn;t': 'could not',
        'couldn;t;ve': 'could not have', 'couldn´t': 'could not',
        'couldn´t´ve': 'could not have', 'couldn’t': 'could not', 'couldn’t’ve': 'could not have',
        'could´ve': 'could have',
        'could’ve': 'could have', "didn't": 'did not', 'didn,t': 'did not', 'didn;t': 'did not', 'didn´t': 'did not',
        'didn’t': 'did not', "doesn't": 'does not', 'doesn,t': 'does not', 'doesn;t': 'does not', 'doesn´t': 'does not',
        'doesn’t': 'does not', "don't": 'do not', 'don,t': 'do not', 'don;t': 'do not', 'don´t': 'do not',
        'don’t': 'do not',
        "hadn't": 'had not', "hadn't've": 'had not have', 'hadn,t': 'had not', 'hadn,t,ve': 'had not have',
        'hadn;t': 'had not',
        'hadn;t;ve': 'had not have', 'hadn´t': 'had not', 'hadn´t´ve': 'had not have', 'hadn’t': 'had not',
        'hadn’t’ve': 'had not have', "hasn't": 'has not', 'hasn,t': 'has not', 'hasn;t': 'has not', 'hasn´t': 'has not',
        'hasn’t': 'has not',
        "haven't": 'have not', 'haven,t': 'have not', 'haven;t': 'have not', 'haven´t': 'have not',
        'haven’t': 'have not', "he'd": 'he would',
        "he'd've": 'he would have', "he'll": 'he will',
        "he's": 'he is', 'he,d': 'he would', 'he,d,ve': 'he would have', 'he,ll': 'he will', 'he,s': 'he is',
        'he;d': 'he would',
        'he;d;ve': 'he would have', 'he;ll': 'he will', 'he;s': 'he is', 'he´d': 'he would', 'he´d´ve': 'he would have',
        'he´ll': 'he will',
        'he´s': 'he is', 'he’d': 'he would', 'he’d’ve': 'he would have', 'he’ll': 'he will', 'he’s': 'he is',
        "how'd": 'how did', "how'll": 'how will',
        "how's": 'how is', 'how,d': 'how did', 'how,ll': 'how will', 'how,s': 'how is', 'how;d': 'how did',
        'how;ll': 'how will',
        'how;s': 'how is', 'how´d': 'how did', 'how´ll': 'how will', 'how´s': 'how is', 'how’d': 'how did',
        'how’ll': 'how will',
        'how’s': 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', 'i,d': 'i would',
        'i,ll': 'i will',
        'i,m': 'i am', 'i,ve': 'i have', 'i;d': 'i would', 'i;ll': 'i will', 'i;m': 'i am', 'i;ve': 'i have',
        "isn't": 'is not',
        'isn,t': 'is not', 'isn;t': 'is not', 'isn´t': 'is not', 'isn’t': 'is not', "it'd": 'it would',
        "it'll": 'it will', "It's": 'it is',
        "it's": 'it is', 'it,d': 'it would', 'it,ll': 'it will', 'it,s': 'it is', 'it;d': 'it would',
        'it;ll': 'it will', 'it;s': 'it is', 'it´d': 'it would', 'it´ll': 'it will', 'it´s': 'it is',
        'it’d': 'it would', 'it’ll': 'it will', 'it’s': 'it is',
        'i´d': 'i would', 'i´ll': 'i will', 'i´m': 'i am', 'i´ve': 'i have', 'i’d': 'i would', 'i’ll': 'i will',
        'i’m': 'i am',
        'i’ve': 'i have', "let's": 'let us', 'let,s': 'let us', 'let;s': 'let us', 'let´s': 'let us',
        'let’s': 'let us', "ma'am": 'madam', 'ma,am': 'madam', 'ma;am': 'madam', "mayn't": 'may not',
        'mayn,t': 'may not', 'mayn;t': 'may not',
        'mayn´t': 'may not', 'mayn’t': 'may not', 'ma´am': 'madam', 'ma’am': 'madam', "might've": 'might have',
        'might,ve': 'might have', 'might;ve': 'might have', "mightn't": 'might not', 'mightn,t': 'might not',
        'mightn;t': 'might not', 'mightn´t': 'might not',
        'mightn’t': 'might not', 'might´ve': 'might have', 'might’ve': 'might have', "must've": 'must have',
        'must,ve': 'must have', 'must;ve': 'must have',
        "mustn't": 'must not', 'mustn,t': 'must not', 'mustn;t': 'must not', 'mustn´t': 'must not',
        'mustn’t': 'must not', 'must´ve': 'must have',
        'must’ve': 'must have', "needn't": 'need not', 'needn,t': 'need not', 'needn;t': 'need not',
        'needn´t': 'need not', 'needn’t': 'need not', "oughtn't": 'ought not', 'oughtn,t': 'ought not',
        'oughtn;t': 'ought not',
        'oughtn´t': 'ought not', 'oughtn’t': 'ought not', "sha'n't": 'shall not', 'sha,n,t': 'shall not',
        'sha;n;t': 'shall not', "shan't": 'shall not',
        'shan,t': 'shall not', 'shan;t': 'shall not', 'shan´t': 'shall not', 'shan’t': 'shall not',
        'sha´n´t': 'shall not', 'sha’n’t': 'shall not',
        "she'd": 'she would', "she'll": 'she will', "she's": 'she is', 'she,d': 'she would', 'she,ll': 'she will',
        'she,s': 'she is', 'she;d': 'she would', 'she;ll': 'she will', 'she;s': 'she is', 'she´d': 'she would',
        'she´ll': 'she will',
        'she´s': 'she is', 'she’d': 'she would', 'she’ll': 'she will', 'she’s': 'she is', "should've": 'should have',
        'should,ve': 'should have', 'should;ve': 'should have',
        "shouldn't": 'should not', 'shouldn,t': 'should not', 'shouldn;t': 'should not', 'shouldn´t': 'should not',
        'shouldn’t': 'should not', 'should´ve': 'should have',
        'should’ve': 'should have', "that'd": 'that would', "that's": 'that is', 'that,d': 'that would',
        'that,s': 'that is', 'that;d': 'that would',
        'that;s': 'that is', 'that´d': 'that would', 'that´s': 'that is', 'that’d': 'that would', 'that’s': 'that is',
        "there'd": 'there had',
        "there's": 'there is', 'there,d': 'there had', 'there,s': 'there is', 'there;d': 'there had',
        'there;s': 'there is',
        'there´d': 'there had', 'there´s': 'there is', 'there’d': 'there had', 'there’s': 'there is',
        "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have',
        'they,d': 'they would', 'they,ll': 'they will', 'they,re': 'they are', 'they,ve': 'they have',
        'they;d': 'they would', 'they;ll': 'they will', 'they;re': 'they are',
        'they;ve': 'they have', 'they´d': 'they would', 'they´ll': 'they will', 'they´re': 'they are',
        'they´ve': 'they have', 'they’d': 'they would', 'they’ll': 'they will',
        'they’re': 'they are', 'they’ve': 'they have', "wasn't": 'was not', 'wasn,t': 'was not', 'wasn;t': 'was not',
        'wasn´t': 'was not',
        'wasn’t': 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have',
        'we,d': 'we would', 'we,ll': 'we will',
        'we,re': 'we are', 'we,ve': 'we have', 'we;d': 'we would', 'we;ll': 'we will', 'we;re': 'we are',
        'we;ve': 'we have',
        "weren't": 'were not', 'weren,t': 'were not', 'weren;t': 'were not', 'weren´t': 'were not',
        'weren’t': 'were not', 'we´d': 'we would', 'we´ll': 'we will',
        'we´re': 'we are', 'we´ve': 'we have', 'we’d': 'we would', 'we’ll': 'we will', 'we’re': 'we are',
        'we’ve': 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is',
        "what've": 'what have', 'what,ll': 'what will', 'what,re': 'what are', 'what,s': 'what is',
        'what,ve': 'what have', 'what;ll': 'what will', 'what;re': 'what are',
        'what;s': 'what is', 'what;ve': 'what have', 'what´ll': 'what will',
        'what´re': 'what are', 'what´s': 'what is', 'what´ve': 'what have', 'what’ll': 'what will',
        'what’re': 'what are', 'what’s': 'what is',
        'what’ve': 'what have', "where'd": 'where did', "where's": 'where is', 'where,d': 'where did',
        'where,s': 'where is', 'where;d': 'where did',
        'where;s': 'where is', 'where´d': 'where did', 'where´s': 'where is', 'where’d': 'where did',
        'where’s': 'where is',
        "who'll": 'who will', "who's": 'who is', 'who,ll': 'who will', 'who,s': 'who is', 'who;ll': 'who will',
        'who;s': 'who is',
        'who´ll': 'who will', 'who´s': 'who is', 'who’ll': 'who will', 'who’s': 'who is', "won't": 'will not',
        'won,t': 'will not', 'won;t': 'will not',
        'won´t': 'will not', 'won’t': 'will not', "wouldn't": 'would not', 'wouldn,t': 'would not',
        'wouldn;t': 'would not', 'wouldn´t': 'would not',
        'wouldn’t': 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', 'you,d': 'you would',
        'you,ll': 'you will',
        'you,re': 'you are', 'you;d': 'you would', 'you;ll': 'you will',
        'you;re': 'you are', 'you´d': 'you would', 'you´ll': 'you will', 'you´re': 'you are', 'you’d': 'you would',
        'you’ll': 'you will', 'you’re': 'you are',
        '´cause': 'because', '’cause': 'because', "you've": "you have", "could'nt": 'could not',
        "havn't": 'have not', "here’s": "here is", 'i""m': 'i am', "i'am": 'i am', "i'l": "i will", "i'v": 'i have',
        "wan't": 'want', "was'nt": "was not", "who'd": "who would",
        "who're": "who are", "who've": "who have", "why'd": "why would", "would've": "would have", "y'all": "you all",
        "y'know": "you know", "you.i": "you i",
        "your'e": "you are", "arn't": "are not", "agains't": "against", "c'mon": "common", "doens't": "does not",
        'don""t': "do not", "dosen't": "does not",
        "dosn't": "does not", "shoudn't": "should not", "that'll": "that will", "there'll": "there will",
        "there're": "there are",
        "this'll": "this all", "u're": "you are", "ya'll": "you all", "you'r": "you are", "you’ve": "you have",
        "d'int": "did not", "did'nt": "did not", "din't": "did not", "dont't": "do not", "gov't": "government",
        "i'ma": "i am", "is'nt": "is not", "‘I": 'I',
        'ᴀɴᴅ': 'and', 'ᴛʜᴇ': 'the', 'ʜᴏᴍᴇ': 'home', 'ᴜᴘ': 'up', 'ʙʏ': 'by', 'ᴀᴛ': 'at', '…and': 'and',
        'civilbeat': 'civil beat', \
        'TrumpCare': 'Trump care', 'Trumpcare': 'Trump care', 'OBAMAcare': 'Obama care', 'ᴄʜᴇᴄᴋ': 'check', 'ғᴏʀ': 'for',
        'ᴛʜɪs': 'this', 'ᴄᴏᴍᴘᴜᴛᴇʀ': 'computer', \
        'ᴍᴏɴᴛʜ': 'month', 'ᴡᴏʀᴋɪɴɢ': 'working', 'ᴊᴏʙ': 'job', 'ғʀᴏᴍ': 'from', 'Sᴛᴀʀᴛ': 'start', 'gubmit': 'submit',
        'CO₂': 'carbon dioxide', 'ғɪʀsᴛ': 'first', \
        'ᴇɴᴅ': 'end', 'ᴄᴀɴ': 'can', 'ʜᴀᴠᴇ': 'have', 'ᴛᴏ': 'to', 'ʟɪɴᴋ': 'link', 'ᴏғ': 'of', 'ʜᴏᴜʀʟʏ': 'hourly',
        'ᴡᴇᴇᴋ': 'week', 'ᴇɴᴅ': 'end', 'ᴇxᴛʀᴀ': 'extra', \
        'Gʀᴇᴀᴛ': 'great', 'sᴛᴜᴅᴇɴᴛs': 'student', 'sᴛᴀʏ': 'stay', 'ᴍᴏᴍs': 'mother', 'ᴏʀ': 'or', 'ᴀɴʏᴏɴᴇ': 'anyone',
        'ɴᴇᴇᴅɪɴɢ': 'needing', 'ᴀɴ': 'an', 'ɪɴᴄᴏᴍᴇ': 'income', \
        'ʀᴇʟɪᴀʙʟᴇ': 'reliable', 'ғɪʀsᴛ': 'first', 'ʏᴏᴜʀ': 'your', 'sɪɢɴɪɴɢ': 'signing', 'ʙᴏᴛᴛᴏᴍ': 'bottom',
        'ғᴏʟʟᴏᴡɪɴɢ': 'following', 'Mᴀᴋᴇ': 'make', \
        'ᴄᴏɴɴᴇᴄᴛɪᴏɴ': 'connection', 'ɪɴᴛᴇʀɴᴇᴛ': 'internet', 'financialpost': 'financial post', 'ʜaᴠᴇ': ' have ',
        'ᴄaɴ': ' can ', 'Maᴋᴇ': ' make ', 'ʀᴇʟɪaʙʟᴇ': ' reliable ', 'ɴᴇᴇᴅ': ' need ',
        'ᴏɴʟʏ': ' only ', 'ᴇxᴛʀa': ' extra ', 'aɴ': ' an ', 'aɴʏᴏɴᴇ': ' anyone ', 'sᴛaʏ': ' stay ', 'Sᴛaʀᴛ': ' start',
        'SHOPO': 'shop',
    }
    mispell_dict = {'SB91': 'senate bill', 'tRump': 'trump', 'utmterm': 'utm term', 'FakeNews': 'fake news',
                    'Gʀᴇat': 'great', 'ʙᴏᴛtoᴍ': 'bottom', 'washingtontimes': 'washington times',
                    'garycrum': 'gary crum', 'htmlutmterm': 'html utm term', 'RangerMC': 'car',
                    'TFWs': 'tuition fee waiver', 'SJWs': 'social justice warrior', 'Koncerned': 'concerned',
                    'Vinis': 'vinys', 'Yᴏᴜ': 'you', 'Trumpsters': 'trump', 'Trumpian': 'trump', 'bigly': 'big league',
                    'Trumpism': 'trump', 'Yoyou': 'you', 'Auwe': 'wonder', 'Drumpf': 'trump', 'utmterm': 'utm term',
                    'Brexit': 'british exit', 'utilitas': 'utilities', 'ᴀ': 'a', '😉': 'wink', '😂': 'joy',
                    '😀': 'stuck out tongue', 'theguardian': 'the guardian', 'deplorables': 'deplorable',
                    'theglobeandmail': 'the globe and mail', 'justiciaries': 'justiciary',
                    'creditdation': 'Accreditation', 'doctrne': 'doctrine', 'fentayal': 'fentanyl',
                    'designation-': 'designation', 'CONartist': 'con-artist', 'Mutilitated': 'Mutilated',
                    'Obumblers': 'bumblers', 'negotiatiations': 'negotiations', 'dood-': 'dood', 'irakis': 'iraki',
                    'cooerate': 'cooperate', 'COx': 'cox', 'racistcomments': 'racist comments',
                    'envirnmetalists': 'environmentalists', }

    def correct_spelling(x, dic):
        for word in dic.keys():
            x = x.replace(word, dic[word])
        return x

    def clean_contractions(text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: x.lower())
    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    data = data.apply(lambda x: clean_contractions(x, contraction_mapping))
    data = data.astype(str).apply(lambda x: correct_spelling(x, mispell_dict))
    return data


def scorer(pred, truth):
    return metrics.roc_auc_score(pred[:,0:1], truth,sample_weight = pred[:,1:2])

def distinctConvert_np(c_list):
    '''
    1. Convert list data to numpy zero padded data, 2 distinct matrices for headlines and bodies
    2. Also outputs sequences lengths as np vector
    '''
    # Compute sequences lengths
    n_sentences = len(c_list)
    c_seqlen = []
    for i in range(n_sentences):
        c_seqlen.append(len(c_list[i]))

    c_max_len = max(c_seqlen)

    # Convert to numpy
    count = 0
    c_np = np.zeros((n_sentences, c_max_len))
    for i in range(n_sentences):
        if (c_seqlen[i] == 0):
            c_seqlen[i] = 1
            count = count + 1
        else:
            c_np[i, :c_seqlen[i]] = c_list[i]

    return c_np, np.array(c_seqlen)

pre_start = time.time()
# read train, test csv
all_data = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')[:200]
train = all_data[:100]
valid = all_data[100:]

gc.collect()
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
# preprocess data
x_train = preprocess(train['comment_text']).values
x_valid = preprocess(valid['comment_text']).values
# Overall
weights = np.ones((len(all_data),)) / 4
# Subgroup
weights += (all_data[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (all_data['target'].values>=0.5).astype(bool).astype(np.int) +
   (all_data[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (all_data['target'].values<0.5).astype(bool).astype(np.int) +
   (all_data[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()
y_train = np.vstack([(train['target'].values).astype(np.float),weights[:100]]).T
y_valid = np.vstack([(valid['target'].values).astype(np.float),weights[100:]]).T
y_aux_train = train[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
y_aux_valid = valid[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = preprocess(test['comment_text'])
y_train_con = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32)
y_valid_con = torch.tensor(np.hstack([y_valid, y_aux_valid]), dtype=torch.float32)
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
del all_data
# tokenize
# tokenizer = text.Tokenizer()
# tokenizer.fit_on_texts(list(x_train) + list(x_valid) + list(x_test))
# bert_config = BertConfig('../input/bert-inference/bert/bert_config.json')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir = None, do_lower_case = False)
x_train, x_valid, x_test = convert_lines(x_train, x_valid, x_test, MAX_LEN, tokenizer)
# x_train = tokenizer.texts_to_sequences(x_train)
# x_valid = tokenizer.texts_to_sequences(x_valid)
# x_test = tokenizer.texts_to_sequences(x_test)

# convert to numpy array
# x_train, x_train_len = distinctConvert_np(x_train)
# x_valid, x_valid_len = distinctConvert_np(x_valid)
# x_test, x_test_len = distinctConvert_np(x_test)

#calculate length
# if np.shape(x_train)[1] > MAX_LEN:
#     x_train = x_train[:, 0:MAX_LEN]
# x_train_len = np.minimum(x_train_len, MAX_LEN)
# if np.shape(x_valid)[1] > MAX_LEN:
#     x_valid = x_valid[:, 0:MAX_LEN]
# x_valid_len = np.minimum(x_valid_len, MAX_LEN)
# if np.shape(x_test)[1] > MAX_LEN:
#     x_test = x_test[:, 0:MAX_LEN]
# x_test_len = np.minimum(x_test_len, MAX_LEN)
#
x_train = torch.from_numpy(x_train).long()
# x_train_len = torch.from_numpy(x_train_len).long()
x_valid = torch.from_numpy(x_valid).long()
# x_valid_len = torch.from_numpy(x_valid_len).long()
x_test = torch.from_numpy(x_test).long()
# x_test_len = torch.from_numpy(x_test_len).long()
train_dataset = data.TensorDataset(x_train, y_train_con)
# output_dim = y_train_con.shape[-1]
# del x_train
# del x_train_len
# del y_train_con
# gc.collect()

print('Preprocess train done.')
valid_dict = {'x_valid': x_valid,
              'y_valid_con': y_valid_con
}
del x_valid
del y_valid_con
gc.collect()
print('Preprocess valid done.')
test_dict = {'x_test': x_test,
}
del x_test
gc.collect()
print('Preprocess test done.')
gc.collect()

#construct embedding matrix
# embedding_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
# print('n unknown words (crawl): ', len(unknown_words_crawl))
# glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
# print('n unknown words (glove): ', len(unknown_words_glove))
# embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
# embedding_matrix = (crawl_matrix + glove_matrix) / 2
# embedding_matrix.shape
# del crawl_matrix
# del glove_matrix
gc.collect()
pre_end = time.time()
print('Preprocess done. Takes {:.2f}s'.format(pre_end - pre_start))

def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    # bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    # return (bce_loss_1 * loss_weight) + bce_loss_2
    return bce_loss_1 * loss_weight

def train_model(model, train_dataset, valid_dict, test_dict,lr, loss_fn):
    # param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    # optimizer = torch.optim.Adam(param_lrs, lr=lr)
    # optimizer = torch.optim.Adam(param_lrs, lr=lr)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.8 ** epoch)
    x_valid = valid_dict['x_valid']
    y_valid_con = valid_dict['y_valid_con']
    x_test = test_dict['x_test']
    # for epoch in range(n_epochs):
    #     start = time.time()
    #     print('Epoch {} starts.'.format(epoch+1))
    #     train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    #     model.train()
    #     scheduler.step()
    #     avg_loss = 0
    #     # run batches
    #     count = 0
    #     for i, data in enumerate(train_loader):
    #         # len = len(data)
    #         x_train , y_train_con = data
    #         out = model(x_train)
    #         loss = loss_fn(out, y_train_con)
    #         # print(i + 1, 'loss: ',loss.item())
    #         avg_loss += loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         count += 1
    #     print('average loss:', avg_loss / count)
    #     gc.collect()
    #     model.eval()
    N = batch_size
    #     valid_preds = np.zeros((len(x_valid), 1))
    #     valid_batches = int(len(x_valid) // N)
    #     batch_start = 0
    #     batch_end = 0
    #     valid_preds = np.zeros((len(x_valid), 1))
    #     for i in range(valid_batches):
    #         batch_start = (i * N)
    #         batch_end = (i + 1) * N
    #         x_batch = x_valid[batch_start:batch_end, :]
    #         out = model(x_batch)
    #         valid_preds[batch_start: batch_end] = sigmoid(out.detach().numpy()[:, 0:1])
    #         # valid_preds[batch_start: batch_end] = out.cpu().detach().numpy()[:, 0:1]
    #     if (batch_end < len(x_valid)):
    #         x_batch = x_valid[batch_end:, :]
    #         out = model(x_batch)
    #         valid_preds[batch_end:] = sigmoid(out.detach().numpy()[:, 0:1])
    #         # valid_preds[batch_end:] = out.cpu().detach().numpy()[:, 0:1]
    #     y = y_valid[:,0] > 0.5
    #     y = [int(val) for val in y]
    #     y_weight = y_valid[:,1]
    #     y_con = np.vstack([y, y_weight]).T
    #     score = scorer(y_con, valid_preds)
    #     end = time.time()
    #     print('Epoch {}/{} \t score={:.4f} \t time={:.2f}s'.format(
    #         epoch + 1, n_epochs, score, end - start))
    model.eval()
    test_batches = int(len(x_test) // N)
    batch_start = 0
    batch_end = 0
    test_preds = np.zeros((len(x_test), 1))
    for i in range(test_batches):
        batch_start = (i * N)
        batch_end = (i + 1) * N
        x_batch = x_test[batch_start:batch_end, :]
        out = model(x_batch)
        test_preds[batch_start: batch_end] = sigmoid(out.detach().numpy()[:,0:1])
        # test_preds[batch_start: batch_end] = out.cpu().detach().numpy()[:, 0:1]
    if (batch_end < len(x_test)):
        x_batch = x_test[batch_end:, :]
        out = model(x_batch)
        test_preds[batch_end:] = sigmoid(out.detach().numpy()[:,0:1])
        # test_preds[batch_end:] = out.cpu().detach().numpy()[:,0:1]
    # state_dict = {
    #     'net': model.state_dict(),
    #     'optimizer': optimizer.state_dict()
    # }
    # model_version = "model" + str(model_idx) + ".pth"
    # torch.save(state_dict, model_version)
    return test_preds


all_test_preds = []
seed_everything(2)
model = NeuralNet(y_aux_train.shape[-1])
model.load_state_dict(torch.load('../input/model-6-19-3/model0.pth')['net'])
model = model.cuda()
test_preds = train_model(model,train_dataset, valid_dict, test_dict, lr=0.000001,
                         loss_fn=custom_loss)
all_test_preds.append(test_preds)
model0 = NeuralNet(y_aux_train.shape[-1])
model0.load_state_dict(torch.load('../input/model-6-17-3/model0.pth')['net'])
model0 = model0.cuda()
test_preds = train_model(model0,train_dataset, valid_dict, test_dict, lr=0.00001,
loss_fn=custom_loss)
all_test_preds.append(test_preds)
model = NeuralNet(y_aux_train.shape[-1])
model.load_state_dict(torch.load('../input/model-6-20-3/model0.pth')['net'])
model = model.cuda()
test_preds = train_model(model,train_dataset, valid_dict, test_dict, lr=0.000001,
loss_fn=custom_loss)
all_test_preds.append(test_preds)
seed_everything(5)
model = NeuralNet2(y_aux_train.shape[-1])
model.load_state_dict(torch.load('../input/model-6-15-2/model0.pth')['net'])
model = model.cuda()
test_preds = train_model(model,train_dataset, valid_dict, test_dict, lr=0.000001,
                         loss_fn=custom_loss)
all_test_preds.append(test_preds)
model = NeuralNet1(y_aux_train.shape[-1])
model.load_state_dict(torch.load('../input/model-6-22-3/model0.pth')['net'])
model = model.cuda()
test_preds = train_model(model,train_dataset, valid_dict, test_dict, lr=0.000001,
loss_fn=custom_loss)
all_test_preds.append(test_preds)
submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': np.mean(all_test_preds, axis=0)[:, 0]
})

submission.to_csv('submission.csv', index=False)