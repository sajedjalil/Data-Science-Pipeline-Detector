import os 
import re
import time
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

SEED = 2019

def seed_everything(seed = SEED):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_param():
    param = dict()
    param['train_data_path'] = '../input/train.csv'
    param['test_data_path'] = '../input/test.csv'
    param['glove_path'] = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    param['paragram_path'] = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    param['wiki_path'] = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    param['google_path'] = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    param['sentence_len'] = 120
    param['embed_size'] = 300
    param['max_words'] = None
    return param


def load_data(param):
    train_data = pd.read_csv(param['train_data_path'])
    train_data = train_data.sample(frac = 1, random_state = SEED)
    train_data = train_data.reset_index(drop = True)
    test_data = pd.read_csv(param['test_data_path'])

    X_train = train_data.question_text.str.lower()
    X_test = test_data.question_text.str.lower()
    
    def clean(X):
        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '。']
        for punct in puncts:
            X = X.replace(punct, f' {punct} ')   
        return X
        
    def clean_numbers(x):
        x = re.sub('[0-9]{5}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x
        
    mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

    def _get_mispell(mispell_dict):
        mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
        return mispell_dict, mispell_re
    
    mispellings, mispellings_re = _get_mispell(mispell_dict)
    def replace_typical_misspell(text):
        def replace(match):
            return mispellings[match.group(0)]
        return mispellings_re.sub(replace, text)

    X_train = X_train.apply(lambda x: replace_typical_misspell(x))
    X_test = X_test.apply(lambda x: replace_typical_misspell(x))

    X_train = X_train.apply(lambda x : clean(x))
    X_test = X_test.apply(lambda x : clean(x))

    X_train = X_train.apply(lambda x : clean_numbers(x))
    X_test = X_test.apply(lambda x : clean_numbers(x))

    tokenizer = Tokenizer(num_words = param['max_words'])
    tokenizer.fit_on_texts(pd.concat([X_train, X_test]))

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = train_data.target.values
    test_id = test_data.qid.values
        
    return np.array(X_train), y_train, np.array(X_test), test_id, tokenizer.word_index


def load_embedding(param, vocab, embed_name):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype = 'float32')
    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(param[embed_name], encoding = 'utf-8', errors = 'ignore') if len(o) > 100)
    all_embs = np.stack(embedding_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    np.random.seed(SEED)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (param['vocab_size'], embed_size))
    for word, i in vocab.items():
        if i >= param['vocab_size']: 
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is None:
            embedding_vector = embedding_index.get(word.lower())
        if embedding_vector is None:
            embedding_vector = embedding_index.get(word.upper())
        if embedding_vector is None:
            embedding_vector = embedding_index.get(word.capitalize())
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def f1_smart(y_true, y_pred):
    thresholds = list()
    for threshold in np.arange(0.1, 0.501, 0.01):
        threshold = np.round(threshold, 2)
        res = f1_score(y_true, (y_pred > threshold).astype(int))
        thresholds.append([threshold, res])
    thresholds.sort(key = lambda x: x[1], reverse = True)
    return thresholds[0][1], thresholds[0][0]


class CyclicLR(object):
    def __init__(self, optimizer, base_lr = 1e-3, max_lr = 6e-3,
                 step_size = 2000, mode = 'triangular', gamma = 1.,
                 scale_fn = None, scale_mode = 'cycle', last_batch_iteration = -1):

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

    def batch_step(self, batch_iteration = None):
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

d_model = 300  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 100  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 3

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.lin = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, Q, K, V, attn_mask=None):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        if attn_mask is not None: # attn_mask : [batch_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.lin(context)
        return self.norm(output + residual), attn # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        # self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        # enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, param, embedding_matrix, sinusoid_table):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(param['vocab_size'], d_model)
        self.src_emb.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.src_emb.weight.requires_grad = False
        self.pos_emb = nn.Embedding(param['sentence_len']+1, d_model)
        self.pos_emb.weight = nn.Parameter(sinusoid_table)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, pos_inputs): # enc_inputs : [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self, param, embedding_matrix, sinusoid_table):
        super(Transformer, self).__init__()
        self.encoder = Encoder(param, embedding_matrix, sinusoid_table)
        self.projection = nn.Linear(d_model, 1)
    def forward(self, inputs):
        enc_inputs, pos_inputs = inputs
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, pos_inputs)
        out, _ = torch.max(enc_outputs, 1)
        out = self.projection(out)
        return out


def get_batch(X, index, max_length, batch_size = 512):
    start = index * batch_size
    end = min((index + 1) * batch_size, len(X))
    x_batch = X[start:end]             
    length = [len(x) for x in x_batch]
    max_len = min(round(max(length) * 0.95), max_length)
    x_batch = pad_sequences(x_batch, maxlen = max_len)
    x_batch = torch.tensor(x_batch, dtype = torch.long).cuda()

    position = np.expand_dims(np.arange(1, max_len+1, 1), 0)
    pos_batch = position.repeat(end - start, 0)
    pos_batch = torch.tensor(pos_batch, dtype = torch.long).cuda()
    
    assert(x_batch.shape == pos_batch.shape)

    return [x_batch, pos_batch]


def train_model(model, X, y, val_X, val_y, X_test, max_len, batch_size = 512, n_epochs = 2, validate = True):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.003)
    step_size = len(X) // batch_size // 2
    scheduler = CyclicLR(optimizer, base_lr = 0.001, max_lr = 0.003, step_size = step_size, mode = 'exp_range', gamma = 0.99994)   
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean').cuda()
    
    for epoch in range(n_epochs):
        if epoch == n_epochs - 1:
            model.encoder.src_emb.weight.requires_grad = True
 
        start_time = time.time()
        model.train()
        avg_loss = 0.
        i, end = 0, 0
        while end < len(X):
            x_batch = get_batch(X, i, max_len, batch_size)
            start = i * batch_size
            end = min((i + 1) * batch_size, len(X))                       
            y_batch = torch.tensor(y[start:end], dtype = torch.float).cuda()
            y_pred = model(x_batch)
            scheduler.batch_step()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            i += 1
        avg_loss /= i
        
        np.random.seed(SEED)
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        model.eval()        
        valid_preds = np.zeros((val_X.shape[0]))
        if validate:
            avg_val_loss = 0.
            i, end = 0, 0
            while end < len(val_X):
                x_batch = get_batch(val_X, i, max_len, batch_size)
                start = i * batch_size
                end = min((i + 1) * batch_size, len(val_X))                       
                y_batch = torch.tensor(val_y[start:end], dtype = torch.float).cuda()
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item()
                valid_preds[start:end] = np.squeeze(torch.sigmoid(y_pred).cpu().numpy())
                i += 1
            avg_val_loss /= i
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss = {:.4f} \t val_loss = {:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
            f1, threshold = f1_smart(np.squeeze(val_y), valid_preds)
            print('F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss = {:.4f} \t time = {:.2f}s'.format(epoch + 1, n_epochs, avg_loss, elapsed_time))
    
    if not validate:
        i, end = 0, 0
        while end < len(val_X):
            x_batch = get_batch(val_X, i, max_len, batch_size)
            start = i * batch_size
            end = min((i + 1) * batch_size, len(val_X))                       
            y_batch = torch.tensor(val_y[start:end], dtype = torch.float).cuda()
            y_pred = model(x_batch).detach()
            valid_preds[i * batch_size : (i + 1) * batch_size] = np.squeeze(torch.sigmoid(y_pred).cpu().numpy())
            i += 1
        # f1, threshold = f1_smart(np.squeeze(val_y), valid_preds)
        # print('F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))

    test_preds = np.zeros(X_test.shape[0])   
    i, end = 0, 0
    while end < len(X_test):
        x_batch = get_batch(X_test, i, batch_size)
        y_pred = model(x_batch).detach()
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_test))
        test_preds[start:end] = np.squeeze(torch.sigmoid(y_pred).cpu().numpy())
        i += 1
    
    return valid_preds, test_preds


if __name__ == '__main__':
    param = init_param()
    X_train_ori, y_train_ori, X_test, test_id, word2index = load_data(param)
    param['vocab_size'] = len(word2index) + 1
    embedding_glove = load_embedding(param, word2index, 'glove_path')
    embedding_paragram = load_embedding(param, word2index, 'paragram_path')
    # embedding_wiki = load_embedding(param, word2index, 'wiki_path')
    embedding_matrix = np.mean([embedding_glove, embedding_paragram], axis = 0)
    del embedding_glove, embedding_paragram, word2index

    sinusoid_table = get_sinusoid_encoding_table(param['sentence_len']+1 , d_model)

    local_test = True
    if local_test:
        X_train, X_test, y_train, y_test = train_test_split(X_train_ori, y_train_ori, test_size = 0.15, random_state = SEED)
    else:
        X_train, y_train = X_train_ori, y_train_ori
   
    batch_size = 512
    n_epochs = 50

    fold = 3
    num = 1
    y_test_pred = np.zeros((X_test.shape[0], fold * num))
    y_target = np.zeros((y_train.shape[0], num))
    for n in range(num):
        print('='*80)
        seed = SEED * n
        kfold = list(StratifiedKFold(n_splits = fold, random_state = seed, shuffle = True).split(X_train, y_train))
        for i, (train_index, val_index) in enumerate(kfold):
            X_fold = X_train[train_index]
            y_fold = y_train[train_index, np.newaxis]
            val_X_fold = X_train[val_index]
            val_y_fold = y_train[val_index, np.newaxis]
     
            seed_everything(seed + i)
            model = Transformer(param, embedding_matrix, sinusoid_table)
            model.cuda()
            print(f'Fold {i + 1}') 
            y_pred, test_pred = train_model(model, X_fold, y_fold, val_X_fold, val_y_fold, X_test, param['sentence_len'], batch_size, n_epochs, local_test)
            y_test_pred[:, fold * n + i] = test_pred
            y_target[val_index, n] = y_pred

    y_train_pred = y_target.mean(axis = 1)
    f1, threshold = f1_smart(y_train, y_train_pred)
    print('Final F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
    # print(pd.DataFrame(y_test_pred).corr())
    pred_test_y = (y_test_pred.mean(axis = 1) > threshold).astype(int)
    if local_test:
        f1 = f1_score(y_test, pred_test_y)
        print('local test F1: {:.4f}'.format(f1))
    else:
        sub = pd.DataFrame(columns = ['qid', 'prediction'])
        sub.qid = test_id
        sub.prediction = pred_test_y
        sub.to_csv('submission.csv', index = False)