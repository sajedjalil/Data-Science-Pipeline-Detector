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
from torch.utils.data import TensorDataset, DataLoader

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
    param['sentence_len'] = 72
    param['embed_size'] = 300
    param['max_words'] = 200000
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

    X_train = pad_sequences(X_train, maxlen = param['sentence_len'])
    X_test = pad_sequences(X_test, maxlen = param['sentence_len'])

    y_train = train_data.target.values
    test_id = test_data.qid.values
        
    return X_train, y_train, X_test, test_id, tokenizer.word_index


def load_embedding(param, vocab, embed_name):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype = 'float32')
    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(param[embed_name], encoding = 'utf-8', errors = 'ignore') if len(o) > 100)
    all_embs = np.stack(embedding_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    np.random.seed(SEED)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (param['vocab_size'], embed_size))
    for word, i in vocab.items():
        if i >= param['vocab_size']: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
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


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias = True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask = None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
        
        eij = torch.tanh(eij)   
        a = torch.softmax(eij, 1)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Capsule(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule = 8, dim_capsule = 8, 
                 routings = 3, share_weights = True,
                 activation = 'default', **kwargs):
        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))

    def forward(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = torch.softmax(b, dim = 2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik, bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis = -1, T_epsilon = 1e-7):
        s_squared_norm = (x ** 2).sum(axis, keepdim = True)
        scale = torch.sqrt(s_squared_norm + T_epsilon)
        return x / scale


class NeuralNet(nn.Module):
    def __init__(self, param, embedding_matrix):
        super(NeuralNet, self).__init__()
        
        hidden_size = 60
        
        self.embedding = nn.Embedding(param['vocab_size'], param['embed_size'])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(param['embed_size'], hidden_size, bidirectional = True, batch_first = True)
        self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional = True, batch_first = True)
        
        self.lstm_attention = Attention(hidden_size*2, param['sentence_len'])
        self.gru_attention = Attention(hidden_size*2, param['sentence_len'])

        self.linear = nn.Linear(hidden_size*8, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        
        h_lstm, _ = self.lstm(h_embedding)
                
        h_lstm_atten = self.lstm_attention(h_lstm)

        h_gru, _ = self.gru(h_lstm)
        
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        return out


def train_model(model, X, y, val_X, val_y, test_loader, batch_size = 512, n_epochs = 2, validate = True):
    train = TensorDataset(X, y)
    valid = TensorDataset(val_X, val_y)       
    train_loader = DataLoader(train, batch_size, shuffle = True)
    valid_loader = DataLoader(valid, batch_size, shuffle = False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.003)
    scheduler = CyclicLR(optimizer, base_lr = 0.001, max_lr = 0.003, step_size = 300, mode = 'exp_range', gamma = 0.99994)   
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean').cuda()
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.      
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            scheduler.batch_step()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()        
        valid_preds = np.zeros((val_X.size(0)))
        if validate:
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size : (i + 1) * batch_size] = np.squeeze(torch.sigmoid(y_pred).cpu().numpy())
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss = {:.4f} \t val_loss = {:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
            f1, threshold = f1_smart(np.squeeze(val_y.cpu().numpy()), valid_preds)
            print('F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss = {:.4f} \t time = {:.2f}s'.format(epoch + 1, n_epochs, avg_loss, elapsed_time))
    
    if not validate:
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds[i * batch_size : (i + 1) * batch_size] = np.squeeze(torch.sigmoid(y_pred).cpu().numpy())
        # f1, threshold = f1_smart(np.squeeze(val_y.cpu().numpy()), valid_preds)
        # print('F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))

    test_preds = np.zeros((len(test_loader.dataset)))   
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()
        test_preds[i * batch_size : (i + 1) * batch_size] = np.squeeze(torch.sigmoid(y_pred).cpu().numpy())
    
    return valid_preds, test_preds


def result_analyze(X_test, word2index, y_test, pred_test_y):
    def inverse(index2word, text):
        return ' '.join([index2word[token] for token in text if not token == 0])
        
    index2word = dict([[v,k] for k,v in word2index.items()])
    result = pd.DataFrame(columns = ['text', 'pred', 'real'])
    for i, (x, y, pred_y) in enumerate(zip(X_test, y_test, pred_test_y)):
        if not y == pred_y:
            text = inverse(index2word, x)
            result = result.append(pd.DataFrame({'text': text, 'pred': pred_y, 'real': y}, index = [i]))
    result.to_csv('result.csv', index = False)


if __name__ == '__main__':
    param = init_param()
    X_train_ori, y_train_ori, X_test, test_id, word2index = load_data(param)
    param['vocab_size'] = min(param['max_words'], len(word2index) + 1)
    embedding_glove = load_embedding(param, word2index, 'glove_path')
    embedding_paragram = load_embedding(param, word2index, 'paragram_path')
    # embedding_wiki = load_embedding(param, word2index, 'wiki_path')
    embedding_matrix = np.mean([embedding_glove, embedding_paragram], axis = 0)
    del embedding_glove, embedding_paragram

    local_test = False
    if local_test:
        X_train, X_test, y_train, y_test = train_test_split(X_train_ori, y_train_ori, test_size = 0.15, random_state = SEED)
    else:
        X_train, y_train = X_train_ori, y_train_ori
   
    batch_size = 512
    n_epochs = 5

    X_test = torch.tensor(X_test, dtype = torch.long).cuda()
    test = TensorDataset(X_test)
    test_loader = DataLoader(test, batch_size, shuffle = False)

    fold = 3
    num = 3
    y_test_pred = np.zeros((X_test.shape[0], fold * num))
    y_target = np.zeros((y_train.shape[0], num))
    for n in range(num):
        print('='*80)
        seed = SEED * n
        kfold = list(StratifiedKFold(n_splits = fold, random_state = seed, shuffle = True).split(X_train, y_train))
        for i, (train_index, val_index) in enumerate(kfold):
            X_fold = torch.tensor(X_train[train_index], dtype = torch.long).cuda()
            y_fold = torch.tensor(y_train[train_index, np.newaxis], dtype = torch.float32).cuda()
            val_X_fold = torch.tensor(X_train[val_index], dtype = torch.long).cuda()
            val_y_fold = torch.tensor(y_train[val_index, np.newaxis], dtype = torch.float32).cuda()
     
            seed_everything(seed + i)
            model = NeuralNet(param, embedding_matrix)
            model.cuda()
            print(f'Fold {i + 1}') 
            y_pred, test_pred = train_model(model, X_fold, y_fold, val_X_fold, val_y_fold, test_loader, batch_size, n_epochs, False)
            y_test_pred[:, fold * n + i] = test_pred
            y_target[val_index, n] = y_pred

    y_train_pred = y_target.mean(axis = 1)
    f1, threshold = f1_smart(y_train, y_train_pred)
    print('Final F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
    print(pd.DataFrame(y_test_pred).corr())
    pred_test_y = (y_test_pred.mean(axis = 1) > threshold).astype(int)
    if local_test:
        f1 = f1_score(y_test, pred_test_y)
        print('local test F1: {:.4f}'.format(f1))
        result_analyze(X_test.cpu().numpy(), word2index, y_test, pred_test_y)
    else:
        sub = pd.DataFrame(columns = ['qid', 'prediction'])
        sub.qid = test_id
        sub.prediction = pred_test_y
        sub.to_csv('submission.csv', index = False)