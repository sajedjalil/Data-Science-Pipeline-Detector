'''
Referenced
https://www.kaggle.com/nz0722/lstm-fast-ai-simple-tuned
https://github.com/atselousov/transformer_chatbot
'''

import os
import math
from time import time
import numpy as np
import pandas as pd
import gc
import random
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import re

TRAIN_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'
MAX_LEN = 200
BATCH_SIZE = 100
PUNCT = "/#$%()*+/:<=>@[\\]^_`{|}~`" + '""“”' + '∞θ÷α•à−β∅³π₹´°£€\×™√²—–&'

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(439)


class Preprocessor:
  
    def __init__(self, train, test, embedding):
        self.train = train
        self.test = test
        self.embedding = embedding
      
    @staticmethod
    def text_process(data):
        def clean_special_chars(text, punct):
            for p in punct:
                text = text.replace(p, ' ')
            text = text.replace(';', '.')
            text = text.replace('‘', "'")
            text = text.replace('՚', "'")
            text = text.replace('’', "'")

            text = re.sub(r'\.{2,}', '…', text)
            text = re.sub(r' \'', ' ', text)
            text = re.sub(r'\' ', ' ', text)
            text = re.sub(r'!+', '!', text)
            text = re.sub(r'\?+', '?', text)

            text = text.replace('.', ' . ')
            text = text.replace('…', ' … ')
            text = text.replace("'", " ' ")
            text = text.replace('-', ' - ')
            text = text.replace(',', ' , ')
            text = text.replace('!', ' ! ')
            text = text.replace('?', ' ? ')
            text = " ".join(text.split()).lower()
            return text

        data = data.astype(str).apply(lambda x: clean_special_chars(x, PUNCT))
        return data
      
    @staticmethod
    def build_matrix(word_index, path):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        def load_embeddings(path):
            with open(path) as f:
                return dict(get_coefs(*line.strip().split(' ')) for line in f)
              
        embedding_index = load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        unknown_words = []

        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
        return embedding_matrix, unknown_words

    def preprocess(self, train_samples=None):
        if train_samples is not None:
            train = pd.read_csv(self.train, nrows=train_samples)
        else:
            train = pd.read_csv(self.train)
        test = pd.read_csv(self.test)
        self.test_id = test['id']
        print('data loaded')
        
        permutation = np.random.permutation(len(train))
        train = train.iloc[permutation]
        
        x_train = Preprocessor.text_process(train['comment_text'])
        y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
        x_test = Preprocessor.text_process(test['comment_text'])
        
        identity_columns = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        # Overall
        weights = np.ones((len(x_train),)) / 4
        # Subgroup
        weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
        # Background Positive, Subgroup Negative
        weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
           (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
        # Background Negative, Subgroup Positive
        weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
           (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
        self.loss_weight = 1.0 / weights.mean()
        
        y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T
        
        print('data text-processed')
        
        tokenizer = text.Tokenizer(filters=PUNCT)
        tokenizer.fit_on_texts(list(x_train) + list(x_test))
        self.n_vocabs = len(tokenizer.word_index) + 1
        self.tokenizer = tokenizer

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')
        self.x_train, self.y_train, self.y_aux_train, self.x_test = x_train, y_train, y_aux_train, x_test
        self.num_aux_targets = y_aux_train.shape[-1]
        print('data tokenized')
        
        self.embedding_matrix, self.unknown_words = Preprocessor.build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
        print('embedding built')
        
        
    def wrap(self, batch_size):
        x_train_torch = torch.tensor(self.x_train, dtype=torch.long)
        #y_train_torch = torch.tensor(self.y_train, dtype=torch.float32)
        y_train_torch = torch.tensor(np.hstack([self.y_train, self.y_aux_train]), dtype=torch.float32)
        x_test_torch = torch.tensor(self.x_test, dtype=torch.long)

        #valid_cut = int(len(x_train_torch)*0.9)
        valid_cut = len(x_train_torch)

        train_dataset = torch.utils.data.TensorDataset(x_train_torch[:valid_cut], y_train_torch[:valid_cut])
        #valid_dataset = torch.utils.data.TensorDataset(x_train_torch[valid_cut:], y_train_torch[valid_cut:])
        test_dataset = torch.utils.data.TensorDataset(x_test_torch)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.test_size = len(test_dataset)
        print('data wrapped')


class PosEncoding(nn.Module):
    def __init__(self, embeddings_size):
        super(PosEncoding, self).__init__()
        def pos_angle(pos, i):
            return pos / np.power(10000, 2*(i//2) / embeddings_size)

        def pos_vector(pos):
            return [pos_angle(pos, i) for i in range(embeddings_size)]

        sinusoid_table = np.array([pos_vector(pos) for pos in range(MAX_LEN)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        self.sinusoid_table = torch.cuda.FloatTensor(sinusoid_table)
        
    def forward(self, x_shape, padding_mask):
        encoding = torch.zeros(x_shape).cuda()
        for i in range(x_shape[0]):
            non_pad_idxs = (~padding_mask[i]).nonzero()
            encoding[i, non_pad_idxs, :] = self.sinusoid_table[non_pad_idxs, :]
        return encoding
      
      
class MultiheadAttention(nn.Module):
    def __init__(self, n_features, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, padding_mask):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)
        w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        w.masked_fill_(padding_mask.all(dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3), 0)
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, x, padding_mask):
        query, key, value = self.qkv_proj(x).split(self.n_features, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x
      
      
class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, d_in, d_out, dropout):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(d_in, d_out*2)
        self.layer_2 = nn.Linear(d_out*2, d_out)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        return x
      

class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_out, n_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()
        assert(d_in % d_out == 0)
        reduction = d_in // d_out

        self.attn = MultiheadAttention(d_in, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(d_in)
        self.ff = FeedForward(d_in, d_out, ff_dropout)
        self.ff_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.avg_pool = nn.AvgPool1d(reduction, reduction)

    def forward(self, x, padding_mask):
        a = self.attn(x, padding_mask)
        a = self.dropout(a)
        x = self.attn_norm(x + a)

        f = self.ff(x)
        f = self.dropout(f)
        x = self.avg_pool(x)
        x = self.ff_norm(x + f)

        return x
      
      
class TransformerModel(nn.Module):
    def __init__(self, n_vocabs, embedding_matrix, dropout, embed_dropout, attn_dropout, ff_dropout, num_aux_targets):
        super(TransformerModel, self).__init__()

        embeddings_size = embedding_matrix.shape[1]
        
        self.embeddings = nn.Embedding(n_vocabs, embeddings_size)
        self.embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embeddings.weight.requires_grad = False
        
        self.pos_encoding = PosEncoding(embeddings_size)
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        #layers_1 = [TransformerBlock(embeddings_size, embeddings_size, 6, dropout, attn_dropout, ff_dropout) for i in range(2)]
        #rlayer_1 = [TransformerBlock(embeddings_size, embeddings_size//2, 6, dropout, attn_dropout, ff_dropout)]
        #layers_2 = [TransformerBlock(embeddings_size//2, embeddings_size//2, 6, dropout, attn_dropout, ff_dropout) for i in range(1)]
        #rlayer_2 = [TransformerBlock(embeddings_size//2, embeddings_size//10, 6, dropout, attn_dropout, ff_dropout)]
        #layers_3 = [TransformerBlock(embeddings_size//10, embeddings_size//10, 6, dropout, attn_dropout, ff_dropout) for i in range(2)]
        #self.layers = nn.ModuleList(layers_1 + rlayer_1 + layers_2 + rlayer_2 + layers_3)
        #self.layers = nn.ModuleList(layers_1 + rlayer_1 + layers_2)
        self.layers = nn.ModuleList([TransformerBlock(embeddings_size, embeddings_size, 6, dropout, attn_dropout, ff_dropout) for i in range(5)])
        self.linear_1 = nn.Linear(embeddings_size, 1)
        self.linear_2 = nn.Linear(MAX_LEN, 50)
        #self.post_layers = nn.Linear(MAX_LEN * embeddings_size//2, 100)
        #self.post_layers = nn.Linear(MAX_LEN * (embeddings_size//10), 50)
        self.linear_out = nn.Linear(50, 1)
        self.linear_aux_out = nn.Linear(50, num_aux_targets)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, aux=True):
        padding_mask = x.eq(0)

        x = self.embeddings(x)
        x = x + self.pos_encoding(x.shape, padding_mask)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, padding_mask)
            padding_mask = x.eq(0).all(dim=-1)
        
        #x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        x = self.linear_1(x).squeeze()
        x = FeedForward.gelu(self.linear_2(x))
        result = self.linear_out(x)
        out = result
        if aux:
            aux_result = self.linear_aux_out(x)
            out = torch.cat([result, aux_result], 1)
        
        return out
      

def custom_loss(data, targets, loss_weight):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2
    
    
class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.
    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class NoamOpt:
    def __init__(self, embeddings_size, factor, warmup, optimizer):
        self.embeddings_size = embeddings_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer

        self._step = 0
        
    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
            
        return self.factor * (self.embeddings_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


preprocessor = Preprocessor(TRAIN_PATH, TEST_PATH, GLOVE_EMBEDDING_PATH)
preprocessor.preprocess()
preprocessor.wrap(BATCH_SIZE)
print()

model = TransformerModel(preprocessor.n_vocabs, preprocessor.embedding_matrix, 0.1, 0.1, 0.1, 0.1, preprocessor.num_aux_targets).cuda()
base_optimizer = Adam(params=model.parameters(), lr=0.0007, weight_decay=0.02)
optimizer = NoamOpt(preprocessor.embedding_matrix.shape[1], 0.0007, 10000, base_optimizer)
for epoch in range(7):
    print('< epoch {} >'.format(epoch+1))
    start = time()
    train_loss = 0.0
    for i, (x, y) in enumerate(preprocessor.train_loader):
        logits = model(x.cuda()).squeeze()
        loss = custom_loss(logits, y.cuda(), preprocessor.loss_weight)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        if i % 1000 == 999:
            end = time()
            print('step {}, loss {:.3f} - took {} sec'.format(i+1, train_loss/1000, int(end-start)))
            start = end
            train_loss = 0.0
    print()
print('train finished!')
print()

sigmoid = nn.Sigmoid()
predictions = np.zeros(preprocessor.test_size)

for i, (inputs,) in enumerate(preprocessor.test_loader):
    outputs = model(inputs.cuda(), aux=False).cpu()
    y_preds = sigmoid(outputs).data[:, 0]
    end = min((i+1)*BATCH_SIZE, preprocessor.test_size)
    predictions[i*BATCH_SIZE:end] = y_preds
    
submission = pd.DataFrame.from_dict({
    'id': preprocessor.test_id,
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False)
print('prediction finished!')