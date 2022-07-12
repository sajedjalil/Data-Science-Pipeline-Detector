import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import gc
import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from copy import deepcopy
from sklearn.metrics import log_loss
import os

######## hyper-parameters setting ######
BERT = 'large'
SEED = 23
L = 8
S_DIM = 64

########### BERT info #####

BASE_BERT_URL = 'https://storage.googleapis.com/bert_models/2018_10_18/'
if BERT == 'base':
    BERT_NAME = 'uncased_L-12_H-768_A-12'
    BERT_SIZE = 768
    BERT_UNCASE = True
else:
    BERT_NAME = 'cased_L-24_H-1024_A-16'
    BERT_SIZE = 1024
    BERT_UNCASE = False

########### download package and data #####
print('installing apex')
os.system('git clone -q https://github.com/NVIDIA/apex.git')
os.system('pip install -q --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/')
os.system('rm -rf apex')

print('download bert')
os.system('pip install pytorch-pretrained-bert -q')
os.system('wget {}{}.zip -q'.format(BASE_BERT_URL, BERT_NAME))
os.system('unzip -q {}.zip'.format(BERT_NAME))
os.system('rm {}.zip'.format(BERT_NAME))
os.system('wget https://raw.githubusercontent.com/huggingface/pytorch-pretrained-BERT/master/'
          'pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py -q')
os.system('python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path {}/bert_model.ckpt --bert_config_file {}/bert_config.json'
          ' --pytorch_dump_path {}/pytorch_model.bin > /dev/null'.format(BERT_NAME, BERT_NAME, BERT_NAME))
os.system('rm {}/bert_model.ckpt.* convert_tf_checkpoint_to_pytorch.py'.format(BERT_NAME))

print('download data')
os.system('pip install pytorch-pretrained-bert -q')
os.system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q')
os.system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q')
os.system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q')

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer

print('load data')
#os.system('ls -lah')
bert = BertModel.from_pretrained(BERT_NAME).cuda()

gap_dev = pd.read_csv('gap-development.tsv', delimiter='\t')
gap_val = pd.read_csv('gap-validation.tsv', delimiter='\t')
gap_test = pd.read_csv('gap-test.tsv', delimiter='\t')

all_data = pd.concat([gap_dev, gap_val, gap_test])
all_data = all_data.reset_index(drop=True)

def bert_tokenize(text, p, a, b, p_offset, a_offset, b_offset):
    idxs = {}
    tokens = []
    
    a_span = [a_offset, a_offset+len(a), 'a']
    b_span = [b_offset, b_offset+len(b), 'b']
    p_span = [p_offset, p_offset+len(p), 'p']
    
    spans = [a_span, b_span, p_span]
    spans = sorted(spans, key=lambda x: x[0])
    
    last_offset = 0
    idx = -1
    
    def token_part(string):
        _idxs = []
        nonlocal idx
        for w in tokenizer.tokenize(string):
            idx += 1
            tokens.append(w)
            _idxs.append(idx)
        return _idxs
    
    
    for span in spans:
        token_part(text[last_offset:span[0]])
        idxs[span[2]] = token_part(text[span[0]:span[1]])
        last_offset = span[1]
    token_part(text[last_offset:])
    return tokens, idxs
    
tokenizer = BertTokenizer.from_pretrained(BERT_NAME + '/vocab.txt', do_lower_case= BERT_UNCASE)
wp_tokenizer = WordpieceTokenizer(tokenizer.vocab)

print('tokenize...')
_ = all_data.apply(lambda x: bert_tokenize(x['Text'], x['Pronoun'], x['A'], x['B'], x['Pronoun-offset'], x['A-offset'], x['B-offset']), axis=1)
all_data['encode'] = [tokenizer.convert_tokens_to_ids(i[0]) for i in _]
all_data['p_idx'] = [i[1]['p'] for i in _]
all_data['a_idx'] = [i[1]['a'] for i in _]
all_data['b_idx'] = [i[1]['b'] for i in _]

print('clean..')
all_data.at[2602, 'encode'] = all_data.loc[2602, 'encode'][:280]
all_data.at[3674, 'encode'] = all_data.loc[3674, 'encode'][:280]  # too long, target in head
all_data.at[209, 'encode'] = all_data.loc[209, 'encode'][60:]
all_data.at[209, 'a_idx'] = [_ - 60 for _ in all_data.loc[209, 'a_idx']]  # too log, traget in tail
all_data.at[209, 'b_idx'] = [_ - 60 for _ in all_data.loc[209, 'b_idx']]
all_data.at[209, 'p_idx'] = [_ - 60 for _ in all_data.loc[209, 'p_idx']]

class GPTData(Dataset):
    
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        _ = self.data.loc[idx]
        sample = {'id': _['ID'],
                  'encode': torch.LongTensor([101] + _['encode'] + [102]),
                  'p_idx': torch.LongTensor(_['p_idx'])+1,
                  'a_idx': torch.LongTensor(_['a_idx'])+1,
                  'b_idx': torch.LongTensor(_['b_idx'])+1,
                  'coref': torch.LongTensor([0 if _['A-coref'] else 1 if _['B-coref'] else 2])
                 }
        return sample
        
class SortLenSampler(Sampler):
    
    def __init__(self, data_source, key):
        self.sorted_idx = sorted(range(len(data_source)), key=lambda x: len(data_source[x][key]))
    
    def __iter__(self):
        return iter(self.sorted_idx)
    
    def __len__(self):
        return len(self.sorted_idx)
        

def gpt_collate_func(x):
    _ = [[], [], [], [], [], []]
    for i in x:
        _[0].append(i['encode'])
        _[1].append(i['p_idx'])
        _[2].append(i['a_idx'])
        _[3].append(i['b_idx'])
        _[4].append(i['coref'])
        _[5].append(i['id'])
    return torch.nn.utils.rnn.pad_sequence(_[0], batch_first=True, padding_value=0), \
           torch.nn.utils.rnn.pad_sequence(_[1], batch_first=True, padding_value=-1), \
           torch.nn.utils.rnn.pad_sequence(_[2], batch_first=True, padding_value=-1), \
           torch.nn.utils.rnn.pad_sequence(_[3], batch_first=True, padding_value=-1), \
           torch.cat(_[4], dim=0), _[5]

def meanpooling(x, idx, pad=-1):
    """x: Layer X Seq X Feat, idx: Seq """
    t_type = torch.cuda.FloatTensor if isinstance(x, torch.cuda.FloatTensor) else torch.FloatTensor
    
    _ = torch.zeros((x.shape[0], x.shape[2]))
    cnt = 0
    for i in idx:
        if i == pad:
            break
        for j in range(x.shape[0]):
            _[j] += x[j,i,:]
        cnt += 1
    if cnt == 0:
        raise ValueError('0 dive')
    return _/cnt

def get_span_tensor(bert_t, index, last_layer=L, pad_id=-1):
    """return Seq X Layer X Feat"""
    span_tensor = []
    for i in index:
        if i == pad_id:
            break
        span_tensor.append(bert_t[-last_layer:, i, :])
    return torch.stack(span_tensor)
    
_ = GPTData(all_data)
gpt_iter = DataLoader(_, batch_size=5, sampler=SortLenSampler(_, 'encode'), collate_fn=gpt_collate_func)

bert_feats = []
print('extract bert features..')
bert.eval()
for (x, p, a, b, y, id_) in gpt_iter:
    r = bert.forward(x.cuda(), attention_mask= (x!=0).cuda())
    _ = torch.stack(r[0][-L:]).cpu().data.clone()
    del(r)
    for i, v in enumerate(id_):
        bert_feats.append({'a': meanpooling(_[:,i,:],a[i]),
                           'b': meanpooling(_[:,i,:],b[i]),
                           'p': meanpooling(_[:,i,:], p[i]),
                           'ap': (a[i][0] - p[i][0]).type(torch.FloatTensor),
                           'bp': (b[i][0] - p[i][0]).type(torch.FloatTensor),
                           'y': y[i],
                           'id': v})

print('extract bert features finished.')       

torch.manual_seed(SEED)
np.random.seed(SEED)


############

class BERTfeature(Dataset):
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def bert_collate_func(x):
    _ = [[] for i in range(6)]
    for i in x:
        _[0].append(i['a'])
        _[1].append(i['b'])
        _[2].append(i['p'])
        _[3].append(i['y'])
        _[4].append(i['ap'])
        _[5].append(i['bp'])
    return [torch.stack(i) for i in _]

############

test = [i for i in bert_feats if 'dev' in i['id']]
train = [i for i in bert_feats if 'dev' not in i['id']]

######## model define
class SimilarityLayer(nn.Module):
    
    def __init__(self, hidden_dim, dropout=0.3):
        super(SimilarityLayer, self).__init__()
        self.ffnn = nn.Linear(hidden_dim*5, S_DIM)
        nn.init.kaiming_normal_(self.ffnn.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, a, b, p):
        _input = torch.cat([p, a, b, p*a, p*b], dim=1)
        y = self.ffnn(self.dropout(_input))
        
        return y
    

class MSnet(nn.Module):
    
    def __init__(self, hidden_dim, dropout=0.5, hidden_layer=4):
        super(MSnet, self).__init__()
        self.sim_layers = nn.ModuleList([SimilarityLayer(hidden_dim, dropout=dropout) for i in range(hidden_layer)])
        self.bn = nn.BatchNorm1d(S_DIM*hidden_layer)
        self.dropout = nn.Dropout(dropout)
        self.mention_score = nn.Linear(S_DIM*hidden_layer+2, 3)
        self.dist_ecoding = nn.Linear(1,1)
        
    def forward(self, a, b, p, ap, bp):
        y = []
        for i, l in enumerate(self.sim_layers):
            y.append(l(a[:,i,:], b[:,i,:], p[:,i,:]))
        y = torch.cat(y, dim=1) # B X 64*Layer
        y = self.dropout(self.bn(y).relu())
        ap = self.dist_ecoding(ap[:,None]).tanh()
        bp = self.dist_ecoding(bp[:,None]).tanh()
        return self.mention_score(torch.cat([y, ap, bp], dim=1))


def training_cuda(epoch, model, lossfunc, optimizer, train_iter, val_iter, test_iter, start=5):
    best_score = 10
    for i in range(epoch):
        model.train()
        epoch_score = np.array([])
        for (a, b, p, y, ap, bp) in iter(train_iter):
            model.zero_grad()
            pred = model.forward(a.cuda(), b.cuda(), p.cuda(), ap.cuda(), bp.cuda())
            # loss = lossfunc(pred, y.cuda()) + l2 * torch.stack([torch.norm(i[1]) for i in model.named_parameters() if 'weight' in i[0]]).sum()
            loss = lossfunc(pred, y.cuda())
            s = score(pred.softmax(1), y.cuda())
            epoch_score = np.append(epoch_score, s.cpu().data.numpy())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            model.zero_grad()
            val_score =  np.array([])
            for (va, vb, vp, vy, vap, vbp) in val_iter:
                vpred = model.forward(va.cuda(), vb.cuda(), vp.cuda(), vap.cuda(), vbp.cuda())
                vs = score(vpred.softmax(1), vy.cuda())
                val_score = np.append(val_score, vs.cpu().data.numpy())
            print('epcoh {:02} - train_score {:.4f} - val_score {:.4f} '.format(
                                i, np.mean(epoch_score), np.mean(val_score)))
            if  np.mean(val_score) < best_score:
                best_score = np.mean(val_score)
                if i > start:
                    torch.save(model.state_dict(), 'tmp.m')
    model.load_state_dict(torch.load('tmp.m'))
    test_pred = np.array([])
    for (ta, tb, tp, ty, tap, tbp) in test_iter:
        vpred = model.forward(ta.cuda(), tb.cuda(), tp.cuda(), tap.cuda(), tbp.cuda())
        test_pred = np.append(test_pred, vpred.softmax(1).cpu().data.numpy())
    return best_score, test_pred


def score(pred, y):
    t_float = torch.FloatTensor
    if isinstance(pred, torch.cuda.FloatTensor):
        t_float = torch.cuda.FloatTensor
    y = (torch.cumsum(torch.ones(y.shape[0], 3), dim=1) -1).type(t_float) == y[:,None].type(t_float)
    s = (y.type(t_float) * pred).sum(1).log()
    return -s

#####################

print('training')
m = MSnet(BERT_SIZE, dropout=0.6, hidden_layer=L).cuda()
optimizer = optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-5)
loss_fuc = nn.CrossEntropyLoss()
batch_size = 32

kfold = KFold(n_splits=5, random_state=SEED, shuffle=True)
scores = []
m_s = deepcopy(m.state_dict().copy())
opt_s = deepcopy(optimizer.state_dict().copy())

k_th = 0
test_iter = DataLoader(BERTfeature(test), batch_size=batch_size, shuffle=False, collate_fn=bert_collate_func)
test_preds = []

for train_idx, val_idx in kfold.split(list(range(len(train)))):
    
    _train = [v for i, v in enumerate(train) if i in train_idx]
    _val = [v for i, v in enumerate(train) if i in val_idx]
    train_iter = DataLoader(BERTfeature(_train), batch_size=batch_size, shuffle=True, collate_fn=bert_collate_func)
    val_iter = DataLoader(BERTfeature(_val), batch_size=batch_size, shuffle=False, collate_fn=bert_collate_func)
    
    m.load_state_dict(m_s)
    optimizer.load_state_dict(opt_s)
    s, y = training_cuda(30, m, loss_fuc, optimizer, train_iter, val_iter, test_iter)
    scores.append(s)
    test_preds.append(y)
    
    k_th += 1
    print('------------'*3)
    
print('Score: {:.4f} {:.4f}'.format(np.mean(scores), np.std(scores)))
probs = np.mean(test_preds, axis=0).reshape((-1, 3))
true = torch.cat([ty for (ta, tb, tp, ty, tap, tbp) in test_iter], dim=0).data.numpy()
t_ids = [i['id'] for i in test]
print(log_loss(true, probs))
sub = pd.DataFrame()
sub['ID'] = t_ids
sub['A'] = probs[:,0]
sub['B'] = probs[:,1]
sub['NEITHER'] = probs[:,2]
sub.to_csv("submission.csv", index=False)