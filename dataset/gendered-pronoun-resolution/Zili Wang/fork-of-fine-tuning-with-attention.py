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
S_DIM = 16
L_TUNING = 24
TUNING = 'mature'
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
    
############# other para
if TUNING == 'mature':
    S1_EPOCH = 30
    S1_EARLY_STOP = 3
    S1_WD = 1e-2
else:
    S1_EPOCH = 10
    S1_EARLY_STOP = 0
    S1_WD = 0

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

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer, BertAdam

print('load data')
#os.system('ls -lah')
torch.manual_seed(SEED)
np.random.seed(SEED) 
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
    _ = torch.zeros((x.shape[0], x.shape[2]))
    if x.is_cuda:
        _ = _.cuda()
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

cache_bert = {}
print('extract bert features..')
bert.eval()
for (x, p, a, b, y, id_) in gpt_iter:
    r = bert.forward(x.cuda(), attention_mask= (x!=0).cuda())
    _ = torch.stack(r[0][-L:]).cpu().data.clone()
    del(r)
    for i, v in enumerate(id_):
        cache_bert[v] = {'a': get_span_tensor(_[:,i,:],a[i]),
                         'b': get_span_tensor(_[:,i,:],b[i]),
                         'p': meanpooling(_[:,i,:], p[i])}
print('cache bert features finished.')  

######## model define
def get_mask(t, shape=(8,123), padding_value=0):
    """input padded batch input B X Seq X Layer X Feats, output mask with shape BXMask """
    if padding_value != 0:
        raise ValueError
    padding_value = torch.zeros(shape)
    if t.is_cuda:
        padding_value = padding_value.cuda()
    mask = []
    for i in t:
        _ = torch.zeros(i.shape[0])
        if t.is_cuda:
            _ = _.cuda()
        for j in range(i.shape[0]):
            if (i[j] == padding_value).sum()==shape[0]*shape[1]:
                break
            _[j] = 1
        mask.append(_)
    return torch.stack(mask)

def masked_softmax(vec, mask, dim=1, epsilon=1e-15):
    exps = torch.exp(vec)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps/masked_sums


class AttentionSimilarityLayer(nn.Module):
    
    def __init__(self, hidden_dim, dropout=0.3):
        super(AttentionSimilarityLayer, self).__init__()
        self.ffnn = nn.Linear(hidden_dim*5, S_DIM)
        nn.init.kaiming_normal_(self.ffnn.weight)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)
        self.repr_dropout = nn.Dropout(0.4)
        self.rescale = 1/np.sqrt(hidden_dim)

    def forward(self, a, a_mask, b, b_mask, p):
        a = self.ln(a)
        b = self.ln(b)
        a_s = (self.repr_dropout(a) @ p[:,:,None]) * self.rescale
        b_s = (self.repr_dropout(b) @ p[:,:,None]) * self.rescale
        a_attn = masked_softmax(a_s.squeeze(2), a_mask, dim=1)
        b_attn = masked_softmax(b_s.squeeze(2), b_mask, dim=1)
        a = (a * a_attn[:,:,None]).sum(1)
        b = (b * b_attn[:,:,None]).sum(1)
        _input = torch.cat([p, a, b, p*a, p*b], dim=1)
        y = self.ffnn(self.dropout(_input))

        return y
    

class MSnet(nn.Module):
    
    def __init__(self, hidden_dim, dropout=0.5, hidden_layer=4):
        super(MSnet, self).__init__()
        self.sim_layers = nn.ModuleList([AttentionSimilarityLayer(hidden_dim, dropout=dropout) for i in range(hidden_layer)])
        self.bn = nn.BatchNorm1d(S_DIM*hidden_layer)
        self.dropout = nn.Dropout(dropout)
        self.mention_score = nn.Linear(S_DIM*hidden_layer+2, 3)
        self.dist_ecoding = nn.Linear(1,1)
        
    def forward(self, a, b, p, ap, bp):
        y = []
        a_mask = get_mask(a, shape=a.shape[2:])
        b_mask = get_mask(b, shape=b.shape[2:])
        for i, l in enumerate(self.sim_layers):
            y.append(l(a[:,:,i,:], a_mask, b[:,:,i,:], b_mask, p[:,i,:]))
        y = torch.cat(y, dim=1) # B X 64*Layer
        y = self.dropout(self.bn(y).relu())
        ap = self.dist_ecoding(ap[:,None]).tanh()
        bp = self.dist_ecoding(bp[:,None]).tanh()
        return self.mention_score(torch.cat([y, ap, bp], dim=1))


######### training
def freeze_collate_func(x):
    _ = [[] for i in range(6)]
    for i in x:
        feats =  cache_bert[i['id']]
        _[0].append(feats['a'])
        _[1].append(feats['b'])
        _[2].append(feats['p'])
        _[3].append(i['coref'])
        _[4].append((i['a_idx'][0] - i['p_idx'][0]).type(torch.FloatTensor))
        _[5].append((i['b_idx'][0] - i['p_idx'][0]).type(torch.FloatTensor))
    return [pad_sequence(v, batch_first=True) if i < 2 else torch.stack(v) for i, v in enumerate(_)] 

def score(pred, y):
    t_float = torch.FloatTensor
    if isinstance(pred, torch.cuda.FloatTensor):
        t_float = torch.cuda.FloatTensor
    y = (torch.cumsum(torch.ones(y.shape[0], 3), dim=1) -1).type(t_float) == y[:,None].type(t_float)
    s = (y.type(t_float) * pred).sum(1).log()
    return -s


def stage1_training(msnet, msnet_opt, lossfunc, train_iter, val_iter, epoch, early_stop=3, start_record=2):
    best_score = 10
    no_improve_epoch = 0
    for e in range(epoch):
        msnet.train()
        epoch_score = np.array([])
        for (a, b, p, y, ap, bp) in iter(train_iter):
            msnet.zero_grad()
            pred = msnet.forward(a.cuda(), b.cuda(), p.cuda(), ap.cuda(), bp.cuda())
            loss = lossfunc(pred, y.squeeze(1).cuda())
            s = score(pred.softmax(1), y.squeeze(1).cuda())
            epoch_score = np.append(epoch_score, s.cpu().data.clone().numpy())
            loss.backward()
            msnet_opt.step()
        with torch.no_grad():
            msnet.eval()
            msnet.zero_grad()
            val_score =  np.array([])
            for (a, b, p, y, ap, bp) in val_iter:
                pred = msnet.forward(a.cuda(), b.cuda(), p.cuda(), ap.cuda(), bp.cuda())
                s = score(pred.softmax(1), y.squeeze(1).cuda())
                val_score = np.append(val_score, s.cpu().data.clone().numpy())
            if  np.mean(val_score) < best_score:
                best_score = np.mean(val_score)
                no_improve_epoch = 0
                if e > start_record:
                    torch.save(msnet.state_dict(), 'tmp.m')
            else:
                no_improve_epoch += 1
                if no_improve_epoch == early_stop:
                    break
    print('stage 1 best score: {:.6f}'.format(best_score))

def step_predict(bert, msnet, x, p, a, b):
    r = bert.forward(x.cuda(), attention_mask= (x!=0).cuda())
    feat = torch.stack(r[0][-L:]).clone()
    del(r)
    ap = (a[:,0] - p[:,0]).type(torch.FloatTensor)
    bp = (b[:,0] - p[:,0]).type(torch.FloatTensor)
    enc_a = []
    enc_b = []
    enc_p = []
    for i in range(a.shape[0]):
        enc_a.append(get_span_tensor(feat[:,i,:],a[i]))
        enc_b.append(get_span_tensor(feat[:,i,:],b[i]))
        enc_p.append(meanpooling(feat[:,i,:], p[i]))
    del(feat)
    enc_a = pad_sequence(enc_a, batch_first=True)
    enc_b = pad_sequence(enc_b, batch_first=True)
    enc_p = torch.stack(enc_p)
    pred = msnet.forward(enc_a, enc_b, enc_p, ap.cuda(), bp.cuda())
    return pred


def stage2_training(epoch, bert, bert_opt, msnet, msnet_opt, lossfunc, train_iter, val_iter, test_iter, early_stop=4):
    best_score = 10
    no_improve_epoch = 0
    for e in range(epoch):
        bert.train()
        msnet.train()
        epoch_score = np.array([])
        for (x, p, a, b, y, id_) in train_iter:
            bert.zero_grad()
            msnet.zero_grad()
            pred = step_predict(bert, msnet, x, p, a, b)
            loss = lossfunc(pred, y.cuda())
            s = score(pred.softmax(1), y.cuda())
            epoch_score = np.append(epoch_score, s.cpu().data.clone().numpy())
            loss.backward()
            msnet_opt.step()
            bert_opt.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            bert.eval()
            msnet.eval()
            bert.zero_grad()
            msnet.zero_grad()
            val_score =  np.array([])
            for (x, p, a, b, y, id_) in val_iter:
                pred = step_predict(bert, msnet, x, p, a, b)
                s = score(pred.softmax(1), y.cuda())
                val_score = np.append(val_score, s.cpu().data.clone().numpy())
            torch.cuda.empty_cache()
        print('epcoh {:02} - train_score {:.4f} - val_score {:.4f} '.format(
                                e, np.mean(epoch_score), np.mean(val_score)))
        if  np.mean(val_score) < best_score:
            best_score = np.mean(val_score)
            no_improve_epoch = 0
            torch.save({'msnet': msnet.state_dict(), 'bert': bert.state_dict()}, 'tmp.m')
        else:
            no_improve_epoch += 1
            if no_improve_epoch == early_stop:
                break
    _ = torch.load('tmp.m')
    bert.load_state_dict(_['bert'])
    msnet.load_state_dict(_['msnet'])
    test_pred = np.array([])
    with torch.no_grad():
        bert.eval()
        msnet.eval()
        for (x, p, a, b, y, id_) in test_iter:
            pred = step_predict(bert, msnet, x, p, a, b)
            test_pred = np.append(test_pred, pred.softmax(1).cpu().data.clone().numpy())
    torch.cuda.empty_cache()
    return best_score, test_pred


#####################

######### train/val/test split

test_df = all_data[:2000]
train_df = all_data[2000:]
test_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test = GPTData(test_df)
train = GPTData(train_df)

test_iter = DataLoader(test, batch_size=10, collate_fn=gpt_collate_func)

def tune_paras(bert_model, tuning_last=12, bert_h=24):
    for name, w in bert_model.named_parameters():
        _ = re.search('encoder\.layer\.(\d+)\.', name)
        if _ and int(_.group(1)) in range(bert_h - tuning_last, bert_h):
            w.requires_grad = True
        else:
            w.requires_grad = False
            
            
torch.manual_seed(SEED)
np.random.seed(SEED) 
tune_paras(bert, L_TUNING)
s1_lr = 4e-4
s2_lr = 5e-6
s2_epoch = 20
s2_wd = 1e-2
s2_early_stop=4

tune_paras(bert, L_TUNING)
msnet = MSnet(BERT_SIZE, dropout=0.6, hidden_layer=L).cuda()
bert_opt = BertAdam(filter(lambda p: p.requires_grad, bert.parameters()), lr=s2_lr, weight_decay=s2_wd)
msnet_opt = optim.Adam(msnet.parameters(), lr=s1_lr, weight_decay=S1_WD)
lossfunc = nn.CrossEntropyLoss()

kfold = KFold(n_splits=5, random_state=SEED, shuffle=True)
s1_batch_size = 32
s2_batch_size = 5
scores = []
bert_s = deepcopy(bert.state_dict().copy())
bert_opt_s = deepcopy(bert_opt.state_dict().copy())
msnet_s = deepcopy(msnet.state_dict().copy())
msnet_opt_s = deepcopy(msnet_opt.state_dict().copy())

########## 5cv control
k_th = 0
test_preds = []

for train_idx, val_idx in kfold.split(list(range(len(train)))):
    _train = [train[i] for i in train_idx]
    _val = [train[i] for i in val_idx]
    msnet.load_state_dict(msnet_s)
    msnet_opt.load_state_dict(msnet_opt_s)
    bert.load_state_dict(bert_s)
    bert_opt.load_state_dict(bert_opt_s)
    print('stage 1')
    train_iter = DataLoader(_train, batch_size=s1_batch_size, shuffle=True, collate_fn=freeze_collate_func)
    val_iter = DataLoader(_val, batch_size=s1_batch_size, shuffle=False, collate_fn=freeze_collate_func)
    stage1_training(msnet, msnet_opt, lossfunc, train_iter, val_iter, epoch=S1_EPOCH, early_stop=S1_EARLY_STOP)
    print('stage 2')
    torch.cuda.empty_cache()
    msnet.load_state_dict(torch.load('tmp.m'))
    train_iter = DataLoader(_train, batch_size=s2_batch_size, shuffle=True, collate_fn=gpt_collate_func)
    val_iter = DataLoader(_val, batch_size=s2_batch_size, shuffle=False, collate_fn=gpt_collate_func)
    msnet_opt = optim.Adam(msnet.parameters(), lr=s2_lr, weight_decay=s2_wd)
    s, y = stage2_training(s2_epoch, bert, bert_opt, msnet, msnet_opt, lossfunc, train_iter, val_iter, test_iter, early_stop=s2_early_stop)
    scores.append(s)
    test_preds.append(y)
    
    k_th += 1
    print('best score: {:.6f}'.format(s))
    print('------------'*3)

    
print('Score: {:.6f} {:.6f}'.format(np.mean(scores), np.std(scores)))
probs = np.mean(test_preds, axis=0).reshape((-1, 3))
true = []
t_ids = []
for (x, p, a, b, y, id_) in test_iter:
    true.append(y)
    t_ids += id_
true = torch.cat(true, dim=0).data.numpy()
print(log_loss(true, probs))
sub = pd.DataFrame()
sub['ID'] = t_ids
sub['A'] = probs[:,0]
sub['B'] = probs[:,1]
sub['NEITHER'] = probs[:,2]
sub.to_csv("submission.csv", index=False)

