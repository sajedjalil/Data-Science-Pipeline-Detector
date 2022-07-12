# coding: utf-8

# In[1]:

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = '../input/avito-demand-prediction/'
textdata_path = '../input/adp-prepare-kfold-text/textdata.csv'
#EMB_PATH = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
#EMB_PATH = '../input/fasttext-russian-2m/wiki.ru.vec'
target_col = 'deal_probability'

SUBMIT = True


# In[2]:

def reduce_memory(df):
    for c in df.columns:
        if df[c].dtype=='int':
            if df[c].min()<0:
                if df[c].abs().max()<2**7:
                    df[c] = df[c].astype('int8')
                elif df[c].abs().max()<2**15:
                    df[c] = df[c].astype('int16')
                elif df[c].abs().max()<2**31:
                    df[c] = df[c].astype('int32')
                else:
                    continue
            else:
                if df[c].max()<2**8:
                    df[c] = df[c].astype('uint8')
                elif df[c].max()<2**16:
                    df[c] = df[c].astype('uint16')
                elif df[c].max()<2**32:
                    df[c] = df[c].astype('uint32')
                else:
                    continue
    return df

usecols = [
           'user_id_common', 
           'region', 
           'city', 
           'parent_category_name', 
           'category_name', 
           'param_1', 
           'param_2', 
           'param_3', 
           'price', 
           'item_seq_number', 
           'user_type', 
           'image', 
           'image_top_1', 
           'dow'
           ]

dl_feat_idx = pd.read_csv('../input/adp-prepare-kfold-text/textdata.csv', 
                          usecols=['eval_set', 'label'])
train_num = (dl_feat_idx['eval_set']!=10).sum()
eval_sets = dl_feat_idx['eval_set'].values
labels = dl_feat_idx['label'].values
del dl_feat_idx; gc.collect()

df = pd.read_csv('../input/adp-prepare-data-labelencoder/data_lbe.csv', usecols=usecols)
print(df.info())
df = reduce_memory(df)
print(df.info())

# In[3]:

cat_cols = [
           'user_id_common', 
           'region', 
           'city', 
           'parent_category_name', 
           'category_name', 
           'param_1', 
           'param_2', 
           'param_3', 
           'item_seq_number', 
           'user_type', 
           'image', 
           'image_top_1', 
           'dow'
           ]
df = df[cat_cols]

for c in cat_cols:
    df[c] = df[c] - df[c].min() + 1
    print(c, df[c].min(), df[c].max())
feature_sizes = [int(df[c].max())+1 for c in cat_cols]


# In[9]:


from time import time
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class DataBuilder(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.size = len(X)
    def __getitem__(self, index):
        X_i = torch.LongTensor(self.X[index].tolist())
        if self.y is not None:
            y_i = torch.FloatTensor([float(self.y[index])])
        else:
            y_i = torch.FloatTensor([-1])
        return X_i, y_i
    def __len__(self):
        return self.size
    
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def init_args(self):
        raise NotImplementedError('subclass must implement this')
    
    def check_args(self):
        default_values = {
            'optimizer_type': 'adam',
            'save_sub_dir': '.', 
            #'valid_scores_topk': [-1, -1, -1],
            'log_interval': 50,
            'valid_interval': 200,
            'patience': 3,
            'valid_interval_re_decay': 0.7,
            'valid_interval_min': 50,
            'lr_re_decay': 0.5,
            'batch_size': 32,
            'lr': 0.001,
            'weight_decay': 0.0,
            'n_epochs': 2,
        }
        args = self.args
        if 'greater_is_better' not in args.__dict__ or args.greater_is_better not in [True, False]:
            raise NotImplementedError('args.greater_is_better must be in [True, False]')
        if args.greater_is_better:
            default_values['valid_scores_topk'] = [-999]*default_values['patience']
        else:
            default_values['valid_scores_topk'] = [999]*default_values['patience']
        for k, v in default_values.items():
            if k not in args.__dict__:
                args.__dict__[k] = v
                print('Fill in arg %s with default value'%(k), v)
    
    def forward(self, x):
        raise NotImplementedError('subclass must implement this')
    
    def get_optimizer_caller(self, optimizer_type):
        choice_d = {'sgd' : torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsp': torch.optim.RMSprop,
                    'adag': torch.optim.Adagrad}
        assert optimizer_type in choice_d
        return choice_d[optimizer_type]
    
    def logit2label(self, logits, onehot=False):
        logits_ = np.array(logits)
        if onehot:
            return (logits_/logits_.max()).astype(np.int8).astype(np.float32)
        else:
            return np.argmax(logits_, axis=len(logits_.shape)-1)
            
    def save(self, path):
        torch.save(self.state_dict(), path)
        print('model saved at', path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print('model loaded from', path)
    
    def save_args_dict(self, args, path):
        with open(path, 'wb') as f:
            pickle.dump(args.__dict__, f)
        print('args_dict saved at', path)
                    
    def load_args_dict(self, path):
        with open(path, 'rb') as f:
            args_dict = pickle.load(f)
        print('args_dict loaded from', path)
        print('returned')
        return args_dict

    def save_finished_args(self, args):
        args.finished = True
        args_fin_path = os.path.join(args.save_sub_dir, 'args.pkl')
        self.save_args_dict(args, args_fin_path)
        print('Finished! Topk:', args.valid_scores_topk)
        return args  

    def fit_batch(self, 
                  X_batch, y_batch, weight=None):
        model = self.train()
        x = Variable(X_batch)
        y = Variable(y_batch).view(-1)
        if self.use_cuda:
            x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        outputs = model(x)
        pred = outputs
        if self.use_cuda:
            pred = pred.cpu()
        loss = self.criterion(outputs, y, weight=weight)
        loss.backward()
        self.optimizer.step()
        return float(loss), pred.data.numpy()
    
    def valid_batch(self, 
                    X_batch, y_batch, weight=None):
        model = self.eval()
        x = Variable(X_batch, volatile=True)
        y = Variable(y_batch).view(-1)
        if self.use_cuda:
            x, y = x.cuda(), y.cuda()
        outputs = model(x)
        pred = outputs
        if self.use_cuda:
            pred = pred.cpu()
        loss = self.criterion(outputs, y, weight=weight)
        return float(loss), pred.data.numpy()

    def predict_batch(self,
                      X_batch):
        model = self.eval()
        x = Variable(X_batch, volatile=True)
        if self.use_cuda:
            x = x.cuda()
        outputs = model(x)
        pred = outputs
        if self.use_cuda:
            pred = pred.cpu()
        return pred.data.numpy()

    def predict(self, 
                X_test=None, use_topk=-1, reduce=True, reduce_mode='weighted'):
        assert X_test is not None or self.test_generator is not None, "Either 'X_test' or 'self.test_generator' need to be provided"
        if X_test is not None:
            test_dataset = DataBuilder(X_test)
            self.test_generator = DataLoader(dataset=test_dataset,
                                             batch_size=self.batch_size)
            print("'self.test_generator' is updated by 'X_test'")
        return self.predict_generator(self.test_generator, 
                                      use_topk=use_topk, 
                                      reduce=reduce, 
                                      reduce_mode=reduce_mode)
    
    def predict_generator(self, 
                          test_generator, 
                          use_topk=-1,
                          reduce=True,
                          reduce_mode='weighted'):
        args = self.args
        model = self.eval()
        print('predict with checkpoint at', args.save_sub_dir)
        pred_all = []
        cnt = 0
        n_pred = len(args.valid_scores_topk)
        if use_topk==-1:
            use_topk = n_pred
        for top_idx in range(use_topk):
            cnt += 1
            pred = []
            model_path = os.path.join(args.save_sub_dir, str(top_idx)+'.pth')
            if os.path.exists(model_path):
                model.load(model_path)
            else:
                continue
            model.eval()
            for bx, _ in test_generator:
                p = model.predict_batch(bx)
                pred.extend(p)
            pred_all.append(pred)
            if cnt==use_topk:
                break
        if not reduce:
            pred_res = pred_all
        elif reduce and reduce_mode=='mean':
            pred_res = np.mean(pred_all, axis=0)
        elif reduce and reduce_mode=='weighted':
            weights = np.array(args.valid_scores_topk[:len(pred_all)])
            weights = np.exp(-weights)/np.exp(-weights).sum()
            pred_res = np.sum([np.array(pred_all[i])*weights[i] for i in range(len(pred_all))], axis=0)
        print('prediction done!')
        return pred_res

    def fit(self, 
            X_train=None, y_train=None, 
            X_valid=None, y_valid=None):
        TRAIN_NULL_FLAG = (X_train is None) or (y_train is None)
        VALID_NULL_FLAG = (X_valid is None) or (y_valid is None)
        assert TRAIN_NULL_FLAG==False or self.train_generator is not None, "Either 'X/y_train' or 'self.train_generator' need to be provided"
        
        args = self.args

        if args.save_sub_dir and not os.path.exists(args.save_sub_dir):
            print("Save path is not existed!")
            print('Creating dir at', args.save_sub_dir)
            os.makedirs(args.save_sub_dir, exist_ok=True)
        
        if not TRAIN_NULL_FLAG:
            train_dataset = DataBuilder(X_train, y_train)
            self.train_generator = DataLoader(dataset=train_dataset,
                                              batch_size=self.batch_size)
            print("'self.train_generator' is updated by 'X/y_train'")
            print('Train with {} samples'.format(len(y_train)))
            
        if not VALID_NULL_FLAG:
            valid_dataset = DataBuilder(X_valid, y_valid)
            self.valid_generator = DataLoader(dataset=valid_dataset,
                                              batch_size=self.batch_size)
            print("'self.valid_generator' is updated by 'X/y_valid'")
            print('Validate with {} samples'.format(len(y_valid)))
        
        args = self.fit_generator(args, 
                                  self.train_generator,
                                  self.valid_generator)
    
    def fit_generator(self, 
                      args, train_generator, valid_generator=None):
        args.n_iter = 0
        args.restarted = 0
        args.finished = False
        args.valid_scores = []
        args.train_begin_time = time()
        
        self.optimizer = self.optimizer_caller(self.parameters(), 
                                               lr=args.lr,
                                               weight_decay=args.weight_decay)
        
        total_loss = 0.0
        for epoch in range(args.n_epochs):
            if args.finished:
                break
            batch_begin_time = time()
            ma_loss = 0.0
            running_pred_train = []
            running_y_train = []
            for batch_idx, (bx, by) in enumerate(train_generator):
                if args.finished:
                    break
                args.n_iter += 1
                loss_tr, pred_tr = self.fit_batch(bx, by)
                total_loss += loss_tr
                ma_loss += loss_tr
                running_pred_train.extend(pred_tr)
                running_y_train.extend(by)
                if args.n_iter % args.log_interval == 0:
                    score = self.eval_metric(running_y_train, 
                                             running_pred_train)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' % (epoch + 1, 
                           args.n_iter, 
                           ma_loss/args.log_interval, 
                           score, 
                           time()-batch_begin_time))
                    ma_loss = 0.0
                    running_pred_train = []
                    running_y_train = []
                    batch_begin_time = time()
                if valid_generator is not None and args.n_iter % args.valid_interval == 0:
                    args = self.evaluate_generator(args, valid_generator)
                    args = self.check_early_stopping(args)
                    print('valid time: %.1f s' % (time()-batch_begin_time))
                    batch_begin_time = time()
        return args
    
    def valid(self, 
              X_valid=None, y_valid=None):
        VALID_NULL_FLAG = (X_valid is None) or (y_valid is None)
        assert VALID_NULL_FLAG==False or self.valid_generator is not None, "Either 'X/y_valid' or 'self.valid_generator' need to be provided"
        
        args = self.args
        
        if not VALID_NULL_FLAG:
            valid_dataset = DataBuilder(X_valid, y_valid)
            self.valid_generator = DataLoader(dataset=valid_dataset,
                                              batch_size=self.batch_size)
            print("'self.valid_generator' is updated by 'X/y_valid'")
            print('Validate with {} samples'.format(len(y_valid)))
        begin_time = time()
        args = self.evaluate_generator(args, self.valid_generator)
        print('valid time: %.1f s' % (time()-begin_time))
    
    def evaluate_generator(self, 
                           args, valid_generator):
        running_pred_valid = []
        running_y_valid = []
        val_total_loss = 0.0
        for _bx,_y in valid_generator:
            loss_val, pred_val = self.valid_batch(_bx,_y)
            running_pred_valid.extend(pred_val)
            running_y_valid.extend(_y)
            val_total_loss += loss_val*len(_y)
        _score = self.eval_metric(running_y_valid, running_pred_valid)
        args.valid_scores.append(_score)
        print('*'*50)
        print('valid loss: %.6f metric: %.6f total time: %.1f s' %
              (val_total_loss/len(running_y_valid), 
               _score, 
               time()-args.train_begin_time))
        print('*'*50)
        return args
    
    def check_early_stopping(self, args):
        _score = args.valid_scores[-1]
        early_stopping_flag = True
        for top_idx, top_scr in enumerate(args.valid_scores_topk):
            if (_score - top_scr > 0) == args.greater_is_better:
                args.valid_scores_topk[top_idx] = _score
                print('Best %d-th valid score:' % top_idx, _score)
                save_topkth_path = os.path.join(args.save_sub_dir, 
                                                str(top_idx)+'.pth')
                self.save(save_topkth_path)
                early_stopping_flag = False
                break
        if early_stopping_flag:
            if args.restarted < args.patience:
                save_top0th_path = os.path.join(args.save_sub_dir, str(0)+'.pth')
                print()
                print('\t\tEarly stopped, restarting from', save_top0th_path)
                print()
                self.load(save_top0th_path)
                args.restarted += 1
                args.valid_interval = max(int(args.valid_interval * args.valid_interval_re_decay), 
                    args.valid_interval_min)
                args.lr = args.lr * args.lr_re_decay
                self.optimizer = self.optimizer_caller(self.parameters(), 
                                                       lr=args.lr, 
                                                       weight_decay=args.weight_decay)
            else:
                args = self.save_finished_args(args)
        return args


# In[10]:


from sklearn.metrics import mean_squared_error

## https://github.com/nzc/dnn_ctr/blob/master/model/FNN.py
class FM(BaseModel):
    def __init__(self):
        super(FM, self).__init__()
    def _eval_metric(self, labels, preds):
        return np.sqrt(mean_squared_error(labels, preds))
    def _criterion(self, input, target, weight=None):
        if weight is None:
            return torch.sqrt(F.mse_loss(input, target, size_average=True))
        else:
            return torch.sum(weight * (input - target) ** 2)
    
    def init_args(self, args, n_output, 
                  feature_sizes, embed_size, 
                  dropout, n_final_state):
        if not torch.cuda.is_available():
            args.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")
        else:
            args.use_cuda = True
        
        torch.manual_seed(223)
        
        self.use_cuda = args.use_cuda
        self.optimizer_caller = self.get_optimizer_caller(args.optimizer_type)
        
        self.criterion = self._criterion
        self.eval_metric = self._eval_metric
        self.batch_size = args.batch_size
        self.args = args
        self.check_args()
        print('args initialized')

        self.n_output = n_output
        self.feature_sizes = feature_sizes
        self.embed_size = embed_size
        self.n_final_state = n_final_state
        self.dropout = dropout
        
        self.pretrain_fm = True
        self.return_final_state = False
        
        self.fm_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True) #w0
        self.fm_first_order_embeddings = nn.ModuleList([
            nn.Embedding(feature_size, 1) for feature_size in feature_sizes
        ]) #wi
        self.fm_second_order_embeddings = nn.ModuleList([
            nn.Embedding(feature_size, embed_size) for feature_size in feature_sizes
        ]) #vi
        
        self.dropout = nn.Dropout(dropout)
        #field_size = len(feature_sizes)
        self.final_state = nn.Linear(1 + len(feature_sizes) + len(feature_sizes) * embed_size, n_final_state)
        self.fc = nn.Linear(n_final_state, n_output)
        
        if self.use_cuda:
            return self.cuda()
        else:
            return self
    def forward(self, x):
        fm_first_order_emb_arr = [emb(x[:, i]) for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_second_order_emb_arr = [emb(x[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        if self.pretrain_fm:
            fm_first_order_sum = sum(fm_first_order_emb_arr).view(-1)
            
            fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
            sum_square = fm_sum_second_order_emb * fm_sum_second_order_emb # (x+y)^2
            fm_second_order_emb_square = [item*item for item in fm_second_order_emb_arr]
            square_sum = sum(fm_second_order_emb_square) #x^2+y^2
            fm_second_order = (sum_square - square_sum) * 0.5
            fm_second_order_sum = torch.sum(fm_second_order, 1)
            x = self.fm_bias+fm_first_order_sum+fm_second_order_sum
            return torch.clamp(x, 0, 1)
        else:
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            fm_second_order = torch.cat(fm_second_order_emb_arr, 1)
            ones = Variable(torch.ones(x.data.shape[0], 1))
            if self.use_cuda:
                ones = ones.cuda()
            fm_bias = self.fm_bias * ones
            deep_emb = torch.cat([fm_bias, fm_first_order, fm_second_order], 1)
            # Dropout & output
            x = self.dropout(deep_emb)  #dim: (batch_size, 1 + field_size + field_size*embed_size)
            x = self.final_state(x)
            if self.return_final_state:
                return x
            else:
                x = F.relu(x)
                x = self.fc(x)
                return torch.clamp(x, 0, 1)


# In[11]:


batch_size = 1024*8
n_epochs = 10

from argparse import Namespace
args = Namespace()

args.use_cuda = True
args.optimizer_type = 'adam'
args.save_sub_dir = '.'
args.patience = 5
args.valid_scores_topk = [999] * args.patience
args.greater_is_better = False
args.log_interval = 100
args.valid_interval = 500
args.valid_interval_re_decay = 0.7
args.valid_interval_min = 50
args.lr_re_decay = 0.5

args.batch_size = batch_size
args.lr=0.001
args.weight_decay=0.0
args.n_epochs=n_epochs
args.model_name='FM'

args.model_params = dict(n_output=1, 
                         feature_sizes=feature_sizes, embed_size=16, 
                         dropout=0.5, 
                         n_final_state=16)

def get_split_masks(eval_sets, valid_fold, test_fold):
    mask_val = eval_sets==valid_fold
    mask_te = eval_sets==test_fold
    mask_tr = ~mask_val & ~mask_te
    return mask_tr, mask_val, mask_te
valid_fold = 0
mask_tr, mask_val, mask_te = get_split_masks(eval_sets, valid_fold, 10)


# In[13]:


model = FM()
model = model.init_args(args, **args.model_params)


# In[14]:

model.pretrain_fm = True
model.fit(df.values[mask_tr], labels[mask_tr], df.values[mask_val], labels[mask_val])



n_epochs = 100
batch_size = 512

args.lr = 0.0001
args.batch_size = batch_size
args.n_epochs=n_epochs

model = FM()
model = model.init_args(args, **args.model_params)
model.load('0.pth')
model.pretrain_fm = False
model.fit(df.values[mask_tr], labels[mask_tr], df.values[mask_val], labels[mask_val])


# In[15]:


if SUBMIT:
    
    pred_val_all = model.predict(df.values[mask_val], use_topk=-1, reduce=False)
    
    
    # In[16]:
    
    
    for pred_val in pred_val_all:
        print(np.sqrt(mean_squared_error(labels[mask_val], pred_val)))
    
    
    # In[17]:
    
    
    topk_avg_scores = []
    for idx, pred_val in enumerate(pred_val_all):
        topk_avg_scores.append(np.sqrt(mean_squared_error(labels[mask_val], np.mean(pred_val_all[:idx+1], 0))))
        print('top %d'%(idx+1), topk_avg_scores[-1])
    topk_wavg_scores = []
    for idx, pred_val in enumerate(pred_val_all):
        weights = np.array(args.valid_scores_topk[:idx+1])
        weights = np.exp(-weights)/np.exp(-weights).sum()
        topk_wavg_scores.append(np.sqrt(mean_squared_error(labels[mask_val], 
                                                           np.dot(np.hstack(pred_val_all[:idx+1]), weights.reshape(-1, 1)))))
        print('top %d weighted'%(idx+1), topk_wavg_scores[-1])
    
    
    # In[18]:
    
    
    if min(topk_avg_scores)<=min(topk_wavg_scores):
        best_topk = np.argmin(topk_avg_scores)+1
        best_reduce_mode = 'mean'
    else:
        best_topk = np.argmin(topk_wavg_scores)+1
        best_reduce_mode = 'weighted'
    best_valid_score = min(min(topk_avg_scores), min(topk_wavg_scores))
    print('best top:', best_topk, 'mode:', best_reduce_mode)
    
    
    # In[19]:
    
    
    pred_val = model.predict(df.values[mask_val], use_topk=best_topk, reduce_mode=best_reduce_mode)
    np.save('valid_%d_pred.npy'%valid_fold, pred_val)
    pred_test = model.predict(df.values[mask_te], use_topk=best_topk, reduce_mode=best_reduce_mode)
    np.save('test_pred.npy', pred_test)
    
    
    # In[20]:
    
    
    sns.distplot(pred_test)
    sns.distplot(labels[mask_val])
    
    
    # In[21]:
    
    
    sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
    sub[target_col] = pred_test
    sub.to_csv(args.model_name+'_%.6f.csv'%best_valid_score, index=False)
    print('save to', args.model_name+'_%.6f.csv'%best_valid_score)
    sub.head()
    
    
    # In[22]:
    
    
    del model; gc.collect()
    model = FM()
    model = model.init_args(args, **args.model_params)
    
    
    # In[23]:
    
    
    model.return_final_state = True
    model.pretrain_fm = False
    test_state = model.predict(df.values[mask_te], use_topk=best_topk, reduce_mode=best_reduce_mode)
    
    
    # In[24]:
    
    
    test_state.shape
    
    
    # In[25]:
    
    
    plt.plot(test_state.mean(0))
    plt.show()
    
    
    # In[26]:
    
    
    valid_state = model.predict(df.values[mask_val], use_topk=best_topk, reduce_mode=best_reduce_mode)
    
    
    # In[27]:
    
    
    from scipy import sparse
    valid_state = sparse.csr_matrix(valid_state)
    test_state = sparse.csr_matrix(test_state)
    sparse.save_npz('valid_%d_state.npz'%valid_fold, valid_state, compressed=True)
    sparse.save_npz('test_state.npz', test_state, compressed=True)
