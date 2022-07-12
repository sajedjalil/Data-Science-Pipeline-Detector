# This notebook is for inference only.
# Our solution code is available in https://github.com/haradai1262/kaggle_riiid-test-answer-prediction

# Manual edit

DEBUG = False

# * GBDT_model name , use seeds
gbdt_name = 'GBDT_cat1'
gbdt_seeds = [
    1209,
    461224,
    20201209,
    20214646,
    46202146,
    46462021,
]

# sequence_model name
seq_model_names = [
    'tran_transformer_nariv2_1',
    'tran_transformer_020_8_20210106171509'
]

# stacking name
stacking_model_name = 'catboost_with_fe_imp_sa3'

INPUT_DIR = '/kaggle/input'
SAVE_NAME_STACKING = f'{INPUT_DIR}/stackingv48'

MAX_SEQ = 120

cfg1 = {'compe': {'name': 'riiid-test-answer-prediction'},
 'common': {'seed': 2020,
  'metrics': {'name': 'auc', 'params': {}},
  'drop': ['lecture_idx'],
  'debug': False,
  'kaggle': {'data': False, 'notebook': False}},
 'model': {'backbone': 'transformer_saint_v6_2',
  'n_classes': 1,
  'epochs': 30,
  'params': {'dim_model': 256,
   'num_en': 2,
   'num_de': 2,
   'heads_en': 8,
   'heads_de': 8,
   'total_ex': 13523,
   'total_cat': 7,
   'total_tg': 188,
   'total_in': 2,
   'total_exp': 2,
   'seq_len': 121},
  'multi_gpu': True,
  'head': None},
 'data': {'train': {'dataset_type': 'CustomTrainDataset7_2',
   'is_train': True,
   'params': {'n_skill': 13523, 'max_seq': 121},
   'loader': {'shuffle': True, 'batch_size': 512, 'num_workers': 4},
   'transforms': None},
  'valid': {'dataset_type': 'CustomTestDataset7_2',
   'is_train': False,
   'params': {'n_skill': 13523, 'max_seq': 121},
   'loader': {'shuffle': False, 'batch_size': 512, 'num_workers': 4},
   'transforms': None},
  'test': {'dataset_type': 'CustomTestDataset7_2',
   'is_train': False,
   'params': {'n_skill': 13523, 'max_seq': 121},
   'loader': {'shuffle': False, 'batch_size': 512, 'num_workers': 4},
   'transforms': None}},
 'loss': {'name': 'BCEWithLogitsLoss', 'params': {}},
 'optimizer': {'name': 'Adam', 'params': {'lr': 0.001}},
 'scheduler': {'name': 'CosineAnnealingLR', 'params': {'T_max': 30}}}

cfg1['model']['params']['total_cnt'] = 2
cfg1['model']['params'].pop('total_exp')

cfg2 = {'compe': {'name': 'riiid-test-answer-prediction'},
 'common': {'seed': 2020,
  'metrics': {'name': 'auc', 'params': {}},
  'drop': ['lecture_idx'],
  'debug': False,
  'kaggle': {'data': False, 'notebook': False}},
 'model': {'backbone': 'transformer_saint_v6_2',
  'n_classes': 1,
  'epochs': 30,
  'params': {'dim_model': 256,
   'num_en': 2,
   'num_de': 2,
   'heads_en': 8,
   'heads_de': 8,
   'total_ex': 13523,
   'total_cat': 7,
   'total_tg': 188,
   'total_in': 2,
   'total_exp': 2,
   'seq_len': 121},
  'multi_gpu': True,
  'head': None},
 'data': {'train': {'dataset_type': 'CustomTrainDataset7_2',
   'is_train': True,
   'params': {'n_skill': 13523, 'max_seq': 121},
   'loader': {'shuffle': True, 'batch_size': 512, 'num_workers': 4},
   'transforms': None},
  'valid': {'dataset_type': 'CustomTestDataset7_2',
   'is_train': False,
   'params': {'n_skill': 13523, 'max_seq': 121},
   'loader': {'shuffle': False, 'batch_size': 512, 'num_workers': 4},
   'transforms': None},
  'test': {'dataset_type': 'CustomTestDataset7_2',
   'is_train': False,
   'params': {'n_skill': 13523, 'max_seq': 121},
   'loader': {'shuffle': False, 'batch_size': 512, 'num_workers': 4},
   'transforms': None}},
 'loss': {'name': 'BCEWithLogitsLoss', 'params': {}},
 'optimizer': {'name': 'Adam', 'params': {'lr': 0.001}},
 'scheduler': {'name': 'CosineAnnealingLR', 'params': {'T_max': 30}}}



import sys
import time
import tarfile
import pandas as pd
import numpy as np
from tqdm import tqdm

def extract_tar_file(path, EXT_DIR):
    with tarfile.open(path, 'r:*') as tar:
        tar.extractall(EXT_DIR)

extract_tar_file('../input/rapids/rapids.0.16.0', '/opt/conda/envs/')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

import catboost
print('CAT Version', catboost.__version__)

sys.path.append('../input/pickle5/pickle5-backport-master')
import pickle5

# transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter('ignore')

import psutil
import os
import time
import math
from contextlib import contextmanager

@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.3f}GB({sign}{delta:.3f}GB):{time.time() - t0:.3f}sec] {title} ", file=sys.stderr)
    
    
def prep_tags(x): return [int(i) for i in x.split()]
def make_content_map_dict():
    questions_df = pd.read_csv(f'{INPUT_DIR}/riiid-test-answer-prediction/questions.csv')
    q2p = dict(questions_df[['question_id', 'part']].values)
    q2p = np.array(list(q2p.values()))

    questions_df['tags'] = questions_df['tags'].fillna(0)
    questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
    questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
    q2tg = dict(questions_df[['question_id', 'tag_list']].values)
    q2tg = np.array(list(q2tg.values()))

    with open(f'{SAVE_NAME_STACKING}/te_content_id_by_answered_correctly_mod.pkl', mode='rb') as f:
        te_dict = pickle5.load(f)
    te_df = pd.DataFrame.from_dict(te_dict).sort_index().iloc[:13523]
    q2te = np.mean(te_df.values, axis=1)
    q2ws = np.load(f'{SAVE_NAME_STACKING}/q2ws.npy')

    return q2p, q2tg, q2te, q2ws



if DEBUG is True:
    with trace("Question settings"):
        q2p, q2tg, q2te, q2ws = make_content_map_dict()

        question = pd.read_csv(f'{INPUT_DIR}/riiid-test-answer-prediction/questions.csv')
        question['tags'] = question['tags'].fillna('0').apply(prep_tags)
        question['part'] = question['part'].fillna(0).astype('int8')
else:
    q2p, q2tg, q2te, q2ws = make_content_map_dict()

    question = pd.read_csv(f'{INPUT_DIR}/riiid-test-answer-prediction/questions.csv')
    question['tags'] = question['tags'].fillna('0').apply(prep_tags)
    question['part'] = question['part'].fillna(0).astype('int8')
    
    
class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len, device):
    mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)).to(device)
    return mask


def get_pos(seq_len, device):
    # use sine positional embeddinds
    return torch.arange(seq_len, device=device).unsqueeze(0)


# https://github.com/arshadshk/SAINT-pytorch/blob/main/saint.py

class Encoder_block_1(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, total_tg, total_cnt, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
        self.embd_cat = nn.Embedding(total_cat + 1, embedding_dim=dim_model)
        self.embd_tg = nn.Embedding(total_tg + 1, embedding_dim=dim_model)
        self.embd_cnt = nn.Embedding(total_cnt, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)
        self.dt_fc = nn.Linear(1, dim_model, bias=False)
        self.cate_proj = nn.Sequential(
            nn.Linear(dim_model*5, dim_model),
            nn.LayerNorm(dim_model),
        )   

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, in_ex, in_cat, in_tg, in_dt, in_cnt, first_block=True):
        device = in_ex.device

        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_cat = self.embd_cat(in_cat)

            in_dt = in_dt.unsqueeze(-1)
            in_dt = self.dt_fc(in_dt)

            in_tg = self.embd_tg(in_tg)
            avg_in_tg_embed = in_tg.mean(dim=2)
            max_in_tg_embed = in_tg.max(dim=2).values
            
            in_cnt = self.embd_cnt(in_cnt)

            # combining the embedings
            # out = in_ex + in_cat + in_dt + (avg_in_tg_embed + max_in_tg_embed) + in_cnt
            out = torch.cat([in_ex, in_cat, in_dt, (avg_in_tg_embed + max_in_tg_embed), in_cnt], axis=2)
            out = self.cate_proj(out)
        else:
            out = in_ex

        in_pos = get_pos(self.seq_len, device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=get_mask(seq_len=n, device=device))
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        return out


class Decoder_block_1(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """
    def __init__(self, dim_model, total_in, heads_de, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.embd_in = nn.Embedding(total_in, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(self.seq_len, embedding_dim=dim_model)
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.el_fc = nn.Linear(1, dim_model, bias=False)
#         self.el_fc = nn.Sequential(
#             nn.Linear(1, dim_model, bias=False),
#             nn.LayerNorm(dim_model)
#         )
        self.cate_proj = nn.Sequential(
            nn.Linear(dim_model*2, dim_model),
            nn.LayerNorm(dim_model),
        )   
        
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, in_in, in_el, en_out, first_block=True):
        device = in_in.device

        if first_block:
            in_in = self.embd_in(in_in)

            in_el = in_el.unsqueeze(-1)
            in_el = self.el_fc(in_el)

            # out = in_in + in_el
            out = torch.cat([in_in, in_el], axis=2)
            out = self.cate_proj(out)
        else:
            out = in_in

        in_pos = get_pos(self.seq_len, device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape

        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n, device=device))
        out = skip_out + out

        en_out = en_out.permute(1, 0, 2)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n, device=device))
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        return out


class SAINT_1(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_tg, total_in, total_cnt,
                      heads_de, seq_len, num_fc_in_dim=2, num_fc_out_dim=128):
        super().__init__()

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones(Encoder_block_1(dim_model, heads_en, total_ex, total_cat, total_tg, total_cnt, seq_len), num_en)
        self.decoder = get_clones(Decoder_block_1(dim_model, total_in, heads_de, seq_len), num_de)
        
        self.num_fc = nn.Linear(in_features=num_fc_in_dim, out_features=num_fc_out_dim)
        self.out_fc1 = nn.Linear(in_features=dim_model, out_features=num_fc_out_dim)
        self.out_fc2 = nn.Linear(in_features=num_fc_out_dim * 2, out_features=1)

    def forward(self, feat):
        in_ex = feat['in_ex']
        in_dt = feat['in_dt']
        in_el = feat['in_el']
        in_tg = feat['in_tag']
        in_cat = feat['in_cat']
        in_in = feat['in_de']
        ###
        num_feat = feat['num_feat']
        in_cnt = feat['in_cnt']
        ###

        first_block = True
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            in_ex = self.encoder[x](in_ex, in_cat, in_tg, in_dt, in_cnt, first_block=first_block)
            in_cat = in_ex

        first_block = True
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            in_in = self.decoder[x](in_in, in_el, en_out=in_ex, first_block=first_block)

        num_feat = self.num_fc(num_feat)
        in_in = self.out_fc1(in_in)
        in_in = torch.cat([in_in, num_feat], dim=2)
        in_in = self.out_fc2(in_in)
    
        return in_in.squeeze(-1)

def replace_fc(model, cfg):
    return model

class CustomModel_1(nn.Module):
    def __init__(self, cfg):
        super(CustomModel_1, self).__init__()
        self.cfg = cfg
        self.base_model = SAINT_1(**cfg['model']['params'])
        self.model = replace_fc(self.base_model, cfg)

    def forward(self, x):
        x = self.model(x)
        return x
    
    
    
# https://github.com/arshadshk/SAINT-pytorch/blob/main/saint.py

class Encoder_block_2(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, total_tg, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
        self.embd_cat = nn.Embedding(total_cat + 1, embedding_dim=dim_model)
        self.embd_tg = nn.Embedding(total_tg + 1, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)
        self.dt_fc = nn.Linear(1, dim_model, bias=False)

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, in_ex, in_cat, in_tg, in_dt, first_block=True):
        device = in_ex.device

        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_cat = self.embd_cat(in_cat)

            in_dt = in_dt.unsqueeze(-1)
            in_dt = self.dt_fc(in_dt)

            in_tg = self.embd_tg(in_tg)
            avg_in_tg_embed = in_tg.mean(dim=2)
            max_in_tg_embed = in_tg.max(dim=2).values

            out = in_ex + in_cat + in_dt + (avg_in_tg_embed + max_in_tg_embed)
        else:
            out = in_ex

        in_pos = get_pos(self.seq_len, device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=get_mask(seq_len=n, device=device))
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        return out


class Decoder_block_2(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """
    def __init__(self, dim_model, total_in, total_exp, heads_de, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.embd_in = nn.Embedding(total_in, embedding_dim=dim_model)
        # self.embd_exp = nn.Embedding(total_exp, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(self.seq_len, embedding_dim=dim_model)
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.el_fc = nn.Linear(1, dim_model, bias=False)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, in_in, in_el, en_out, first_block=True):
        device = in_in.device

        if first_block:
            in_in = self.embd_in(in_in)

            in_el = in_el.unsqueeze(-1)
            in_el = self.el_fc(in_el)

            out = in_in + in_el
        else:
            out = in_in

        in_pos = get_pos(self.seq_len, device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape

        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n, device=device))
        out = skip_out + out

        en_out = en_out.permute(1, 0, 2)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n, device=device))
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        return out


class SAINT_2(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_tg, total_in, total_exp, heads_de, seq_len):
        super().__init__()

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones(Encoder_block_2(dim_model, heads_en, total_ex, total_cat, total_tg, seq_len), num_en)
        self.decoder = get_clones(Decoder_block_2(dim_model, total_in, total_exp, heads_de, seq_len), num_de)

        self.out = nn.Linear(in_features=dim_model, out_features=1)

    def forward(self, feat):
        in_ex = feat['in_ex']
        in_dt = feat['in_dt']
        in_el = feat['in_el']
        in_tg = feat['in_tag']
        in_cat = feat['in_cat']
        in_in = feat['in_de']

        first_block = True
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            in_ex = self.encoder[x](in_ex, in_cat, in_tg, in_dt, first_block=first_block)
            in_cat = in_ex

        first_block = True
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            in_in = self.decoder[x](in_in, in_el, en_out=in_ex, first_block=first_block)

        in_in = self.out(in_in)

        return in_in.squeeze(-1)

def replace_fc(model, cfg):
    return model

class CustomModel_2(nn.Module):
    def __init__(self, cfg):
        super(CustomModel_2, self).__init__()
        self.cfg = cfg
        self.base_model = SAINT_2(**cfg['model']['params'])
        self.model = replace_fc(self.base_model, cfg)
    def forward(self, x):
        x = self.base_model(x)
        return x

class CustomTestDataset7_2(Dataset):
    def __init__(self, df, q2p, q2tg, q2te, q2ws, cfg=None):
        super(CustomTestDataset7_2, self).__init__()
        self.max_seq = cfg['params']['max_seq']
        self.n_skill = cfg['params']['n_skill']
        self.df = df
        self.n_tag = 188

        self.q2p = q2p
        self.q2tg = q2tg
        ###
        self.q2te = q2te
        self.q2ws = q2ws
        ###

    def __len__(self):
        return len(self.df)

    def np_append(self, seq, val, dtype=int):
        new_seq = np.zeros(self.max_seq-1, dtype=dtype)
        new_seq[:-1] = seq[2:]
        new_seq[-1] = val
        return new_seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']
        uid = row["user_id"]
        cid = row["content_id"]
        timestamp = row["timestamp"]
        et = row["prior_question_elapsed_time"]

        cid_seq = np.zeros(self.max_seq, dtype=int)
        dt_seq = np.zeros(self.max_seq, dtype=int)
        et_seq = np.zeros(self.max_seq, dtype=int)
        qa_seq = np.zeros(self.max_seq, dtype=int)
        seq_len = 0
            
        # cid
        u_prev_cids = riiidFE.user_prev_ques_dict[uid]
        seq_len = len(u_prev_cids)
        if seq_len > 0:     
            # difftime
            difftime = riiidFE.user_timestamp_dict[uid] + [timestamp]
            difftime = np.diff(difftime)

            # elapsed_time
            elapsedtime = riiidFE.user_et_dict[uid]
            if len(elapsedtime) != 0:
                # elapsedtime[0] = np.nan
                elapsedtime[0] = 0

            # answer_correctly
            qa = np.array(list(riiidFE.user_rec_dict[uid]), dtype=np.uint8)

            if seq_len >= self.max_seq:
                cid_seq = u_prev_cids[-self.max_seq:]
                dt_seq = difftime[-self.max_seq:]
                et_seq = elapsedtime[-self.max_seq:]
                qa_seq = qa[-self.max_seq:]
            else:
                cid_seq[-seq_len:] = u_prev_cids
                dt_seq[-seq_len:] = difftime
                et_seq[-seq_len:] = elapsedtime
                qa_seq[-seq_len:] = qa
                
        ## postprocess
        dt_seq = np.array(dt_seq) / 60_000.   # ms -> m
        dt_seq = np.where(dt_seq < 0, 300, dt_seq)
        dt_seq = np.log1p(dt_seq)[1:]
        
        et_seq = self.np_append(et_seq, et, dtype=float)
        et_seq = np.array(et_seq) / 1_000.
        et_seq = np.log1p(et_seq)
        et_seq = np.where(np.isnan(et_seq), np.log1p(21), et_seq)

        cid_seq = self.np_append(cid_seq, cid, dtype=int)
        learn_start_idx = np.where(cid_seq > 0)[0][0]   # 変更した

        part_seq = np.zeros(self.max_seq - 1)
        part_seq[learn_start_idx:] = self.q2p[cid_seq[learn_start_idx:]]   # 変更した

        qtg_seq = np.zeros((self.max_seq - 1, 6)) + self.n_tag
        qtg_seq[learn_start_idx:, :] = self.q2tg[cid_seq[learn_start_idx:]]   # 変更した

        qa_seq = qa_seq[1:]
        
        ###
        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        ac_latest = np.array(qa_seq[learn_start_idx:])
        avg_u_target[learn_start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)
        
        te_content_id = np.zeros(self.max_seq - 1)
        te_content_id[learn_start_idx:] = self.q2te[cid_seq[learn_start_idx:]]
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        
        num_feat = np.vstack([te_content_id, avg_u_target]).T

        cnt = np.zeros(self.max_seq - 1)
        unique_content_id = []
        for idx in range(learn_start_idx, self.max_seq - 1):
            id_ = cid_seq[idx]
            if id_ in unique_content_id:
                cnt[idx] = 1
            else:
                cnt[idx] = 0
                unique_content_id.append(id_)
        
        feat = {
            'in_ex': cid_seq,
            'in_dt': dt_seq,
            'in_el': et_seq,
            'in_tag': qtg_seq,
            'in_cat': part_seq,
            'in_de': qa_seq,
            ###
            'num_feat': num_feat,
            'in_cnt': cnt,
            ###
        }
        return feat

def _test_epoch_multi(models, test_dataset, bs=2048):
    models = [m.eval() for m in models]
    
    dataset_len = test_dataset.__len__()
    
    test_preds = [np.zeros(dataset_len) for i in range(len(models))]
    if dataset_len < bs:
        bs = test_dataset.__len__()
        b_in_ex = np.zeros((bs, MAX_SEQ))
        b_in_dt = np.zeros((bs, MAX_SEQ))
        b_in_el = np.zeros((bs, MAX_SEQ))
        b_in_tag = np.zeros((bs, MAX_SEQ, 6))
        b_in_cat = np.zeros((bs, MAX_SEQ))
        b_in_de = np.zeros((bs, MAX_SEQ))
        b_in_feat = np.zeros((bs, MAX_SEQ, 2))
        b_in_cnt = np.zeros((bs, MAX_SEQ))
        for i in range(bs):
            row = test_dataset[i]
            b_in_ex[i] = row['in_ex']
            b_in_dt[i] = row['in_dt']
            b_in_el[i] = row['in_el']
            b_in_tag[i] = row['in_tag']
            b_in_cat[i] = row['in_cat']
            b_in_de[i] = row['in_de']
            b_in_feat[i] = row['num_feat']
            b_in_cnt[i] = row['in_cnt']
            
        feats = {
            'in_ex': torch.LongTensor(b_in_ex).to(device),
            'in_dt': torch.FloatTensor(b_in_dt).to(device),
            'in_el': torch.FloatTensor(b_in_el).to(device),
            'in_tag': torch.LongTensor(b_in_tag).to(device),
            'in_cat': torch.LongTensor(b_in_cat).to(device),
            'in_de': torch.LongTensor(b_in_de).to(device),
            'num_feat': torch.FloatTensor(b_in_feat).to(device),
            'in_cnt':torch.LongTensor(b_in_cnt).to(device),
        }
        with torch.no_grad():
            for midx, m in enumerate(models):
                preds = m(feats)
                preds = preds[:, -1]
                test_preds[midx] = preds.sigmoid().cpu().detach().numpy()
    else:
        batch_set_num = dataset_len // bs + 1
        for bset_idx in range(batch_set_num):
            
            bst = bset_idx * bs
            if (bset_idx+1) * bs > dataset_len:
                bs = dataset_len - bset_idx * bs
            bed = bst + bs
            
            b_in_ex = np.zeros((bs, MAX_SEQ))
            b_in_dt = np.zeros((bs, MAX_SEQ))
            b_in_el = np.zeros((bs, MAX_SEQ))
            b_in_tag = np.zeros((bs, MAX_SEQ, 6))
            b_in_cat = np.zeros((bs, MAX_SEQ))
            b_in_de = np.zeros((bs, MAX_SEQ))
            b_in_feat = np.zeros((bs, MAX_SEQ, 2))
            b_in_cnt = np.zeros((bs, MAX_SEQ))
            for i in range(bs):
                row = test_dataset[i]
                b_in_ex[i] = row['in_ex']
                b_in_dt[i] = row['in_dt']
                b_in_el[i] = row['in_el']
                b_in_tag[i] = row['in_tag']
                b_in_cat[i] = row['in_cat']
                b_in_de[i] = row['in_de']
                b_in_feat[i] = row['num_feat']
                b_in_cnt[i] = row['in_cnt']
            feats = {
                'in_ex': torch.LongTensor(b_in_ex).to(device),
                'in_dt': torch.FloatTensor(b_in_dt).to(device),
                'in_el': torch.FloatTensor(b_in_el).to(device),
                'in_tag': torch.LongTensor(b_in_tag).to(device),
                'in_cat': torch.LongTensor(b_in_cat).to(device),
                'in_de': torch.LongTensor(b_in_de).to(device),
                'num_feat': torch.FloatTensor(b_in_feat).to(device),
                'in_cnt':torch.LongTensor(b_in_cnt).to(device),
            }
            with torch.no_grad():
                for midx, m in enumerate(models):
                    preds = m(feats)
                    preds = preds[:, -1]
                    test_preds[midx][bst: bed] = preds.sigmoid().cpu().detach().numpy()
    return test_preds


# Load riiidFE
if DEBUG is True:
    with trace("Load riiidFE"):
        with open(f'{SAVE_NAME_STACKING}/riiidFE.pkl', mode='rb') as f:
            riiidFE = pickle5.load(f)
else:
    with open(f'{SAVE_NAME_STACKING}/riiidFE.pkl', mode='rb') as f:
        riiidFE = pickle5.load(f)


# Load GBDT
if DEBUG is True:
    with trace("Load GBDT"):
        gbdt_models = {}
        for s in gbdt_seeds:
            with open(f'{SAVE_NAME_STACKING}/gbdt_model/cat_model_{s}.pkl', mode='rb') as f:
                model = pickle5.load(f)
                gbdt_models[f'{gbdt_name}_{s}'] = model
else:
    gbdt_models = {}
    for s in gbdt_seeds:
        with open(f'{SAVE_NAME_STACKING}/gbdt_model/cat_model_{s}.pkl', mode='rb') as f:
            model = pickle5.load(f)
            gbdt_models[f'{gbdt_name}_{s}'] = model


# Load Sequence model
device = 'cuda:0'
seq_model_names_dict = {
    'tran_transformer_nariv2_1': CustomModel_1,
    'tran_transformer_020_8_20210106171509': CustomModel_2 
}
seq_model_cfg_dict = {
    'tran_transformer_nariv2_1': cfg1,
    'tran_transformer_020_8_20210106171509': cfg2 
}
seq_models = []
for seq_name in seq_model_names:
    weight_path = f'{SAVE_NAME_STACKING}/seq_model/{seq_name}.pt'
    if DEBUG is True:
        with trace("Load Sequence model"):
            seq_model = seq_model_names_dict[seq_name](seq_model_cfg_dict[seq_name]).to(device)
            seq_model.load_state_dict(torch.load(weight_path))
            seq_models.append(seq_model)
    else:
        seq_model = seq_model_names_dict[seq_name](seq_model_cfg_dict[seq_name]).to(device)
        seq_model.load_state_dict(torch.load(weight_path))
        seq_models.append(seq_model)


# Load Stacking model
if DEBUG is True:
    with trace("Load Load Stacking model"):
        with open(f'{SAVE_NAME_STACKING}/{stacking_model_name}/stacking_cat_models.pkl', mode='rb') as f:
            stacking_models = pickle5.load(f)
        with open(f'{SAVE_NAME_STACKING}/{stacking_model_name}/stacking_cat_predsnames.pkl', mode='rb') as f:
            predsnames = pickle5.load(f)
        with open(f'{SAVE_NAME_STACKING}/{stacking_model_name}/stacking_cat_use_features.pkl', mode='rb') as f:
            stacking_features = pickle5.load(f)
else:
    with open(f'{SAVE_NAME_STACKING}/{stacking_model_name}/stacking_cat_models.pkl', mode='rb') as f:
        stacking_models = pickle5.load(f)
    with open(f'{SAVE_NAME_STACKING}/{stacking_model_name}/stacking_cat_predsnames.pkl', mode='rb') as f:
        predsnames = pickle5.load(f)
    with open(f'{SAVE_NAME_STACKING}/{stacking_model_name}/stacking_cat_use_features.pkl', mode='rb') as f:
        stacking_features = pickle5.load(f)

        
import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


if DEBUG is True:

    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:

        if previous_test_df is not None:
            with trace("update riiidFE"):
                previous_test_df['answered_correctly'] = eval(test_df["prior_group_answers_correct"].iloc[0])
                previous_test_df['user_answer'] = eval(test_df["prior_group_responses"].iloc[0])
                riiidFE.add_user_feats(previous_test_df, add_feat=False, update_dict=True, val=True)

        with trace("test_df preprocess"):
            test_df = pd.merge(test_df, question, left_on='content_id', right_on='question_id',  how="left")
            test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(False).astype('int8')
            test_df['prior_question_elapsed_time'] = test_df['prior_question_elapsed_time'].fillna(-1).astype(int)
            test_df['part'] = test_df['part'].fillna(0.0).astype('int8')

        with trace("test_df copy"):
            previous_test_df = test_df.copy()

        with trace("riiidFE.add_user_feats"):
            user_feat_df = riiidFE.add_user_feats(test_df, add_feat=True, update_dict=False, val=True)
            test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
            test_df = pd.concat([test_df, user_feat_df], axis=1)

        with trace("merge content_id_df"):
            test_df = pd.concat([test_df.reset_index(drop=True), riiidFE.content_id_df.reindex(test_df['content_id'].values).reset_index(drop=True).iloc[:, 1:]], axis=1)

        # INFERENCE
        preds_list = []
        preds_names = []

        # INFERENCE GBDT
        with trace("INFERENCE GBDT"):
            for model_name, model in gbdt_models.items():
                gbdt_preds = model.predict_proba(test_df[riiidFE.use_features])[:, 1]

                preds_list.append(gbdt_preds)
                preds_names.append(model_name)

        # INFERENCE Seq Model
        with trace("INFERENCE Seq Model"):
            test_dataset = CustomTestDataset7_2(test_df, q2p, q2tg, q2te, q2ws, cfg1['data']['valid'])
            seq_preds_list = _test_epoch_multi(seq_models, test_dataset, bs=2048)
            for i in range(len(seq_models)):
                preds_list.append(seq_preds_list[i])
                preds_names.append(seq_model_names[i])

        # Ensemble
        with trace("Ensemble"):
            input_df = pd.DataFrame(
                np.array(preds_list).T,
                columns=preds_names
            )[predsnames]
            input_df = pd.concat([input_df, test_df[stacking_features]], axis=1)

            preds = np.zeros(len(test_df))
            for smodel in stacking_models:
                preds += smodel.predict_proba(input_df)[:, 1]
            preds /= len(stacking_models)

        test_df['answered_correctly'] = preds
        env.predict(test_df[['row_id', 'answered_correctly']])

else:
    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:

        if previous_test_df is not None:
            previous_test_df['answered_correctly'] = eval(test_df["prior_group_answers_correct"].iloc[0])
            previous_test_df['user_answer'] = eval(test_df["prior_group_responses"].iloc[0])
            riiidFE.add_user_feats(previous_test_df, add_feat=False, update_dict=True, val=True)

        test_df = pd.merge(test_df, question, left_on='content_id', right_on='question_id',  how="left")
        test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(False).astype('int8')
        test_df['prior_question_elapsed_time'] = test_df['prior_question_elapsed_time'].fillna(-1).astype(int)
        test_df['part'] = test_df['part'].fillna(0.0).astype('int8')
        
        previous_test_df = test_df.copy()
        
        user_feat_df = riiidFE.add_user_feats(test_df, add_feat=True, update_dict=False, val=True)
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
        test_df = pd.concat([test_df, user_feat_df], axis=1)

        test_df = pd.concat([test_df.reset_index(drop=True), riiidFE.content_id_df.reindex(test_df['content_id'].values).reset_index(drop=True).iloc[:, 1:]], axis=1)

        # INFERENCE
        preds_list = []
        preds_names = []

        # INFERENCE GBDT
        for model_name, model in gbdt_models.items():
            gbdt_preds = model.predict_proba(test_df[riiidFE.use_features])[:, 1]

            preds_list.append(gbdt_preds)
            preds_names.append(model_name)

        # INFERENCE Seq Model
        test_dataset = CustomTestDataset7_2(test_df, q2p, q2tg, q2te, q2ws, cfg1['data']['valid'])
        seq_preds_list = _test_epoch_multi(seq_models, test_dataset, bs=2048)
        for i in range(len(seq_models)):
            preds_list.append(seq_preds_list[i])
            preds_names.append(seq_model_names[i])

        # Ensemble
        input_df = pd.DataFrame(
            np.array(preds_list).T,
            columns=preds_names
        )[predsnames]
        input_df = pd.concat([input_df, test_df[stacking_features]], axis=1)

        preds = np.zeros(len(test_df))
        for smodel in stacking_models:
            preds += smodel.predict_proba(input_df)[:, 1]
        preds /= len(stacking_models)

        test_df['answered_correctly'] = preds
        env.predict(test_df[['row_id', 'answered_correctly']])