# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import sys

import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import gc
#import utils

ROBERTA_PATH = "../input/robertalargesquad2"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)
MAX_LEN = 192
VALID_BATCH_SIZE = 8


def seed_torch(seed_value):
    #random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_data(dfx):
    dfx['sentiment_target'] = pd.factorize(dfx['sentiment'])[0]
    dfx['text'] = [' '+t for t in dfx['text']]
    return dfx

dots = ['.' * i for i in range(300)]
space_dots = [''] + [' ' + '.' * (i-1) for i in range(1,300)]
questions = ['?' * i for i in range(300)]
space_questions = [''] + [' ' + '?' * (i-1) for i in range(1,300)]
marks = ['!' * i for i in range(300)]
space_marks = [''] + [' ' + '!' * (i-1) for i in range(1,300)]

def map_token(token_id, offset, text):
    (offset1, offset2) = offset
    token_size = (offset2 - offset1)
    if token_size == 0:
        return [token_id], [offset]
    if text[offset1 : offset2] ==  dots[token_size]:
        return [4]*token_size, [(k, k+1) for k in range(offset1, offset2)]
    elif token_size >= 2 and text[offset1 : offset2] ==  space_dots[token_size]:
        return ([479] + [4]*(token_size - 2), 
                [(offset1, offset1+2)]+[(k, k+1) for k in range(offset1+2, offset2)])
    elif text[offset1 : offset2] ==  questions[token_size]:
        return [116]*token_size, [(k, k+1) for k in range(offset1, offset2)]
    elif token_size >= 2 and text[offset1 : offset2] ==  space_questions[token_size]:
        return ([17487] + [116]*(token_size - 2), 
                [(offset1, offset1+2)]+[(k, k+1) for k in range(offset1+2, offset2)])
    elif text[offset1 : offset2] ==  marks[token_size]:
        return [328]*token_size, [(k, k+1) for k in range(offset1, offset2)]
    elif token_size >= 2 and text[offset1 : offset2] ==  space_marks[token_size]:
        return ([27785] + [328]*(token_size - 2), 
                [(offset1, offset1+2)]+[(k, k+1) for k in range(offset1+2, offset2)])
    return [token_id], [offset]
    
def map_tokens(token_ids, offsets, text):
    maps = [map_token(token_id, offset, text) for token_id, offset in zip(token_ids, offsets)]
    token_ids = [t for m in maps for t in m[0]]
    offsets = [o for m in maps for o in m[1]]
    return token_ids, offsets

def process_data(tweet, selected_text, sentiment, tokenizer, max_len, unicode=True):
    orig_tweet = tweet
    orig_selected_text = selected_text

    idx0 = tweet.find(selected_text)
    idx1 = idx0 + len(selected_text) - 1
    
    strange_quote = 'ï¿½'
    prev_ind = 0
    max_tweet_len = len(tweet) - 2
    found = True
    found_idx = []
    while unicode and found:
        found = False
        for ind in range(prev_ind, max_tweet_len):
            if tweet[ind:ind+3] == strange_quote:
                found = True
                found_idx.append(ind)
                tweet = tweet[:ind] + "'" + tweet[ind+3:]
                max_tweet_len = len(tweet) - 2
                if idx0 > ind:
                    idx0 = max(ind, idx0 - 2)
                if idx1 > ind:
                    idx1 = max(ind, idx1 - 2)
                prev_ind = ind
                break
    
    selected_text = tweet[idx0:idx1+1]

    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_length = len(input_ids_orig)
    offsets = []
    prefixes = []
    prev_offset = 0
    for i in range(tweet_length):
        prefix = tokenizer.decode(input_ids_orig[:i+1])
        prefixes.append(prefix)
        offsets.append((prev_offset, len(prefix)))
        prev_offset = len(prefix)
        
    input_ids_orig, offsets = map_tokens(input_ids_orig, offsets, tweet)
    
    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
        
                
    if len(found_idx) > 0:
        new_offsets = []
        for o1, o2 in offsets:
            for ind in found_idx:
                if o1 > ind:
                    o1 += 2
                if o2 >= ind:
                    o2 += 2
            new_offsets.append((o1, o2))
        offsets = new_offsets
            

    tweet_length = len(input_ids_orig)
    
    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974,
        'space':1437,
    }
    question = ' What %s sentiment?'
    q_ids = tokenizer.encode((question % sentiment)).ids
    len_q = 3 + len(q_ids)
    input_ids = [0] + q_ids + [2] + [2] + input_ids_orig + [2]
    offsets = [(0, 0)] * len_q + offsets + [(0, 0)]
    token_type_ids = [0] * (2 + len(q_ids)) + [1] * (len(input_ids_orig) + 2)
    mask = [1] * len(token_type_ids)

    len_tweet = len(input_ids)    
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)
    

    return {
        'ids': input_ids,
        'offsets': offsets,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'len_tweet':len_tweet,
        'len_q':len_q,
    }

class TweetDataset:
    """
    Dataset which stores the tweets and returns them as processed features
    """
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        #self.dicts = [self.get_data(item) for item in range(len(tweet))]
    
    def __len__(self):
        return len(self.tweet)
    
    def get_data(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        return data

    def __getitem__(self, item):
        #data = self.dicts[item]
        data = self.get_data(item)
        return data

def tweet_collate(batch):
    batch_dict = {}
    max_len = np.max([b['len_tweet'] for b in batch])
    #max_len = max(max_len, MAX_LEN)

    for key in ["ids", "mask", "token_type_ids", "offsets",]:
        batch_dict[key] = (torch.from_numpy(np.stack([b[key][:max_len] for b in batch])).long())
        
    for key in ["len_tweet", 'len_q']:
        batch_dict[key] = torch.from_numpy(np.stack([b[key] for b in batch])).long()
        
    input_keys = ["ids", "mask", "token_type_ids"]
    eval_keys = ["offsets", "len_tweet", 'len_q']
    input_dict = {k: batch_dict[k] for k in input_keys}
    eval_dict = {k: batch_dict[k] for k in eval_keys}
    return input_dict, eval_dict

class TweetModel(transformers.BertPreTrainedModel):
    """
    Model class that combines a pretrained bert model with a linear later
    """
    def __init__(self, conf, path):
        super(TweetModel, self).__init__(conf)
        self.transformer = transformers.RobertaModel.from_pretrained(path, config=conf)
        self.drop_out = nn.Dropout(0.1)
        #self.l0 = nn.Linear(self.transformer.config.hidden_size * 2, 2)
        #torch.nn.init.normal_(self.l0.weight, std=0.02)
        self.l1 = nn.Linear(self.transformer.config.hidden_size * 1, 1, bias=True)
        self.conv1 = nn.Conv1d(self.transformer.config.hidden_size * 1, 1, 2, padding=0)    
        self.conv2 = nn.Conv1d(self.transformer.config.hidden_size * 1, 1, 2, padding=0)  
        
    def compute_embeddings(self, input_dict):
        ids = input_dict["ids"]
        out = self.transformer.embeddings(ids)
        return out

    def forward(self, input_dict):
        ids = input_dict["ids"]
        token_type_ids = input_dict["token_type_ids"]
        mask = input_dict["mask"]

        _, _, out = self.transformer(
                ids,
                attention_mask=mask,
                #token_type_ids=token_type_ids
            ) # bert_layers x bs x SL x (768 * 2)

        out = out[-1] # bs x SL x (768 * 1)
        out = self.drop_out(out) # bs x SL x (768 * 1)
        out = out * token_type_ids.unsqueeze(-1)
        tout = out.transpose(1,2)
        start_logits = self.conv1(tout)             
        end_logits = self.conv2(tout)             
        
        start_logits = start_logits.squeeze(1) # (bs x SL-1)
        end_logits = end_logits.squeeze(1) # (bs x SL-1)
        
        start_logits = F.pad(start_logits, (0, 1))
        end_logits = F.pad(end_logits, (1, 0))
        
        return {'start_logits':start_logits, 'end_logits':end_logits, 
                }

def get_model(dirname, fname, seed, fold, device, path):
    model_config = transformers.RobertaConfig.from_pretrained(path)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config, path=path)

    model.to(device)
    model_path="../input/%s/model_half_%s_%d_%d.bin" % (dirname, fname, seed, fold)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_best_start_end_idxs(_start_logits, _end_logits):
    best_logit = -1000
    best_idxs = None
    for start_idx, start_logit in enumerate(_start_logits):
        for end_idx, end_logit in enumerate(_end_logits[start_idx:]):
            logit_sum = (start_logit + end_logit).item()
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (start_idx, start_idx+end_idx)
    #print(best_idxs)
    return best_idxs

def get_prediction(original_tweet, sentiment_val, 
                   start_logits, end_logits, offsets,):
    
    idx_start, idx_end, = get_best_start_end_idxs(start_logits, end_logits)
    #print(idx_start, idx_end)
    filtered_output  = original_tweet[offsets[idx_start][0] : offsets[idx_end][1]]
    if sentiment_val == "neutral":
        filtered_output = original_tweet
    return filtered_output

def eval_test(dirname, fname, dfx, data_loader, seed_range, fold_range, path):
    """
    Eval model for all folds
    """
    # Read training csv

    seed_fold = [(seed, fold) for seed in seed_range for fold in fold_range]
    for (seed, fold) in (seed_fold):
        print(fname, seed, fold)
        seed_torch(seed_value=fold)

        device = torch.device("cuda")
        model = get_model(dirname, fname, seed, fold, device, path)
        
        starts = []
        ends = []
        offsets_s = []
        with torch.no_grad():
            for bi, batch in enumerate(data_loader):

                input_dict, eval_dict = batch
                input_dict = {k: input_dict[k].to(device) for k in input_dict}
                token_ids = input_dict["ids"].cpu().detach().numpy()

                outputs_dict = model(input_dict)

                start_logits = outputs_dict['start_logits'].cpu().detach().numpy()
                end_logits = outputs_dict['end_logits'].cpu().detach().numpy()
                offsets = eval_dict["offsets"].cpu().detach().numpy()
                len_tweets = eval_dict["len_tweet"].cpu().detach().numpy()
                len_qs = eval_dict["len_q"].cpu().detach().numpy()

                start_logits = [a[len_q:len_tweet] for a, len_q, len_tweet in 
                               zip(start_logits, len_qs, len_tweets)]
                end_logits = [a[len_q:len_tweet] for a, len_q, len_tweet in 
                             zip(end_logits, len_qs, len_tweets)]
                offsets = [a[len_q:len_tweet] for a, len_q, len_tweet in 
                          zip(offsets, len_qs, len_tweets)]
                

                starts.extend(start_logits)
                ends.extend(end_logits)
                offsets_s.extend(offsets)
                del input_dict, eval_dict, batch, token_ids, outputs_dict, 
                del start_logits, end_logits, offsets, len_tweets, len_qs, 
                
        
        dfx['start_logits'] = [x+y for x,y in zip(dfx['start_logits'], starts)]
        dfx['end_logits']  = [x+y for x,y in zip(dfx['end_logits'], ends)]
        dfx['offsets'] = offsets_s
        del model, starts, ends, offsets_s
        gc.collect()

    print('*' * 80)
    return dfx

def eval_sub(sub):
    sub['start_logits'] /= 21
    sub['end_logits'] /= 21
    sub['filtered']  = [get_prediction(*params)
                      for params in zip(sub['text'], sub['sentiment'], 
                                        sub['start_logits'], sub['end_logits'], 
                                        sub['offsets'])]

    sub['score'] = [jaccard(*strs) for strs in zip(sub['filtered'], sub['selected_text'])]

    print(f"Jaccard = {np.mean(sub['score'])}")
    return sub

dfx = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
dfx = preprocess_data(dfx)
dfx['start_logits'] = 0
dfx['end_logits']  = 0
dfx['selected_text']  = dfx['text']

valid_dataset = TweetDataset(
    tweet=dfx.text.values,
    sentiment=dfx.sentiment.values,
    selected_text=dfx.selected_text.values
)

data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    num_workers=0,
    collate_fn=tweet_collate,
)
if len(dfx) < 5000:
    sample = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    sample['selected_text'] = ''
    sample.to_csv('submission.csv', index=False)
else:
    eval_test('transformer-236-2','transformer_236', dfx, data_loader, range(6,7), range(5), '../input/robertalargesquad2')
    eval_test('transformer-236', 'transformer_236', dfx, data_loader, range(6), range(5), '../input/robertalargesquad2')
    #eval_test('transformer_235', dfx, data_loader, range(7), range(5), '../input/robertabase')
    eval_test('transformer-237', 'transformer_237', dfx, data_loader, range(7), range(2, 5), '../input/robertabase')

    sub = eval_sub(dfx)

    sample = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    sample['selected_text'] = sub['filtered']
    sample.to_csv('submission.csv', index=False)

