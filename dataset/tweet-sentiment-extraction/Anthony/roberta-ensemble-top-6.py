### All credit goes to these excellent public kernels:
###
###    - Chris Deotte: https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705
###    - Sazuma: https://www.kaggle.com/shoheiazuma/tweet-sentiment-roberta-pytorch
###    - Abhishek Thakur: https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch
###    - See--: https://www.kaggle.com/seesee/faster-2x-tf-roberta
###

### Import Libraries
import pandas as pd, numpy as np
import multiprocessing
import pickle
import gc
import os
import tokenizers
from transformers import *

# Model 1 Libraries
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
import math

# Model 2 Libraries
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim


# Define Word Tokenizer
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)


#########################
### Model 1 Inference
#########################
def model_one_inference():
    
    MAX_LEN = 192
    EPOCHS = 3
    BATCH_SIZE = 32
    PAD_ID = 1
    SEED = 88888
    LABEL_SMOOTHING = 0.1
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

    
    ### Tokenize Test Data
    test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

    ct = test.shape[0]
    input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
    attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
    token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

    for k in range(test.shape[0]):

        # INPUT_IDS
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)                
        s_tok = sentiment_id[test.loc[k,'sentiment']]
        input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
        attention_mask_t[k,:len(enc.ids)+3] = 1


    ### Build RoBERTa Model
    import pickle

    def load_weights(model, weight_fn):
        with open(weight_fn, 'rb') as f:
            weights = pickle.load(f)
        model.set_weights(weights)
        return model

    def loss_fn(y_true, y_pred):
        # adjust the targets for sequence bucketing
        ll = tf.shape(y_pred)[1]
        y_true = y_true[:, :ll]
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,
            from_logits=False, label_smoothing=LABEL_SMOOTHING)
        loss = tf.reduce_mean(loss)
        return loss


    def build_model():
        ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

        lens = MAX_LEN - tf.reduce_sum(padding, -1)
        max_len = tf.reduce_max(lens)
        ids_ = ids[:, :max_len]
        att_ = att[:, :max_len]
        tok_ = tok[:, :max_len]

        config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
        bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
        x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

        x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x1 = tf.keras.layers.Conv1D(1,1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x2 = tf.keras.layers.Conv1D(1,1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss=loss_fn, optimizer=optimizer)

        # this is required as `model.predict` needs a fixed size!
        x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
        x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

        padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])
        return model, padded_model

    ### Define Evaluation Metric
    def jaccard(str1, str2): 
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        if (len(a)==0) & (len(b)==0): return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


    ### Selecting best start and end positions
    def get_best_start_end_idxs(start_logits, end_logits):
        max_len = len(start_logits)
        a = np.tile(start_logits, (max_len, 1))
        b = np.tile(end_logits, (max_len, 1))
        c = np.tril(a + b.T, k=0).T
        c[c == 0] = -1000
        return np.unravel_index(c.argmax(), c.shape)

    VER='v0'; DISPLAY=0 # USE display=1 FOR INTERACTIVE
    preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for fold,(idxT,idxV) in enumerate(skf.split(input_ids_t,test.sentiment.values)):

        print('#'*25)
        print('### FOLD %i'%(fold+1))
        print('#'*25)

        K.clear_session()
        model, padded_model = build_model()

        weight_fn = '../input/roberta-weights-2/%s-roberta-%i.h5'%(VER,fold)

        print('Loading model...')
        load_weights(model, weight_fn)

        print('Predicting Test...')
        preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
        preds_start += preds[0]/skf.n_splits
        preds_end += preds[1]/skf.n_splits
    
    
    with open('preds_start.pkl', 'wb') as start:
        pickle.dump(preds_start, start)
    with open('preds_end.pkl', 'wb') as end:
        pickle.dump(preds_end, end)
        

        
#########################
### Model 2 Inference
#########################
def model_two_inference():
    
    def seed_everything(seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    seed = 42
    seed_everything(seed)

    class TweetDataset(torch.utils.data.Dataset):
        def __init__(self, df, max_len=96):
            self.df = df
            self.max_len = max_len
            self.labeled = 'selected_text' in df
            self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab_file='../input/roberta-base/vocab.json', 
                merges_file='../input/roberta-base/merges.txt', 
                lowercase=True,
                add_prefix_space=True)

        def __getitem__(self, index):
            data = {}
            row = self.df.iloc[index]

            ids, masks, tweet, offsets = self.get_input_data(row)
            data['ids'] = ids
            data['masks'] = masks
            data['tweet'] = tweet
            data['offsets'] = offsets

            if self.labeled:
                start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
                data['start_idx'] = start_idx
                data['end_idx'] = end_idx

            return data

        def __len__(self):
            return len(self.df)

        def get_input_data(self, row):
            tweet = " " + " ".join(row.text.lower().split())
            encoding = self.tokenizer.encode(tweet)
            sentiment_id = self.tokenizer.encode(row.sentiment).ids
            ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
            offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

            pad_len = self.max_len - len(ids)
            if pad_len > 0:
                ids += [1] * pad_len
                offsets += [(0, 0)] * pad_len

            ids = torch.tensor(ids)
            masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
            offsets = torch.tensor(offsets)

            return ids, masks, tweet, offsets

        def get_target_idx(self, row, tweet, offsets):
            selected_text = " " +  " ".join(row.selected_text.lower().split())

            len_st = len(selected_text) - 1
            idx0 = None
            idx1 = None

            for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
                if " " + tweet[ind: ind+len_st] == selected_text:
                    idx0 = ind
                    idx1 = ind + len_st - 1
                    break

            char_targets = [0] * len(tweet)
            if idx0 != None and idx1 != None:
                for ct in range(idx0, idx1 + 1):
                    char_targets[ct] = 1

            target_idx = []
            for j, (offset1, offset2) in enumerate(offsets):
                if sum(char_targets[offset1: offset2]) > 0:
                    target_idx.append(j)

            start_idx = target_idx[0]
            end_idx = target_idx[-1]

            return start_idx, end_idx


    def get_test_loader(df, batch_size=32):
        loader = torch.utils.data.DataLoader(
            TweetDataset(df), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2)    
        return loader

    class TweetModel(nn.Module):
        def __init__(self):
            super(TweetModel, self).__init__()

            config = RobertaConfig.from_pretrained(
                '../input/roberta-base/config.json', output_hidden_states=True)    
            self.roberta = RobertaModel.from_pretrained(
                '../input/roberta-base/pytorch_model.bin', config=config)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(config.hidden_size, 2)
            nn.init.normal_(self.fc.weight, std=0.02)
            nn.init.normal_(self.fc.bias, 0)

        def forward(self, input_ids, attention_mask):
            _, _, hs = self.roberta(input_ids, attention_mask)

            x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
            x = torch.mean(x, 0)
            x = self.dropout(x)
            x = self.fc(x)
            start_logits, end_logits = x.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return start_logits, end_logits

    def jaccard(str1, str2): 
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        if (len(a)==0) & (len(b)==0): return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    
        
    def get_selected_text(text, start_idx, end_idx, offsets):
        selected_text = ""
        for ix in range(start_idx, end_idx + 1):
            selected_text += text[offsets[ix][0]: offsets[ix][1]]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                selected_text += " "
        return selected_text


    num_epochs = 3
    batch_size = 32
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    
    
    test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    test_df['text'] = test_df['text'].astype(str)
    test_loader = get_test_loader(test_df)
    predictions = []
    models = []
    for fold in range(skf.n_splits):
        model = TweetModel()
        model.cuda()
        model.load_state_dict(torch.load(f'../input/roberta-weights-pytorch-10fold/roberta_fold{fold+1}.pth'))
        model.eval()
        models.append(model)

    print("Models Loaded.")
    
    m2_starts = []
    m2_ends = []
    
    for data in test_loader:
        ids = data['ids'].cuda()
        masks = data['masks'].cuda()
        tweet = data['tweet']
        offsets = data['offsets'].numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)
    
        for i in range(len(ids)):
            
            m2_starts.append(start_logits[i])
            m2_ends.append(end_logits[i])
    

    with open('m2_starts.pkl', 'wb') as start2:
        pickle.dump(m2_starts, start2)
    with open('m2_ends.pkl', 'wb') as end2:
        pickle.dump(m2_ends, end2)


###########################    
# Execute Model 1 inference        
p = multiprocessing.Process(target=model_one_inference)
p.start()
p.join()

###########################
# Execute Model 2 inference        
q = multiprocessing.Process(target=model_two_inference)
q.start()
q.join()



### Get predictions for selected text indexes
with open('preds_start.pkl', 'rb') as start:
    preds_start = pickle.load(start)
with open('preds_end.pkl', 'rb') as end:
    preds_end = pickle.load(end)

with open('m2_starts.pkl', 'rb') as start2:
    preds2_start = pickle.load(start2)
with open('m2_ends.pkl', 'rb') as end2:
    preds2_end = pickle.load(end2)


# Line up indexes    
ensemble_start = []
ensemble_end = []

for i in range(len(preds_start)):
    yo = preds_start[i, 0:94]
    lo = preds2_start[i][2:96]
    yolo = yo + lo
    ensemble_start.append(yolo)
    
    ho = preds_end[i, 0:94]
    bo = preds2_end[i][2:96]
    hobo = ho + bo
    ensemble_end.append(hobo)

    
### Selecting best start and end positions
def get_best_start_end_idxs(start_logits, end_logits):
    max_len = len(start_logits)
    a = np.tile(start_logits, (max_len, 1))
    b = np.tile(end_logits, (max_len, 1))
    c = np.tril(a + b.T, k=0).T
    c[c == 0] = -1000
    return np.unravel_index(c.argmax(), c.shape)

## Combine predictions and prepare submission
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

all = []
for k in range(len(ensemble_start)):
    
    x = get_best_start_end_idxs(ensemble_start[k], ensemble_end[k])
    a = x[0]
    b = x[1]

    if a > b:
        a = b
        
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)
    st = tokenizer.decode(enc.ids[a-2:b-1])
    all.append(st)
    
    
# Post-Processing    
test['selected_text'] = all
test['selected_text'] = test['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
test['selected_text'] = test['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
test['selected_text'] = test['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)

# Submission!
test[['textID','selected_text']].to_csv('submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.head(25)