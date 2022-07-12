import numpy as np
import pandas as pd
import os
import tokenizers
import string
import torch
import transformers
import torch.nn as nn
import re
import numpy as np
import tokenizers

from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

LR = 3e-5
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 50
BERT_PATH = "../input/bert-base-uncased/"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"{BERT_PATH}/vocab.txt",
    lowercase=True
)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "bert"

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self, ids, attention_mask, token_type_ids):
        # not using sentiment at all
        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # (batch_size, num_tokens, 768)
        logits = self.l0(sequence_output)
        # (batch_size, num_tokens, 2)
        # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits

class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        
        len_st = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1
        
        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[3:-1]
        # print(tok_tweet_tokens)
        # print(tok_tweet.offsets)
        # ['[CLS]', 'spent', 'the', 'entire', 'morning', 'in', 'a', 'meeting', 'w', '/', 
        # 'a', 'vendor', ',', 'and', 'my', 'boss', 'was', 'not', 'happy', 'w', '/', 'them', 
        # '.', 'lots', 'of', 'fun', '.', 'i', 'had', 'other', 'plans', 'for', 'my', 'morning', '[SEP]']
        targets = [0] * (len(tok_tweet_tokens) - 4)
        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":
            sub_minus = 8
        else:
            sub_minus = 7

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1 - sub_minus:offset2 - sub_minus]) > 0:
                targets[j] = 1
        
        targets = [0] + [0] + [0] + targets + [0]

        #print(tweet)
        #print(selected_text)
        #print([x for i, x in enumerate(tok_tweet_tokens) if targets[i] == 1])
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1
        
        #print(targets_start)
        #print(targets_end)

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

        padding_length = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([0] * padding_length)
        targets_start = targets_start + ([0] * padding_length)
        targets_end = targets_end + ([0] * padding_length)

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tweet_tokens': " ".join(tok_tweet_tokens),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_length, dtype=torch.long),
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'orig_sentiment': self.sentiment[item]
        }


def train_loop(dataloader, model, optimizer, scheduler=None, accumulation_steps=1, fp16=False, device=DEVICE, model_type="bert"):
    
    loss_fct = BCEWithLogitsLoss()
    tr_loss = 0.
    model.train()
    for step, batch in tqdm(enumerate(dataloader)):

        inputs = {
            "ids": batch["ids"].to(device),
            "attention_mask": batch["mask"].to(device),
            "token_type_ids": batch["token_type_ids"].to(device),
            # "start_positions": batch["targets_start"].to(device),
            # "end_positions": batch["targets_end"].to(device),
        }
        
        if model_type in ["xlm", "roberta", "distilbert"]:
            del inputs["token_type_ids"]
        
        start_logits, end_logits = model(**inputs)
        
        start_loss = loss_fct(start_logits, batch["targets_start"].to(device, dtype=torch.float))
        end_loss  = loss_fct(end_logits, batch["targets_end"].to(device, dtype=torch.float))
        
        loss = (start_loss + end_loss)/2.
        print(step, loss)
        
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
        
        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex to use fp16 training.")

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        tr_loss += loss.item()
        
        if (step + 1) % accumulation_steps == 0:
            
            if fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()


device = DEVICE
model = BERTBaseUncased()
model.to(DEVICE)

df_train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
df_train.dropna(inplace=True);
df_train.reset_index(drop=True, inplace=True);

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0,
)

train_ds = TweetDataset(
    tweet=df_train['text'], 
    sentiment=df_train['sentiment'], 
    selected_text=df_train['selected_text'],
)

train_dl = torch.utils.data.DataLoader(train_ds, 
                      shuffle=True, 
                      batch_size=TRAIN_BATCH_SIZE, 
                      drop_last=True
                     )

for epoch in range(5):
    train_loop(train_dl, model, optimizer)
    torch.save(model.state_dict(), f"my_model_{epoch}.bin")

df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values
test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
    )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)

all_outputs = []
fin_outputs_start = []
fin_outputs_end = []
fin_outputs_start2 = []
fin_outputs_end2 = []
fin_tweet_tokens = []
fin_padding_lens = []
fin_orig_selected = []
fin_orig_sentiment = []
fin_orig_tweet = []
fin_tweet_token_ids = []

with torch.no_grad():
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        tweet_tokens = d["tweet_tokens"]
        padding_len = d["padding_len"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_sentiment = d["orig_sentiment"]
        orig_tweet = d["orig_tweet"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        sentiment = sentiment.to(device, dtype=torch.float)

        outputs_start, outputs_end = model(
            ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())
        
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)

fin_outputs_start = np.vstack(fin_outputs_start)
fin_outputs_end = np.vstack(fin_outputs_end)

fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)
jaccards = []
threshold = 0.3
for j in range(len(fin_tweet_tokens)):
    target_string = fin_orig_selected[j]
    tweet_tokens = fin_tweet_tokens[j]
    padding_len = fin_padding_lens[j]
    original_tweet = fin_orig_tweet[j]
    sentiment_val = fin_orig_sentiment[j]

    if padding_len > 0:
        mask_start = fin_outputs_start[j, 3:-1][:-padding_len] >= threshold
        mask_end = fin_outputs_end[j, 3:-1][:-padding_len] >= threshold
        tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
    else:
        mask_start = fin_outputs_start[j, 3:-1] >= threshold
        mask_end = fin_outputs_end[j, 3:-1] >= threshold
        tweet_token_ids = fin_tweet_token_ids[j, 3:-1]

    mask = [0] * len(mask_start)
    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]
    if len(idx_start) > 0:
        idx_start = idx_start[0]
        if len(idx_end) > 0:
            idx_end = idx_end[0]
        else:
            idx_end = idx_start
    else:
        idx_start = 0
        idx_end = 0

    for mj in range(idx_start, idx_end + 1):
        try:
            mask[mj] = 1
        except Exception as e:
            print(e, mask)

    output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]

    filtered_output = TOKENIZER.decode(output_tokens)
    filtered_output = filtered_output.strip().lower()

    if sentiment_val == "neutral":
        filtered_output = original_tweet

    all_outputs.append(filtered_output.strip())

sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = all_outputs
sample.to_csv("submission.csv", index=False)