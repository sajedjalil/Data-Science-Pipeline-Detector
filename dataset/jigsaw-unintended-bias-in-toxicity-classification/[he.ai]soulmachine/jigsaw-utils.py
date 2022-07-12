import os
import random
import dask.bag as db
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, tqdm_notebook


def seed_everything(seed=73):
    '''
      Make PyTorch deterministic.
    '''    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def convert_lines(lines, max_seq_length, tokenizer):
    '''
      Converting the lines to BERT format.
      
      Copied from https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
    '''
    max_seq_length -= 2  # CLS, SEP
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(lines):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(f'longer: {longer}')
    return np.array(all_tokens)


def convert_lines_parallel(i, lines, max_seq_length, tokenizer):
    total_lines = len(lines)
    num_lines_per_thread = total_lines // os.cpu_count() + 1
    lines = lines[i * num_lines_per_thread : (i+1) * num_lines_per_thread]
    return convert_lines(lines, max_seq_length, tokenizer)


def prepare_data(df, cache_file_prefix, bert_model_name, max_seq_length, tokenizer):
    '''
      cache_file_prefix: for example, x-train, x-test
    '''
    # Make sure all comment_text values are strings
    df['comment_text'] = df['comment_text'].astype(str).fillna("DUMMY_VALUE")
    if 'target' in df.columns:
        # convert target to 0,1
        df['target']=(df['target']>=0.5).astype(np.int)

    if not os.path.exists('./cache'):
        os.mkdir('cache')
    cached_file = f'cache/{cache_file_prefix}-{bert_model_name}-len-{max_seq_length}'
    if os.path.exists(f'{cached_file}.npy'):
        print(f'Loading from cache file {cached_file}')
        X = np.load(f'{cached_file}.npy')
    else:
        print(f'Calculating {cache_file_prefix}')
        X = df["comment_text"]
        X = np.vstack(db.from_sequence(list(range(os.cpu_count()))).map(
            lambda i: convert_lines_parallel(i, X, max_seq_length, tokenizer)
        ).compute())
        np.save(cached_file, X)

    if 'target' in df.columns:
        Y = df[['target']].values
        print(f'X.shape: {X.shape}, X.dtype: {X.dtype}, Y.shape: {Y.shape}, Y.dtype: {Y.dtype}')
        assert Y.shape[1] == 1
        return X, Y
    else:
        print(f'X.shape: {X.shape}, X.dtype: {X.dtype}')
        return X
