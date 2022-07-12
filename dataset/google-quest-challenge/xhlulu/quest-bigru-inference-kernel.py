"""
Training kernel: https://www.kaggle.com/xhlulu/quest-bigru-starter-code
"""
import json
import pickle

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm_notebook as tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def compute_sequences(cols, tokenizer, maxlens):
    sequences = []
    
    for texts, maxlen in zip(cols, maxlens):
        seq = tokenizer.texts_to_sequences(texts.values)
        seq = pad_sequences(seq, maxlen=maxlen)
        sequences.append(seq)
    
    return sequences


# Load Data
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")

# Update train and test host to only use domain
train.host = train.host.apply(lambda h: h.split(".")[-2])
test.host = test.host.apply(lambda h: h.split(".")[-2])
# If it's an unknown domain, force it to something known
test.loc[~test.host.isin(train.host.unique()), 'host'] = 'stackexchange'

# Load Tokenizer, Model, and encoders
model = load_model('/kaggle/input/quest-bi-gru-starter-code/model.h5')
le_cat = joblib.load('/kaggle/input/quest-bi-gru-starter-code/category_encoder.joblib')
le_host = joblib.load('/kaggle/input/quest-bi-gru-starter-code/host_encoder.joblib')
with open('/kaggle/input/quest-bi-gru-starter-code/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Encode category and host data
test_cat_enc = le_cat.transform(test.category) + 1
test_host_enc = le_host.transform(test.host) + 1

# Build test set
test_data = compute_sequences(
    [test.question_title, test.question_body, test.answer], 
    tokenizer,
    [30, 300, 300]
)
test_data += [test_cat_enc, test_host_enc]

# Run model
test_pred = model.predict(test_data, batch_size=256)

# Submit results
submission.loc[:, 'question_asker_intent_understanding':] = test_pred
submission.to_csv('submission.csv', index=False)