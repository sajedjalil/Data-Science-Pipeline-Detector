"""
Inference Kernel: https://www.kaggle.com/xhlulu/quest-bigru-inference-kernel

Changelog:
    * V6 [LB 0.294]: 2-layer BiGRU, trained for 50 epochs
    * V12 [LB 0.276]: Embed category and host (50 dims respectively), 3-layer BiGRU
    * V14: Return to 2-layer BiGRU, increase dimensions, add avg pooling, 
"""

import os
import json
import gc
import pickle

import fasttext
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Input, Embedding,
    GRU, Dense, Bidirectional, 
    GlobalMaxPooling1D, GlobalAveragePooling1D,
    SpatialDropout1D, Dropout, BatchNormalization,
    Reshape, Concatenate
)


def compute_sequences(cols, tokenizer, maxlens):
    sequences = []
    
    for texts, maxlen in zip(cols, maxlens):
        seq = tokenizer.texts_to_sequences(texts.values)
        seq = pad_sequences(seq, maxlen=maxlen)
        sequences.append(seq)
    
    return sequences


def build_embedding_matrix(tokenizer, path):
    num_words = len(tokenizer.word_index) + 1
    
    embedding_matrix = np.zeros((num_words, 300))
    ft_model = fasttext.load_model(path)

    for word, i in tokenizer.word_index.items():
        embedding_matrix[i] = ft_model.get_word_vector(word)
    
    return embedding_matrix


def spearman_table(true_labels, y_pred):    
    table = []
    
    for i, col in enumerate(true_labels.columns):
        corr = spearmanr(true_labels[col], y_pred[:, i]).correlation
        table.append({'column': col, 'correlation': corr})
    
    return pd.DataFrame(table)


def rnn_block(x_in, rnn_layers, embedding, block_name, add_fc, hidden_dim=128):
    x = embedding(x_in)
    x = SpatialDropout1D(0.2, name=f'{block_name}_sp_drop')(x)
    
    for rnn in rnn_layers:
        x = rnn(x)
    
    max_pool = GlobalMaxPooling1D(name=f'{block_name}_max_pool')(x)
    avg_pool = GlobalAveragePooling1D(name=f'{block_name}_avg_pool')(x)
    
    x = Concatenate(name=f'{block_name}_all_pool')([max_pool, avg_pool])
    
    if add_fc:
        x = Dense(hidden_dim, activation='relu', name=f'{block_name}_fc')(x)
        x = BatchNormalization(name=f'{block_name}_bn')(x)
        x = Dropout(0.5, name=f'{block_name}_drop')(x)

    return x


def build_model(embedding_matrix, n_categories, n_hosts, output_shape, 
                n_gru_layers=2, n_fc_layers=2, share_gru=True, add_fc=True,
                gru_dim=128, hidden_dim=512):
    """
    embedding_matrix (ndarray): maps an integer to a vector. Recommend using fastText-300d.
    n_categories (int): number of total categories.
    n_hosts (int): Number of total hosts.
    n_gru_layers (int): How many layers of GRUs to use before using max and avg pooling.
    n_fc_layers (int): How many dense layers to use after concatenating pooled title/body/answer.
    share_gru (bool): Whether to use the same weights for questions and answer, or 2 separate weights.
    add_fc (bool): Whether to add a FC layer after pooling GRU outputs, but before concatenation.
    gru_dim (int): The dimension of GRU weights.
    hidden_dim (int): The dimension of the FC weights after concatenation.
    """
    text_embedding = Embedding(
        *embedding_matrix.shape, 
        weights=[embedding_matrix], 
        trainable=False, 
        name='text_embedding'
    )
    category_embedding = Embedding(
        input_dim=n_categories+1, 
        output_dim=64,
        name='category_embedding'
    )
    host_embedding = Embedding(
        input_dim=n_hosts+1,
        output_dim=64,
        name='host_embedding'
    )
    
    # inputs
    qt_in = Input(shape=(None,), name='qtitle_input')
    qb_in = Input(shape=(None,), name='qbody_input')
    a_in = Input(shape=(None,), name='answer_input')
    c_in = Input(shape=(1,), name='category_input')
    h_in = Input(shape=(1,), name='host_input')
    
    # Embed the host and category labels
    c = category_embedding(c_in)
    h = host_embedding(h_in)
    c = Reshape((64, ), name='category_reshape')(c)
    h = Reshape((64, ), name='host_reshape')(h)
    
    # Pass data through stacked bidirectional GRUs
    if share_gru:
        gru_layers = [
            Bidirectional(GRU(gru_dim, return_sequences=True), name=f'BiGRU{i}')
            for i in range(1, n_gru_layers+1)
        ]
        q_layers = gru_layers
        a_layers = gru_layers
    else:
        q_layers, a_layers = [[
            Bidirectional(GRU(gru_dim, return_sequences=True), name=f'{bname}_BiGRU{i}')
            for i in range(1, n_gru_layers+1)
        ] for bname in ['question', 'answer']]

    qt = rnn_block(qt_in, q_layers, text_embedding, block_name='qtitle', add_fc=add_fc)
    qb = rnn_block(qb_in, q_layers, text_embedding, block_name='qbody', add_fc=add_fc)
    a = rnn_block(a_in, a_layers, text_embedding, block_name='answer', add_fc=add_fc)

    
    # Concatenate pooled title/body/answer with embedded category/host
    hidden = Concatenate(name='combine_all')([qt, qb, a, c, h])
    
    # Fully-connected layers
    for _ in range(n_fc_layers):
        hidden = Dense(hidden_dim, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.5)(hidden)

    out = Dense(output_shape, activation='sigmoid', name='output')(hidden)
    
    model = Model(inputs=[qt_in, qb_in, a_in, c_in, h_in], outputs=[out])
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(lr=0.001, amsgrad=True), 
        metrics=['mae']
    )

    return model


# ############################### CODE STARTS HERE ###############################
# Load dataframes
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")

# Update train host to only use domain
train.host = train.host.apply(lambda h: h.split(".")[-2])

# Create a label encoder for category and host
le_cat = LabelEncoder()
le_host = LabelEncoder()
# Need to increment by 1 since keras embedding starts at 1
train_cat_enc = le_cat.fit_transform(train.category) + 1
train_host_enc = le_host.fit_transform(train.host) + 1


# Create and fit word tokenizer
tokenizer = Tokenizer()
text_cols = [train.question_title, train.question_body, train.answer, 
             test.question_title, test.question_body, test.answer]

for text in text_cols:
    tokenizer.fit_on_texts(text.values)

# Generate train data (tokenize and pad) and labels
train_labels = train.loc[:, 'question_asker_intent_understanding':]
train_data = compute_sequences(
    [train.question_title, train.question_body, train.answer], 
    tokenizer,
    [30, 300, 300]
)

# Add category and host to sequence data
train_data += [train_cat_enc, train_host_enc]

# Creating embedding matrix
path = ('/kaggle/input/'
        'fasttext-crawl-300d-2m-with-subword/'
        'crawl-300d-2m-subword/crawl-300d-2M-subword.bin')
embedding_matrix = build_embedding_matrix(tokenizer, path)

# Build and train model
model = build_model(
    embedding_matrix, 
    n_categories=len(le_cat.classes_),
    n_hosts=len(le_host.classes_),
    output_shape=train_labels.shape[1],
    gru_dim=128,
    share_gru=False,
    add_fc=True,
    n_gru_layers=3,
    n_fc_layers=3
)
model.summary()

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
train_history = model.fit(
    train_data, 
    train_labels.values,
    epochs=50,
    verbose=0,
    validation_split=0.2,
    callbacks=[checkpoint],
    batch_size=256
)

# Evaluate results on training set
model.load_weights('model.h5')
train_pred = model.predict(train_data, batch_size=256)
sp_df = spearman_table(train_labels, train_pred)
sp_df.to_csv(f"train_spearman_corr.csv", index=False)
print("Mean Spearman Correlation:", sp_df.correlation.mean())
print(sp_df)

# Print and save history
hist_df = pd.DataFrame(train_history.history)
hist_df.to_csv(f'history.csv')
print(hist_df)

# Save tokenizer and encoders
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

joblib.dump(le_cat, 'category_encoder.joblib')
joblib.dump(le_host, 'host_encoder.joblib')