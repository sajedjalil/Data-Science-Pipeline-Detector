# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Any results you write to the current directory are saved as output.

# Import package

import os
import re
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import codecs

from string import punctuation
from collections import defaultdict
# from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Activation, LSTM, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D
import keras.backend as K


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))

# Define constants and parameters

Data_Dir = '../input/quora-question-pairs/'
Word_Vec_Dir = '../input/glove-840b-300d/'
Embedding_File = Word_Vec_Dir + 'glove.840B.300d.txt'
Train_Data_File = Data_Dir + 'train.csv'
Test_Data_File = Data_Dir + 'test.csv'
Max_Sequence_Length = 60
Max_Num_Words = 200000 # There are about 201000 unique words in training dataset, 200000 is enough for tokenization
Embedding_Dim = 300
Validation_Split_Ratio = 0.1

Num_Lstm = np.random.randint(175, 275)
Num_Dense = np.random.randint(100, 150)
Rate_Drop_Lstm = 0.15 + np.random.rand() * 0.25
Rate_Drop_Dense = 0.15 + np.random.rand() * 0.25

Lstm_Struc = 'lstm_{:d}_{:d}_{:.2f}_{:.2f}'.format(Num_Lstm, Num_Dense, Rate_Drop_Lstm, \
Rate_Drop_Dense)
print(Lstm_Struc)

act_f = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.4% share in test set

# Create word embedding dictionary from 'glove.840B.300d.txt'

print('Create word embedding dictionary')

embeddings_index = {}
f = open(Embedding_File, encoding='utf-8')

# for line in tqdm(f):
for line in f:
    values = line.split()
    # word = values[0]
    word = ''.join(values[:-300])   
    # coefs = np.asarray(values[1:], dtype='float32')
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found {} word vectors of glove.'.format(len(embeddings_index)))

# Process text in dataset
print('Processing text dataset')

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = [w for w in text if not w in stop_words]
    
    text = " ".join(text)
    
    # Remove punctuation from text
    # text = "".join([c for c in text if c not in punctuation])

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text) # It doesn't make sense to me
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

# load data and process with text_to_wordlist
train_texts_1 = [] 
train_texts_2 = []
train_labels = []

df_train = pd.read_csv(Train_Data_File, encoding='utf-8')
# df_train = df_train.sample(5000) # train data sample to test code
df_train = df_train.fillna('empty')
train_q1 = df_train.question1.values
train_q2 = df_train.question2.values
train_labels = df_train.is_duplicate.values

for text in train_q1:
    train_texts_1.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
    
for text in train_q2:
    train_texts_2.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))

'''
with open(Train_Data_File, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader) # Skip header row
    for values in reader:
        train_texts_1.append(text_to_wordlist(values[3], remove_stopwords=False, stem_words=False))
        train_texts_2.append(text_to_wordlist(values[4], remove_stopwords=False, stem_words=False))
        train_labels.append(int(values[5]))
'''
print('{} texts are found in train.csv'.format(len(train_texts_1)))

test_texts_1 = []
test_texts_2 = []
test_ids = []

df_test = pd.read_csv(Test_Data_File, encoding='utf-8')
# df_test = df_test.sample(5000) # test data sample to test code
df_test = df_test.fillna('empty')
test_q1 = df_test.question1.values
test_q2 = df_test.question2.values
test_ids = df_test.test_id.values

'''
with open(Test_Data_File, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1], remove_stopwords=False, stem_words=False))
        test_texts_2.append(text_to_wordlist(values[2], remove_stopwords=False, stem_words=False))
        test_ids.append(values[0])
'''

for text in test_q1:
    test_texts_1.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
    
for text in test_q2:
    test_texts_2.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
    
print('{} texts are found in test.csv'.format(len(test_texts_1)))

# Tokenize words in all sentences
tokenizer = Tokenizer(num_words=Max_Num_Words)
tokenizer.fit_on_texts(train_texts_1 + train_texts_2 + test_texts_1 + test_texts_2)

train_sequences_1 = tokenizer.texts_to_sequences(train_texts_1)
train_sequences_2 = tokenizer.texts_to_sequences(train_texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('{} unique tokens are found'.format(len(word_index)))

# pad all train with Max_Sequence_Length
train_data_1 = pad_sequences(train_sequences_1, maxlen=Max_Sequence_Length)
train_data_2 = pad_sequences(train_sequences_2, maxlen=Max_Sequence_Length)
# train_labels = np.array(train_labels)
print('Shape of train data tensor:', train_data_1.shape)
print('Shape of train labels tensor:', train_labels.shape)

# pad all test with Max_Sequence_Length
test_data_1 = pad_sequences(test_sequences_1, maxlen=Max_Sequence_Length)
test_data_2 = pad_sequences(test_sequences_2, maxlen=Max_Sequence_Length)
# test_ids = np.array(test_ids)
print('Shape of test data tensor:', test_data_2.shape)
print('Shape of test ids tensor:', test_ids.shape)

# leaky features

questions = pd.concat([df_train[['question1', 'question2']], \
        df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(questions.shape[0]):
        q_dict[questions.question1[i]].add(questions.question2[i])
        q_dict[questions.question2[i]].add(questions.question1[i])

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

leaks = df_train[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
test_leaks = df_test[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)

# Create embedding matrix for embedding layer
print('Preparing embedding matrix')

num_words = min(Max_Num_Words, len(word_index))+1

embedding_matrix = np.zeros((num_words, Embedding_Dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: '.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

# Train Validation split
# np.random.seed(2019)
perm = np.random.permutation(len(train_data_1))
idx_train = perm[:int(len(train_data_1)*(1-Validation_Split_Ratio))]
idx_val = perm[int(len(train_data_1)*(1-Validation_Split_Ratio)):]

data_1_train = np.vstack((train_data_1[idx_train], train_data_2[idx_train]))
data_2_train = np.vstack((train_data_2[idx_train], train_data_1[idx_train]))
leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
labels_train = np.concatenate((train_labels[idx_train], train_labels[idx_train]))

data_1_val = np.vstack((train_data_1[idx_val], train_data_2[idx_val]))
data_2_val = np.vstack((train_data_2[idx_val], train_data_1[idx_val]))
leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
labels_val = np.concatenate((train_labels[idx_val], train_labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.471544715
    weight_val[labels_val==0] = 1.309033281

# The embedding layer containing the word vectors
'''
emb_layer = Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    weights=[embedding_matrix],
    input_length=Max_Sequence_Length,
    trainable=False
)
'''

emb_layer = Embedding(
    input_dim=num_words,
    output_dim=Embedding_Dim,
    weights=[embedding_matrix],
    input_length=Max_Sequence_Length,
    trainable=False
)    


# LSTM layer

lstm_layer = LSTM(Num_Lstm, dropout=Rate_Drop_Lstm, recurrent_dropout=Rate_Drop_Lstm)

# Define inputs
seq1 = Input(shape=(Max_Sequence_Length,), dtype='int32')
seq2 = Input(shape=(Max_Sequence_Length,), dtype='int32')

# Run inputs through embedding
emb1 = emb_layer(seq1)
emb2 = emb_layer(seq2)

# Run through LSTM layers
lstm_a = lstm_layer(emb1)
# glob1a = GlobalAveragePooling1D()(lstm_a)
lstm_b = lstm_layer(emb2)
# glob1b = GlobalAveragePooling1D()(lstm_b)

magic_input = Input(shape=(leaks.shape[1],))
# magic_dense = BatchNormalization()(magic_input)
magic_dense = Dense(int(Num_Dense/2), activation=act_f)(magic_input)

merged = concatenate([lstm_a, lstm_b, magic_dense])
merged = BatchNormalization()(merged)
merged = Dropout(Rate_Drop_Dense)(merged)

merged = Dense(Num_Dense, activation=act_f)(merged)
merged = BatchNormalization()(merged)
merged = Dropout(Rate_Drop_Dense)(merged)

preds = Dense(1, activation='sigmoid')(merged)

# Add class weight
if re_weight:
    class_weight = {0: 1.309033281, 1: 0.471544715}
else:
    class_weight = None
    
# Train the model

model = Model(inputs=[seq1, seq2, magic_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

# Set early stopping (large patience should be useful)
early_stopping =EarlyStopping(monitor='val_loss', patience=10)
bst_model_path = Lstm_Struc + '.h5' 
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train, \
        validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path) # sotre model parameters in .h5 file
bst_val_score = min(hist.history['val_loss'])

# Make the submission
print('Making the submission')

preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('{:.4f}_'.format(bst_val_score)+Lstm_Struc+'_with_GloVe_Embedding.csv', index=False)