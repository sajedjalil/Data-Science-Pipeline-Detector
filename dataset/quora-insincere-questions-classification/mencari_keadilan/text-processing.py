import pandas as pd
import time
import numpy as np
import operator 
import re, os
from tqdm import tqdm
import math
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Activation, CuDNNGRU, Bidirectional, GlobalMaxPool1D, GRU, Conv1D
from keras.models import Model
from keras import optimizers, layers, initializers, regularizers, constraints

def _get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

#print (_get_available_devices())

## Loading Data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

## Loading embeddings
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index
    
# glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
# paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

# print("Extracting GloVe embedding")
# embed_glove = load_embed(glove)
# print("Extracting Paragram embedding")
# embed_paragram = load_embed(paragram)
print("Extracting FastText embedding")
embed_fasttext = load_embed(wiki_news)

## Vocabulary and Coverage functions
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in tqdm(sentences):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
    
## Starting point
train_vocab = build_vocab(train_df['question_text'])
test_vocab = build_vocab(test_df['question_text'])

# print("Train Glove : ")
# train_oov_glove = check_coverage(train_vocab, embed_glove)
# print("Train Paragram : ")
# train_oov_paragram = check_coverage(train_vocab, embed_paragram)
print("Train FastText : ")
train_oov_fasttext = check_coverage(train_vocab, embed_fasttext)

# print("Test Glove : ")
# test_oov_glove = check_coverage(test_vocab, embed_glove)
# print("Test Paragram : ")
# test_oov_paragram = check_coverage(test_vocab, embed_paragram)
print("Test FastText : ")
test_oov_fasttext = check_coverage(test_vocab, embed_fasttext)

print ('\n')
print ('#'*50)
print ('\n')

## Paragram seems to have a significantly lower coverage
train_df['lowered_question'] = train_df['question_text'].apply(lambda x: x.lower())
test_df['lowered_question'] = test_df['question_text'].apply(lambda x: x.lower())

train_vocab_low = build_vocab(train_df['lowered_question'])
test_vocab_low = build_vocab(test_df['lowered_question'])

# print("Train Glove : ")
# train_oov_glove = check_coverage(train_vocab_low, embed_glove)
# print("Train Paragram : ")
# train_oov_paragram = check_coverage(train_vocab_low, embed_paragram)
print("Train FastText : ")
train_oov_fasttext = check_coverage(train_vocab_low, embed_fasttext)

# print("Test Glove : ")
# test_oov_glove = check_coverage(test_vocab_low, embed_glove)
# print("Test Paragram : ")
# test_oov_paragram = check_coverage(test_vocab_low, embed_paragram)
print("Test FastText : ")
test_oov_fasttext = check_coverage(test_vocab_low, embed_fasttext)

print ('\n')
print ('#'*50)
print ('\n')

## word.lower() takes the embedding of word if word.lower() doesn't have an embedding
def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")
    
# print("Train Glove : ")
# add_lower(embed_glove, train_vocab)
# print("Train Paragram : ")
# add_lower(embed_paragram, train_vocab)
print("Train FastText : ")
add_lower(embed_fasttext, train_vocab)

# print("Test Glove : ")
# add_lower(embed_glove, test_vocab)
# print("Test Paragram : ")
# add_lower(embed_paragram, test_vocab)
print("Test FastText : ")
add_lower(embed_fasttext, test_vocab)

print ('\n')
print ('#'*50)
print ('\n')

# print("Train Glove : ")
# train_oov_glove = check_coverage(train_vocab_low, embed_glove)
# print("Train Paragram : ")
# train_oov_paragram = check_coverage(train_vocab_low, embed_paragram)
print("Train FastText : ")
train_oov_fasttext = check_coverage(train_vocab_low, embed_fasttext)

# print("Test Glove : ")
# test_oov_glove = check_coverage(test_vocab_low, embed_glove)
# print("Test Paragram : ")
# test_oov_paragram = check_coverage(test_vocab_low, embed_paragram)
print("Test FastText : ")
test_oov_fasttext = check_coverage(test_vocab_low, embed_fasttext)

print ('\n')
print ('#'*50)
print ('\n')

print (test_oov_fasttext[:10])

print ('#'*127)

print (test_oov_fasttext[:10])

##First faults appearing are :
#*Contractions
#*Words with punctuation in them
def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

print("- Known Contractions -")
# print("   Glove :")
# print(known_contractions(embed_glove))
# print("   Paragram :")
# print(known_contractions(embed_paragram))
print("   FastText :")
print(known_contractions(embed_fasttext))

def clean_contractions(text, mapping):
    specials = ["â€™", "â€˜", "Â´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
    
train_df['treated_question'] = train_df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
test_df['treated_question'] = test_df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

train_vocab = build_vocab(train_df['treated_question'])
test_vocab = build_vocab(test_df['treated_question'])

# print("Train Glove : ")
# train_oov_glove = check_coverage(train_vocab, embed_glove)
# print("Train Paragram : ")
# train_oov_paragram = check_coverage(train_vocab, embed_paragram)
print("Train FastText : ")
train_oov_fasttext = check_coverage(train_vocab, embed_fasttext)

# print("Test Glove : ")
# test_oov_glove = check_coverage(test_vocab, embed_glove)
# print("Test Paragram : ")
# test_oov_paragram = check_coverage(test_vocab, embed_paragram)
print("Test FastText : ")
test_oov_fasttext = check_coverage(test_vocab, embed_fasttext)

print ('\n')
print ('#'*50)
print ('\n')

## Now, let us deal with special characters
def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown
    
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""â€œâ€â€™' + 'âˆžÎ¸Ã·Î±â€¢Ã âˆ’Î²âˆ…Â³Ï€â€˜â‚¹Â´Â°Â£â‚¬\Ã—â„¢âˆšÂ²â€”â€“&'

# print("Glove :")
# print(unknown_punct(embed_glove, punct))
# print("Paragram :")
# print(unknown_punct(embed_paragram, punct))
print("FastText :")
print(unknown_punct(embed_fasttext, punct))

punct_mapping = {"â€˜": "'", "â‚¹": "e", "Â´": "'", "Â°": "", "â‚¬": "e", "â„¢": "tm", "âˆš": " sqrt ", "Ã—": "x", "Â²": "2", "â€”": "-", "â€“": "-", "â€™": "'", "_": "-", "`": "'", 'â€œ': '"', 'â€': '"', 'â€œ': '"', "Â£": "e", 'âˆž': 'infinity', 'Î¸': 'theta', 'Ã·': '/', 'Î±': 'alpha', 'â€¢': '.', 'Ã ': 'a', 'âˆ’': '-', 'Î²': 'beta', 'âˆ…': '', 'Â³': '3', 'Ï€': 'pi', }

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', 'â€¦': ' ... ', '\ufeff': '', 'à¤•à¤°à¤¨à¤¾': '', 'à¤¹à¥ˆ': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

train_vocab = build_vocab(train_df['treated_question'])
test_vocab = build_vocab(test_df['treated_question'])

# print("Train Glove : ")
# train_oov_glove = check_coverage(train_vocab, embed_glove)
# print("Train Paragram : ")
# train_oov_paragram = check_coverage(train_vocab, embed_paragram)
print("Train FastText : ")
train_oov_fasttext = check_coverage(train_vocab, embed_fasttext)

# print("Test Glove : ")
# test_oov_glove = check_coverage(test_vocab, embed_glove)
# print("Test Paragram : ")
# test_oov_paragram = check_coverage(test_vocab, embed_paragram)
print("Test FastText : ")
test_oov_fasttext = check_coverage(test_vocab, embed_fasttext)

print ('\n') 
print ('#'*50)
print ('\n')

print (train_oov_fasttext[:100])

print ('#'*50)

print (test_oov_fasttext[:100])

print ('\n') 
print ('#'*50)
print ('\n')

## Spelling mistakes
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x
    
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
test_df['treated_question'] = test_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

train_vocab = build_vocab(train_df['treated_question'])
test_vocab = build_vocab(test_df['treated_question'])

# print("Train Glove : ")
# train_oov_glove = check_coverage(train_vocab, embed_glove)
# print("Train Paragram : ")
# train_oov_paragram = check_coverage(train_vocab, embed_paragram)
print("Train FastText : ")
train_oov_fasttext = check_coverage(train_vocab, embed_fasttext)

# print("Test Glove : ")
# test_oov_glove = check_coverage(test_vocab, embed_glove)
# print("Test Paragram : ")
# test_oov_paragram = check_coverage(test_vocab, embed_paragram)
print("Test FastText : ")
test_oov_fasttext = check_coverage(test_vocab, embed_fasttext)

print ('\n') 
print ('#'*50)
print ("Text Preprocessing Done")
print ('#'*50)
print ('\n')

train_df_clean = train_df
test_df_clean = test_df

## split to train and val
train_df, val_df = train_test_split(train_df_clean, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 75000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use

## fill up the missing values
train_X = train_df["treated_question"].fillna("_na_").values
val_X = val_df["treated_question"].fillna("_na_").values
test_X = test_df_clean["treated_question"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

# inp = Input(shape=(maxlen,))
# x = Embedding(max_features, embed_size)(inp)
# x = Bidirectional(GRU(64, return_sequences=True))(x)
# x = GlobalMaxPool1D()(x)
# x = Dense(16, activation="relu")(x)
# x = Dropout(0.1)(x)
# x = Dense(1, activation="sigmoid")(x)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## print(model.summary())

## Train the model 
# model.fit(train_X, train_y, batch_size=2018, epochs=2, validation_data=(val_X, val_y))

# pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)
# for thresh in np.arange(0.1, 0.501, 0.01):
#     thresh = np.round(thresh, 2)
#     print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))

# pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)

# del model, inp, x
# import gc; gc.collect()
# time.sleep(10)

## Wiki News FastText Embeddings:
print ('\n') 
print ('#'*50)
print ("Load File Wiki News FastText Embeddings")
## EMBEDDING_FILE = '/home/fahmi/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
## embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(wiki_news) if len(o)>100)

all_embs = np.stack(embed_fasttext.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embed_fasttext.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print ('\n') 
print ('#'*50)
print ("Fit Proses Wiki News FastText Embeddings")
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))

pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)

# del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
# import gc; gc.collect()

pred_test_y = (0.33*pred_fasttext_test_y)/0.33
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df_clean["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)