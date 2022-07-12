from pathlib import Path
import numpy as np
import pandas as pd
import os, gc, time
import tensorflow as tf
from tensorflow import keras
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from gensim import models
import random
import time

# Any results you write to the current directory are saved as output.

RANDOM_STATE = 42
TRAIN_DATA_PATH = '../input/quora-insincere-questions-classification/train.csv'
TEST_DATA_PATH = '../input/quora-insincere-questions-classification/test.csv'

GLOVE_PATH = Path('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')
PARAGRAM_PATH = Path('../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
WIKI_NEWS_PATH = Path('../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
GOOGLE_NEWS_PATH = Path('../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin')

LABEL_COLUMNS = ['target']

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 
                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                       "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
                       "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 
                       "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", 
                       "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
                       "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", 
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' # for adding space before and after in words containing punct

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", 
                 "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', 
                 "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', 
                 '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', } # for glove

mispell_mapping = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 
                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 
                'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 
                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', 
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', 
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 
                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 
                'pokémon': 'pokemon'}

def clean_special_chars(text, punct=punct, mapping=punct_mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ') # f is formatted string literal sifn
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters
    for s in specials:
        text = text.replace(s, specials[s])
    return text

def correct_spelling(text, mapping=mispell_mapping):
    for word in mapping.keys():
        text = text.replace(word, mapping[word])
    return text

def load_train_data(file_path, train_column, label_columns):
    print('Loading train data...')
    data_frame = pd.read_csv(file_path)
    return data_frame[train_column].tolist(), data_frame[label_columns].values

def load_test_data(file_path, train_column):
    print('Loading test data...')
    data_frame = pd.read_csv(file_path)
    return data_frame[train_column].tolist(), data_frame['qid']

def clean_contractions(text, mapping=contraction_mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text_new = []
    for t in text.split(' '):
        if t.lower() in mapping:
            t_new = mapping[t.lower()]
            if t.isupper():
                t_new = t_new.upper()
            elif t[0].isupper() and len(t_new) > 1:
                t_new = t_new[0].upper() + t_new[1:]
            text_new.append(t_new)
        else:
            text_new.append(t)
    return ' '.join(text_new)

def index_word_embeddings(embeddings_path, embedding_dim, skip_first_line=False):
    print('Indexing word embeddings from', str(embeddings_path), '...')
    embeddings_index = {}
    skipped = False
    with open(embeddings_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if skip_first_line and not skipped:
                skipped = True
                continue
            splitted = line.split(' ')
            phrase_len = len(splitted)-embedding_dim
            coefs = ' '.join(splitted[phrase_len:len(splitted)])
            phrase_parts = splitted[0:phrase_len]
            if set(phrase_parts) == set(['']):
                phrase_parts.append('')
            word = ' '.join(phrase_parts)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            assert coefs.size == embedding_dim
            embeddings_index[word] = coefs
    print('Found', len(embeddings_index), 'word vectors.')
    return embeddings_index

def index_keyed_embeddings(keyed_vectors):
    print('Indexing embeddings from keyed vectors...')
    embedding_index = {}
    for word, vector in zip(keyed_vectors.index2word, keyed_vectors.vectors):
        embedding_index[word] = vector
    return embedding_index

# creates Tokenizer instance and fits it
def fit_tokenizer(X_texts, hparams):
    print('Fitting tokenizer...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=hparams['max_words'], filters=hparams['tokenizer_filters'], 
                                                      lower=hparams['tokenizer_lower'], split=hparams['tokenizer_split'], 
                                                      char_level=hparams['tokenizer_char_level'], oov_token=hparams['tokenizer_oov_token'])
    tokenizer.fit_on_texts(X_texts)
    return tokenizer

def prepare_embedding_matrix(max_words, embedding_dim, word_index, embeddings_index, hparams, lower_only=False):
    print('Preparing embedding matrix...')
    np.random.seed(hparams['random_state'])
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    count = 0
    embedding_matrix = np.zeros((max_words, embedding_dim))
    random_vector = np.random.random(embedding_dim)
    for word, i in word_index.items():
        if i >= max_words:
            continue
        
        if word in embeddings_index and word.lower() not in embeddings_index:
            embeddings_index[word.lower()] = embeddings_index[word]

        embedding_vector = embeddings_index.get(word.lower()) if lower_only else embeddings_index.get(word)

        # https://www.kaggle.com/wowfattie/3rd-place
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.upper())

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.capitalize())

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(porter.stem(word)) 

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(lancaster.stem(word)) 

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(snowball.stem(word)) 

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(lemmatizer.lemmatize(word))
         

        if word == hparams['tokenizer_oov_token'] or embedding_vector is None:
            embedding_matrix[i] = random_vector
        else:    
            embedding_matrix[i] = embedding_vector
            count += 1
        
    print('Word vectors coverage:', count / max_words)
    print('Embedding matrix shape:', embedding_matrix.shape)
    return embedding_matrix


def create_padded_sequences(X_texts, tokenizer, hparams):
    print('Converting texts to sequences...')
    X_sequences = tokenizer.texts_to_sequences(X_texts)
    print('Padding sequences...')
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_sequences, maxlen=hparams['max_length'], padding=hparams['padding'], truncating=hparams['truncating'])
    return X_padded

#https://www.tensorflow.org/tutorials/text/nmt_with_attention#write_the_encoder_and_decoder_model
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # values shape == (batch_size, max_length, hidden size)

        # hidden (query?) shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score (with broadcasting)
        hidden_with_time_axis = tf.expand_dims(query, 1)
       
        W1_out = self.W1(values) # (batch_size, max_length, units)
        W2_out = self.W2(hidden_with_time_axis) # (batch_size, 1, units)
        
        # the shape of the tensor (W_out_sum) before applying self.V is (batch_size, max_length, units)
        W_out_sum = W1_out + W2_out # (..., 1, units) broadcasted to (..., max_length, units)
        
        W_out_sum_tanh = tf.nn.tanh(W_out_sum)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        score = self.V(W_out_sum_tanh)

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values # (batch_size, max_length, hidden_size)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

def build_model(hparams, embedding_matrix=None):
    print('Building model...')
    np.random.seed(hparams['random_state'])
    random.seed(hparams['random_state'])
    tf.random.set_seed(hparams['random_state'])


    input_ = keras.layers.Input(shape=(hparams['max_length'],))

    if embedding_matrix is not None:
        x = keras.layers.Embedding(input_dim=hparams['max_words'], output_dim=hparams['emb_out_dim'], 
                                   embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False)(input_)
    else:
        x = keras.layers.Embedding(input_dim=hparams['max_words'], output_dim=hparams['emb_out_dim'])(input_)

    x = keras.layers.SpatialDropout1D(rate=hparams['dropout_rate'])(x)

    x, x_h_state_1_1, _, x_h_state_1_2, _ = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, return_state=True))(x) # x.shape = None, 256, len
    avg_1 = keras.layers.GlobalAveragePooling1D()(x)
    max_1 = keras.layers.GlobalMaxPooling1D()(x)
    x_ctx_1_1, _ = BahdanauAttention(256)(x_h_state_1_1, x[:,:,0:256])
    x_ctx_1_2, _ = BahdanauAttention(256)(x_h_state_1_2, x[:,:,256:])

    x_seq_2_1, x_state_2_1 = keras.layers.GRU(128, return_sequences=True, return_state=True)(x)
    avg_2_1 = keras.layers.GlobalAveragePooling1D()(x_seq_2_1)
    max_2_1 = keras.layers.GlobalMaxPooling1D()(x_seq_2_1)
    x_ctx_2_1, _ = BahdanauAttention(128)(x_state_2_1, x_seq_2_1)
    
    x_seq_2_2, x_state_2_2 = keras.layers.GRU(128, return_sequences=True, return_state=True, go_backwards=True)(x)
    avg_2_2 = keras.layers.GlobalAveragePooling1D()(x_seq_2_2)
    max_2_2 = keras.layers.GlobalMaxPooling1D()(x_seq_2_2)
    x_ctx_2_2, _ = BahdanauAttention(128)(x_state_2_2, x_seq_2_2)
    
    c_out = keras.layers.concatenate([avg_1, 
                                      max_1,
                                      x_ctx_1_1,
                                      x_ctx_1_2,

                                      avg_2_1,
                                      max_2_1,
                                      x_ctx_2_1,
                                      
                                      avg_2_2,
                                      max_2_2,
                                      x_ctx_2_2,
                                    ])

    output = keras.layers.Dense(1, activation='sigmoid')(c_out)

    model = keras.Model(inputs=[input_], outputs=[output])

    model.compile(optimizer=hparams['optimizer'], loss='binary_crossentropy', metrics=[])
    print('Parameters:', model.count_params())
    return model

class CustomAveraging(tf.keras.callbacks.Callback):
    def __init__(self, save_epoch):
        self.save_epoch = save_epoch - 1
        self.saved_weights = None

    def on_epoch_end(self, epoch, logs={}):
        if epoch == self.save_epoch:
            self.saved_weights = self.model.get_weights()
            print('Model weights saved.')

        if epoch == (self.save_epoch + 1):
            model_weights = self.model.get_weights()
            new_weights = [(model_weights[i] + self.saved_weights[i]) / 2 for i in range(len(model_weights))]
            self.model.set_weights(new_weights)
            print('Model weights averaged.')

# main
start = time.time()
X_train_texts, y_train = load_train_data(file_path=TRAIN_DATA_PATH, 
                                         train_column='question_text', 
                                         label_columns=LABEL_COLUMNS)

X_test_texts, qids = load_test_data(TEST_DATA_PATH, train_column='question_text')

hparams = {
    'max_words': None, # for Tokenizer
    'max_length': 40, 
    'batch_size': 512,
    'emb_out_dim': 600, 
    'dropout_rate': 0.4,
    'epochs': 4,
    'optimizer': 'nadam',
    'tokenizer_filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    'tokenizer_lower': False,
    'tokenizer_split': " ",
    'tokenizer_char_level': False,
    'padding': 'post',
    'truncating': 'post',
    'tokenizer_oov_token': '<UNK>', # not a real hyperparameter
    'n_classes': y_train.shape[1], # not a real hyperparameter
    'random_state': RANDOM_STATE # not a real hyperparameter
}


print('Cleaning contractions...')
X_train_texts = [clean_contractions(text) for text in X_train_texts]
X_test_texts = [clean_contractions(text) for text in X_test_texts]

print('Cleaning special characters...')
X_train_texts = [clean_special_chars(text) for text in X_train_texts]
X_test_texts = [clean_special_chars(text) for text in X_test_texts]

print('Correcting spelling...')
X_train_texts = [correct_spelling(text) for text in X_train_texts]
X_test_texts = [correct_spelling(text) for text in X_test_texts]

keyed_vectors_google = models.KeyedVectors.load_word2vec_format(GOOGLE_NEWS_PATH, binary=True) #index_word_embeddings(GOOGLE_NEWS_PATH, 300, True)
embeddings_index_google = index_keyed_embeddings(keyed_vectors_google)

print('Deleting keyed vectors...')
del keyed_vectors_google
gc.collect()

embeddings_index_wiki = index_word_embeddings(WIKI_NEWS_PATH, 300, True)

tokenizer = fit_tokenizer(X_train_texts + X_test_texts, hparams)
VOCAB_SIZE = len(tokenizer.word_index)
if hparams['max_words'] is None:
    hparams['max_words'] = VOCAB_SIZE + 1 # 1 for padding value 0
else:
    hparams['max_words'] += 1
print('Found', VOCAB_SIZE, 'unique train tokens.')
print('MAX WORDS:', hparams['max_words'])

embedding_matrix_wiki = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                 tokenizer.word_index, embeddings_index_wiki, hparams)

embedding_matrix_google = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                   tokenizer.word_index, embeddings_index_google, hparams)
    
embedding_matrix_2 = 0.7 * embedding_matrix_wiki + 0.3 * embedding_matrix_google

print('Deleting used matrices...')
del embeddings_index_wiki, embeddings_index_google, embedding_matrix_wiki, embedding_matrix_google
gc.collect()
    
embeddings_index_glove = index_word_embeddings(GLOVE_PATH, 300)
embeddings_index_paragram = index_word_embeddings(PARAGRAM_PATH, 300)
    
embedding_matrix_glove = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                  tokenizer.word_index, embeddings_index_glove, hparams)

embedding_matrix_paragram = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                     tokenizer.word_index, embeddings_index_paragram, hparams, lower_only=True)
    
embedding_matrix_1 = 0.7 * embedding_matrix_glove + 0.3 * embedding_matrix_paragram

print('Deleting used matrices...')
del embeddings_index_glove, embeddings_index_paragram, embedding_matrix_glove, embedding_matrix_paragram
gc.collect()

embedding_matrix_final = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis=1)

X_train_padded = create_padded_sequences(X_train_texts, tokenizer, hparams)
X_test_padded = create_padded_sequences(X_test_texts, tokenizer, hparams)

model = build_model(hparams, embedding_matrix_final)

model.fit(X_train_padded, y_train, epochs=hparams['epochs'], batch_size=hparams['batch_size'], shuffle=True, 
          callbacks=[CustomAveraging(3)], verbose=2)

print('\nPredicting...')
y_pred = np.squeeze(model.predict(X_test_padded, batch_size=4096))
y_pred = (y_pred >= 0.3780).astype(int)

submission = pd.DataFrame({'qid': qids})
submission['prediction'] = y_pred
submission.to_csv('submission.csv', index=False)

print('done')