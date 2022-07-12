import sys, os, re, csv, codecs, numpy as np, pandas as pd

#=================Keras==============
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv2D, Embedding, Dropout, Activation
from keras.layers import Bidirectional, MaxPooling1D, MaxPooling2D, Reshape, Flatten, concatenate, BatchNormalization
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, backend
#=================nltk===============
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE=f'{path}glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
number_filters = 100 # the number of CNN filters

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    #Remove Special Characters
    text=special_character_removal.sub('',text)
    
    #Replace Numbers
    text=replace_numbers.sub('n',text)
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
    
comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))
    
test_comments=[]
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))
    

tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
# tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(comments + test_comments))
comments_sequence = tokenizer.texts_to_sequences(comments)
test_comments_sequence = tokenizer.texts_to_sequences(test_comments)    
X_t = pad_sequences(comments_sequence , maxlen=maxlen)
X_te = pad_sequences(test_comments_sequence, maxlen=maxlen)

X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
X_te = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
inp = Input(shape=(1, maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x1 = Conv2D(number_filters, (3, embed_size), data_format='channels_first')(x)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((int(int(x1.shape[2])  / 1.5), 1), data_format='channels_first')(x1)
x1 = Flatten()(x1)

x2 = Conv2D(number_filters, (4, embed_size), data_format='channels_first')(x)
x2 = BatchNormalization()(x2)
x2 = Activation('elu')(x2)
x2 = MaxPooling2D((int(int(x2.shape[2])  / 1.5), 1), data_format='channels_first')(x2)
x2 = Flatten()(x2)

x3 = Conv2D(number_filters, (5, embed_size), data_format='channels_first')(x)
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling2D((int(int(x3.shape[2])  / 1.5), 1), data_format='channels_first')(x3)
x3 = Flatten()(x3)

x4 = Conv2D(number_filters, (6, embed_size), data_format='channels_first')(x)
x4 = BatchNormalization()(x4)
x4 = Activation('elu')(x4)
x4 = MaxPooling2D((int(int(x4.shape[2])  / 1.5), 1), data_format='channels_first')(x4)
x4 = Flatten()(x4)

x5 = Conv2D(number_filters, (7, embed_size), data_format='channels_first')(x)
x5 = BatchNormalization()(x5)
x5 = Activation('relu')(x5)
x5 = MaxPooling2D((int(int(x5.shape[2])  / 1.5), 1), data_format='channels_first')(x5)
x5 = Flatten()(x5)

# x6 = Conv2D(number_filters, (5, embed_size), data_format='channels_first')(x)
# x6 = BatchNormalization()(x6)
# x6 = Activation('elu')(x6)
# x6 = MaxPooling2D((int(int(x6.shape[2])  / 1.5), 1), data_format='channels_first')(x6)
# x6 = Flatten()(x6)

# x7 = Conv2D(number_filters, (6, embed_size), data_format='channels_first')(x)
# x7 = BatchNormalization()(x7)
# x7 = Activation('relu')(x7)
# x7 = MaxPooling2D((int(int(x7.shape[2])  / 1.5), 1), data_format='channels_first')(x7)
# x7 = Flatten()(x7)

# x8 = Conv2D(number_filters, (7, embed_size), data_format='channels_first')(x)
# x8 = BatchNormalization()(x8)
# x8 = Activation('elu')(x8)
# x8 = MaxPooling2D((int(int(x8.shape[2])  / 1.5), 1), data_format='channels_first')(x8)
# x8 = Flatten()(x8)

# x9 = Conv2D(number_filters, (8, embed_size), data_format='channels_first')(x)
# x9 = BatchNormalization()(x9)
# x9 = Activation('relu')(x9)
# x9 = MaxPooling2D((int(int(x9.shape[2])  / 1.5), 1), data_format='channels_first')(x9)
# x9 = Flatten()(x9)

# x10 = Conv2D(number_filters, (9, embed_size), data_format='channels_first')(x)
# x10 = BatchNormalization()(x10)
# x10 = Activation('elu')(x10)
# x10 = MaxPooling2D((int(int(x10.shape[2])  / 1.5), 1), data_format='channels_first')(x10)
# x10 = Flatten()(x10)

x = concatenate([x1, x2, x3, x4, x5])
# x = Dropout(0.1)(x)
# x = Dense(512, activation="elu")(x)
# x = Dropout(0.1)(x)
# x = Dense(256, activation="relu")(x)
# x = Dropout(0.1)(x)
# x = Dense(6, activation="softmax")(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_t, y, batch_size=1280, epochs=3)

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission_textcnn.csv', index=False)