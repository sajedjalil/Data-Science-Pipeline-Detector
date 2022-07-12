# Load libraries
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# Load Data
df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', delimiter='\t')
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', delimiter='\t')
pd.set_option('display.max_colwidth', -1)

seed = 101 
np.random.seed(seed)

X = df['Phrase']
temp = test['Phrase']
y = to_categorical(df['Sentiment'])
num_classes = df['Sentiment'].nunique()

# Spilt Train Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y,random_state=seed)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Tokenize Text
max_features = 15000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
temp = tokenizer.texts_to_sequences(temp)

max_words = 50 
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
temp = sequence.pad_sequences(temp, maxlen=max_words)
#print(X_train.shape,X_test.shape)


batch_size = 128
epochs = 7
def get_model(max_features, embed_dim, embedding_matrix):
    np.random.seed(seed)
    K.clear_session()
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1],
                       weights=[embedding_matrix]))#,trainable=False
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def get_embed_mat(EMBEDDING_FILE, max_features=20000):
    # word vectors
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    # embedding matrix
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(embeddings_index.values()) #for random init
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 
                                        (num_words, embed_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    max_features = embedding_matrix.shape[0]
    
    return max_features, embedding_matrix
    
# embedding matrix
EMBEDDING_FILE = '../input/glove6b100dtxt/glove.6B.100d.txt'
embed_dim = 100 #word vector dim
max_features, embedding_matrix = get_embed_mat(EMBEDDING_FILE)

# train the model
model = get_model(max_features, embed_dim, embedding_matrix)
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=epochs, batch_size=batch_size, verbose=2)

sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
sub['Sentiment'] =  model.predict_classes(temp, batch_size=batch_size, verbose=0)
sub.to_csv("lstm_glove.csv", index=False)