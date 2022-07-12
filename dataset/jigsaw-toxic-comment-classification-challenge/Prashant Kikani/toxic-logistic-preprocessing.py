import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import pandas as pd
import re
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, EarlyStopping
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
EMBEDDING_FILE = '../input/glove6b50d/glove.6B.50d.txt'

# PREPROCESSING PART. REMOVE STOP WORDS, ANYTHING OTHER THAN ALPHABETS.

repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data
print("crap removed")
trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
print("only alphabets")

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

'''
stop_words = set(stopwords.words('english'))
new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j in stop_words:
            continue
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j in stop_words:
            continue
        xx += j + " "
    new_test_data.append(xx)
train["comment_text"] = new_train_data
test["comment_text"] = new_test_data
print('removed stopwords')
'''
alldata = train["comment_text"].tolist() + test["comment_text"].tolist()
lente = len(train["comment_text"].tolist())

tokenizer = Tokenizer(nb_words=100)
tokenizer.fit_on_texts(train["comment_text"].values)
trseq = tokenizer.texts_to_sequences(train["comment_text"].values)
teseq = tokenizer.texts_to_sequences(test["comment_text"].values)
wi = tokenizer.word_index
print('Found %s unique tokens.' % len(wi))
tradata = pad_sequences(trseq, maxlen=100)
tesdata = pad_sequences(teseq, maxlen=100)
print('pad complete')


'''
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=20000)
word_vectorizer.fit(all_text)

print('A')
train_word_features = word_vectorizer.transform(train_text)
print('B')
test_word_features = word_vectorizer.transform(test_text)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=20000)
char_vectorizer.fit(all_text)
print('AA')
train_char_features = char_vectorizer.transform(train_text)
print('BB')
test_char_features = char_vectorizer.transform(test_text)


train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

losses = []
predictions = {'id': test['id']}
for class_name in list_classes:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_loss = np.mean(cross_val_score(classifier, tradata, y, cv=3, scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(tradata, y)
    predictions[class_name] = expit(logit(classifier.predict_proba(test_features)[:, 1]) - 0.5)
    print("completed ", class_name)

print('Total CV score is {}'.format(np.mean(losses)))

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('logi_sub.csv', index=False)
'''


embeddings_index = {}
f = open('../input/glove6b50d/glove.6B.50d.txt', 'r', encoding="utf8")
for line in f.readlines():
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

dataaemat = np.zeros((len(wi) + 1, 50))
for word, i in wi.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        dataaemat[i] = embedding_vector
        
embedding_layer = Embedding(len(wi) + 1, 50, weights=[dataaemat], input_length=125, trainable=False)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 125 # max number of words in a comment to use
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

from keras.layers import Input, Dense, Dropout, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization, GRU
from keras.models import Sequential, load_model, Model

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

inp = Input(shape=(100,))
x = Embedding(20000, 50, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(GRU(80, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
import keras.backend as K
def loss(y_true, y_pred):
     return K.binary_crossentropy(y_true, y_pred)
model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])
print('model complied')

def schedule(ind):
    a = [0.002, 0.003, 0.000]
    return a[ind]
lr = callbacks.LearningRateScheduler(schedule)
[X_train, X_val, y_train, y_val] = train_test_split(tradata, y, train_size=0.95)

ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
model.fit(X_train, y_train, batch_size=256, epochs=5, validation_data=(X_val, y_val), callbacks=[lr, ra_val, early])
#model.fit(tradata, y, batch_size=128, epochs=1, validation_split=0.1, verbose = 1)
ans = model.predict([tesdata], batch_size=1024, verbose=1)
sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
sample_submission[list_classes] = ans
sample_submission.to_csv('gru_sub.csv', index=False)



'''
myble = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')
ble = myble.copy()
col = myble.columns

for c, i in enumerate(col):
    ble[i] = (ans[c] + 2 * myble[i]) / 3
ble.to_csv('ble_myble_gru.csv', index = False)
'''