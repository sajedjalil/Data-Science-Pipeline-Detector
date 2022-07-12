print("import libraries")
from gensim.models.word2vec import Word2Vec
import pandas as pd, numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
import re

print("import data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("setup data cleaning")

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
chars = '([1234567890!@#$%^&*_+-=,./<>?;:"[][}]"\'\\|“”¨«»®´·º½¾¿¡§£₤''])'
#create tokenizer to split up words
print('data loaded')
re_tok = re.compile(r'([1234567890!@#$%^&*_+-=,./<>?;:"[][}]"\'\\|“”¨«»®´·º½¾¿¡§£₤''])')
def tokenize(s):
    s = s.lower()
    s = re.sub(r"what's", "what is ", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r"i'm", "i am ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"\'ll", " will ", s)
    s = re.sub(r"\'scuse", " excuse ", s)
    s = re.sub('\W', ' ', s)
    s = re.sub('\s+', ' ', s)
    for c in chars:
        s = s.replace(c, ' ')
    return re_tok.sub(r' \1 ', s).split()


print("cleaning data")

data = []

for sent in train[COMMENT]:
    data.append(tokenize(sent))


for sent in test[COMMENT]:
    data.append(tokenize(sent))


print("setting text quantification parameters")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 100
embed_size = 100 # how big is each word vector

y = train[list_classes].values



print("quantifying text with word2vec")
model = Word2Vec(data, size=embed_size, window=5, min_count=5, workers=10)


print("putting word2vec into lists")
print("\ttraining set")
list_tokenized_train = []
for sent in train[COMMENT]:
    c = 0
    app = []
    for word in tokenize(sent):
        if(c == maxlen):
            break
        try:
            app.append(model[word])
            c += 1
        except:
            pass
    list_tokenized_train.append(app)



print("\ttest set")
list_tokenized_test = []
for sent in test[COMMENT]:
    c = 0
    app = []
    for word in tokenize(sent):
        if(c == maxlen):
            break
        try:
            app.append(model[word])
            c += 1
        except:
            pass
    list_tokenized_test.append(app)



print('padding short sequences')
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)



print('defining neural net parameters')
inp = Input(shape=X_t.shape[1:])
x = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.2))(inp)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(80, return_sequences=True, recurrent_dropout=0.2))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.2))(x)
x = Dropout(0.2)(x)
x = GlobalMaxPool1D()(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("fitting")
def fold(i, n, data):
    if(i == 0):
        return data[int(len(X_t)/n):]
    elif(i == (n-1)):
        return data[:-int(len(X_t)/n)]
    else:
        return np.concatenate((data[:int(len(X_t)*i/n)], data[int(len(X_t)*(1+i)/n):]), axis=0)


num_folds = 3
for z in range(2):
    for x in range(num_folds):
        model.fit(fold(x, num_folds, X_t), fold(x, num_folds, y), batch_size=64, epochs=1, validation_split=0.1)



print('done fitting, predicting')
y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('./predictions.csv', index=False)
