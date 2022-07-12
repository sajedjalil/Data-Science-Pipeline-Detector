# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')


EMBEDDING_FILE = '../input/glove840b/glove.840B.300d.txt'

train_data["comment_text"].fillna("fillna")
test_data["comment_text"].fillna("fillna")
input_train = train_data["comment_text"].str.lower()
output_train = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

input_test = test_data["comment_text"].str.lower()

max_features=100000
maxlen=150
embed_size=300

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


tok=text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(list(input_train)+list(input_test))
input_train=tok.texts_to_sequences(input_train)
input_test=tok.texts_to_sequences(input_test)
x_train=sequence.pad_sequences(input_train,maxlen=maxlen)
x_test=sequence.pad_sequences(input_test,maxlen=maxlen)


embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs



word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



sequence_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
x = SpatialDropout1D(0.35)(x)
x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.15,recurrent_dropout=0.15))(x)
x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
#x = Dense(128, activation='relu')(x)
# x = Dropout(0.1)(x)
preds = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])

batch_size = 128
epochs = 4
X_tra, X_val, y_tra, y_val = train_test_split(x_train, output_train, train_size=0.9, random_state=233)


# filepath="../input/best-model/best.hdf5"
filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
callbacks_list = [ra_val,checkpoint, early]


model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
#Loading model weights
model.load_weights(filepath)
print('Predicting....')
y_pred = model.predict(x_test,batch_size=1024,verbose=1)

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission_2lstm.csv', index=False)