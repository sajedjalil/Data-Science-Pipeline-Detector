#from Vladimir's baseline
import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate,PReLU
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Reshape,Conv2D,MaxPool2D,Concatenate,Flatten,Dropout,Activation
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'


EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 50000
max_len = 200
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=max_len)
x_test = sequence.pad_sequences(X_test, maxlen=max_len)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

filter_sizes=[1,2,3,4]

def get_model():

    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(x)

    x = Reshape((max_len, embed_size, 1))(x)
   # attend = Attention(max_len)(x)
    
    conv_0 = Conv2D(64, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='he_normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(64, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='he_normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(64, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='he_normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(64, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='he_normal',
                                                                                    activation='elu')(x)
    conv_1_dialate0 = Conv2D(64, kernel_size=(2, embed_size), kernel_initializer='he_normal',
                     activation='elu',dilation_rate=(3, 1))(x)    
    conv_1_dialate1 = Conv2D(64, kernel_size=(2, embed_size), kernel_initializer='he_normal',
                     activation='elu',dilation_rate=(6, 1))(x) 
    conv_1_dialate2 = Conv2D(64, kernel_size=(2, embed_size), kernel_initializer='he_normal',
                     activation='elu',dilation_rate=(12, 1))(x)                                  
    conv_2_dialate0 = Conv2D(64, kernel_size=(3, embed_size), kernel_initializer='he_normal',
                     activation='elu',dilation_rate=(6, 1))(x)
    conv_2_dialate1 = Conv2D(64, kernel_size=(3, embed_size), kernel_initializer='he_normal',
                     activation='elu',dilation_rate=(12, 1))(x)
    conv_2_dialate2 = Conv2D(64, kernel_size=(3, embed_size), kernel_initializer='he_normal',
                     activation='elu',dilation_rate=(18, 1))(x)

    maxpool_0 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1))(conv_0) # 直接pool
    maxpool_1 = MaxPool2D(pool_size=(max_len - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_len - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(max_len - filter_sizes[3] + 1, 1))(conv_3)
  

    maxpool_dialate10 =MaxPool2D(pool_size=(197, 1))(conv_1_dialate0)  
    maxpool_dialate11 =MaxPool2D(pool_size=(194, 1))(conv_1_dialate1)
    maxpool_dialate12 =MaxPool2D(pool_size=(188, 1))(conv_1_dialate2)
    maxpool_dialate20 =MaxPool2D(pool_size=(188, 1))(conv_2_dialate0)
    maxpool_dialate21 =MaxPool2D(pool_size=(176, 1))(conv_2_dialate1)
    maxpool_dialate22 =MaxPool2D(pool_size=(164, 1))(conv_2_dialate2)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3]) 
    z1 = Concatenate(axis=1)([maxpool_dialate10,maxpool_dialate11,maxpool_dialate12,maxpool_dialate20,maxpool_dialate21,maxpool_dialate22])  
    z2 = Concatenate(axis=1)([z,z1])
    z = Flatten()(z2)
    z = Dropout(0.3)(z)

    z = Dense(128)(z)
    z = PReLU()(z)
    xs = Activation('sigmoid')(z) #tanh
    z= concatenate([z,xs]) 

    z = Dropout(0.25)(z)
        
    outp = Dense(6, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = get_model()

batch_size = 32
epochs = 3

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=1)


y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)