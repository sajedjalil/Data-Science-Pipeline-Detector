# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
from tqdm import tqdm
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os


EMBEDDING_FILE = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
TRAIN_CSV = '../input/avito-demand-prediction/train.csv'
TEST_CSV = '../input/avito-demand-prediction/test.csv'

max_features = 100000
maxlen = 100
embed_size = 300

train = pd.read_csv(TRAIN_CSV, index_col = 0)
labels = train[['deal_probability']].copy()
train = train[['description']].copy()

tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')

train['description'] = train['description'].astype(str)
tokenizer.fit_on_texts(list(train['description'].fillna('NA').values))


print('getting embeddings')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del embeddings_index
X_train, X_valid, y_train, y_valid = train_test_split(train['description'].values, labels['deal_probability'].values, test_size = 0.1, random_state = 23)
del train
print('convert to sequences')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)

print('padding')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, embed_size, weights = [embedding_matrix],
                    input_length = maxlen, trainable = False)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(CuDNNGRU(32,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.2)(main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error',
                  metrics =[root_mean_squared_error])
    model.summary()
    return model

EPOCHS = 4

model = build_model()
file_path = "model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
history = model.fit(X_train, y_train, batch_size = 256, epochs = EPOCHS, validation_data = (X_valid, y_valid),
                verbose = 1, callbacks = [check_point])
model.load_weights(file_path)
prediction = model.predict(X_valid)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, prediction)))


test = pd.read_csv(TEST_CSV, index_col = 0)
test = test[['description']].copy()

test['description'] = test['description'].astype(str)
X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
prediction = model.predict(X_test,batch_size = 128, verbose = 1)

sample_submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv', index_col = 0)
submission = sample_submission.copy()
submission['deal_probability'] = prediction
submission.to_csv('submission.csv')



