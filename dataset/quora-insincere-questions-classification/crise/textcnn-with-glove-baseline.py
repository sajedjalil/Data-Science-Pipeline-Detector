import numpy as np, pandas as pd
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Conv1D, Embedding, Dropout, MaxPooling1D, concatenate
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import gc
import os
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def preprocessing(train_texts, train_labels, test_texts, maxlen=70):
    
    tokenizer = Tokenizer(num_words=95000)
    tokenizer.fit_on_texts(train_texts)
    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    word_index = tokenizer.word_index
    x_train = pad_sequences(x_train_seq, maxlen)
    x_test = pad_sequences(x_test_seq, maxlen)
    y_train = to_categorical(train_labels, num_classes=2)
    
    return x_train, y_train, x_test, word_index


def glove(word_index, embedding_dim=300, maxlen=70):
    GLOVE_DIR = '/kaggle/input/embeddings/glove.840B.300d/'
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    logger.info('Total %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    logger.info('Length of embedding_matrix:%s' % embedding_matrix.shape[0])
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                mask_zero=False,
                                input_length=maxlen,
                                trainable=False)
    del embedding_matrix
    gc.collect()
    return embedding_layer


def text_cnn(embedding_layer, maxlen=70, max_features=95000, embed_size=32):
    
    comment_seq = Input(shape=[maxlen], name='x_seq')
    emb_comment = embedding_layer(comment_seq)
    
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    
    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)
    
    output = Dense(units=2, activation='softmax')(output)

    model = Model(comment_seq, output)
    
    optimizer = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy', f1])
    model.summary()
    return model

def clean_text(texts):
    cleaned_text = []
    for text in texts:
        cleaned_text.append(text.split(' '))
    return cleaned_text
    
if __name__ == '__main__':
    
    logger.info('start to run')
    train = pd.read_csv('/kaggle/input/train.csv') # 1,306,122
    # train.head()
    # train.info()
    test = pd.read_csv('/kaggle/input/test.csv') # 56,370
    # test.head()
    # test.info()
    train_texts, train_labels = clean_text(train['question_text']), train['target']
    
    qid, test_texts = test['qid'], clean_text(test['question_text'])
    logger.info('complete load datasets')
    
    x_train, y_train, x_test, word_index = preprocessing(train_texts, train_labels, test_texts)
    del train_texts, train_labels, test_texts
    gc.collect()
    logger.info('complete formate data')
    
    embedding_layer = glove(word_index)
    
    model = text_cnn(embedding_layer)
    batch_size = 512
    epochs = 50
    
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
    
    callbacks = [
        earlystopping, 
        lr_reduction
    ]
    
    logger.info('start to train the model')
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              shuffle=True)
              
    logger.info('start to test')
    preds = model.predict(x_test)
    
    labs = np.argmax(preds,axis=1)
    test_sub = pd.DataFrame(labs)
    test_sub.columns=['prediction']
    test_sub['qid'] = qid
    test_sub[['qid', 'prediction']].to_csv('/kaggle/working/submission.csv',index=None)
    
    logger.info('complete predict')
    
