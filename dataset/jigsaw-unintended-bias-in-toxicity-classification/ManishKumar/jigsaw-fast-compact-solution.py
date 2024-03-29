'''
# References
# This Kernel is using several Kernels:
# [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold  (initial version: in all steps)
# [2] https://www.kaggle.com/artgor/cnn-in-keras-on-folds (preprocessing)
# [3] https://www.kaggle.com/taindow/simple-cudnngru-python-keras (preprocessing)
# [4] https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480 (logger)
# [5] https://www.kaggle.com/thousandvoices/simple-lstm/ (embeddings)
# [6] https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/ (updated processing and model)
'''

import numpy as np 
import pandas as pd 
import os
import gc
import logging
import datetime
import warnings
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.layers as L
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

COMMENT_TEXT_COL = 'comment_text'
EMB_MAX_FEAT = 300
MAX_LEN = 220
MAX_FEATURES = 100000
BATCH_SIZE = 512
NUM_EPOCHS = 4
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
NUM_MODELS = 2
EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'


def get_logger():
    '''
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    '''
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_embedding_matrix(word_index, path):
    '''
     credits to: https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
    '''
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
            
    del embedding_index
    gc.collect()
    return embedding_matrix

def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]


def load_data():
    logger.info('Load train and test data')
    train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')
    test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
    return train, test

def perform_preprocessing(train, test):
    logger.info('data preprocessing')
    '''
        credits to: https://www.kaggle.com/artgor/cnn-in-keras-on-folds
        credits to: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
    '''
    specials = ["’", "‘", "´", "`"]
    punct_mapping = {"_":" ", "`":" "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    s_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
    
    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])    
        for p in punct:
            text = text.replace(p, f' {p} ')     
        return text

    def short_preprocess(text, mapping, spec):
        for s in spec:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text


    for df in [train, test]:
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].astype(str)
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].apply(lambda x: short_preprocess(x, s_mapping, specials))
    
    return train, test

def run_proc_and_tokenizer(train, test):
    logger.info('Running processing')
    
    '''
        credits go to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/ 
    '''
    identity_columns = ['asian', 'atheist',
       'bisexual', 'black', 'buddhist', 'christian', 'female',
       'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
       'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
       'muslim', 'other_disability', 'other_gender',
       'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']
       
    # Overall
    weights = np.ones((len(train),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()
    
    y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T
    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
    
    logger.info('Fitting tokenizer')
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))
    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))
    X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    
    with open('temporary.pickle', mode='wb') as f:
        pickle.dump(X_test, f) # use temporary file to reduce memory

    del identity_columns, weights, tokenizer, train, test
    gc.collect()
    
    return X_train, y_train, y_aux_train, word_index, loss_weight

def build_embeddings(word_index):
    logger.info('Load and build embeddings')
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f) for f in EMB_PATHS], axis=-1) 
    return embedding_matrix


def build_model(embedding_matrix, num_aux_targets, loss_weight):
    '''
        credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')

    return model
    

def run_model(X_train, y_train, y_aux_train, embedding_matrix, word_index, loss_weight):
    '''
        credits to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/
    '''
    logger.info('Run model')
    
    checkpoint_predictions = []
    weights = []
    for model_idx in range(NUM_MODELS):
        model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)
        for global_epoch in range(NUM_EPOCHS):
            model.fit(
                X_train, [y_train, y_aux_train],
                batch_size=BATCH_SIZE, epochs=1, verbose=1,
                callbacks=[LearningRateScheduler(lambda epoch: 1.1e-3 * (0.55 ** global_epoch))]
            )
            with open('temporary.pickle', mode='rb') as f:
                X_test = pickle.load(f) # use temporary file to reduce memory
            checkpoint_predictions.append(model.predict(X_test, batch_size=1024)[0].flatten())
            del X_test
            gc.collect()
            weights.append(2 ** global_epoch)
        del model
        gc.collect()
    
    preds = np.average(checkpoint_predictions, weights=weights, axis=0)
    return preds


def submit(sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv(os.path.join(JIGSAW_PATH,'sample_submission.csv'), index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)

def main():
    train, test = load_data()
    train, test = perform_preprocessing(train, test)
    X_train, y_train, y_aux_train, word_index, loss_weight = run_proc_and_tokenizer(train, test)
    embedding_matrix = build_embeddings(word_index)
    sub_preds = run_model(X_train, y_train, y_aux_train, embedding_matrix, word_index, loss_weight)
    submit(sub_preds)
    
if __name__ == "__main__":
    main()
    