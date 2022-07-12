import os
import timeit
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from textblob import TextBlob
from functools import partial
from IPython.display import display

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers import Conv1D
from keras.layers import SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import concatenate
from keras.layers import Dense, Flatten, Dropout
from keras.layers import BatchNormalization, PReLU
from keras.layers import MaxPooling1D, add
from keras.layers import Layer
from keras import optimizers
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.callbacks import Callback
import gc



# ========================================================================
# PREPROCESSING
# ========================================================================

def read_text_embeddings(path):

    print('read embeddings %s' % path)
    
    # for .vec format should skip first row because it contains meta info
    format = path.split('.')[-1]
    skiprows = 1 if format == 'vec' else 0

    embedding_index = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for i in range(skiprows):
            f.readline()
        # for line in tqdm(f):
        for line in f:
            values = line.split(' ')
            # values = line.rstrip().rsplit(' ')
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embedding_index[word] = embedding

    print('done.')

    return embedding_index
    

def read_gensim_embeddings(path):

    print('read embeddings %s' % path)
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    embedding_index = {word: word_vectors['word'] for word in word_vectors.index2entity}
    print('done.')

    return embedding_index
    

def preprocess(train_text, test_text, max_features, max_len,
               fit_on_test=False, oov='normal', truncating='post',
               lower=True, tokenizer=None, keras_tokenizer=None,
               embedding_index=None):

    if tokenizer=='nltk':
        tok = NltkRoutines(tokenizer=True, lemmatizer=False, progress_bar=False)
        train_text = tok.transform(train_text)
        test_text = tok.transform(test_text)
    elif tokenizer=='textblob':
        tok = TextBlobRoutines(tokenizer=True, lemmatizer=False, progress_bar=False)
        train_text = tok.transform(train_text)
        test_text = tok.transform(test_text)

    X, Xtest, tokenizer = tokenize_and_pad(train_text, test_text, max_features, max_len,
                     fit_on_test=fit_on_test, truncating=truncating, lower=lower,
                     tokenizer=keras_tokenizer)

    if embedding_index is not None:
        embedding_matrix = get_embedding_matrix(embedding_index, tokenizer.word_index,
                                            max_features=max_features, oov=oov)
        return X, Xtest, embedding_matrix
    else:
        return X, Xtest, tokenizer


def tokenize_and_pad(train_text, test_text, max_features, max_len,
                     fit_on_test=False, truncating='post', lower=True,
                     char_level=False,  tokenizer=None):

    print('tokenize_and_pad')
    start_time = timeit.default_timer()

    print('tokenizing...')
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_features, lower=lower,
                              char_level=char_level)

    if fit_on_test:
        tokenizer.fit_on_texts(list(train_text)+list(test_text))
    else:
        tokenizer.fit_on_texts(list(train_text))

    X = tokenizer.texts_to_sequences(train_text)
    Xtest = tokenizer.texts_to_sequences(test_text)

    print('padding...')
    X = sequence.pad_sequences(X, maxlen=max_len, padding=truncating, truncating=truncating)
    Xtest = sequence.pad_sequences(Xtest, maxlen=max_len, padding=truncating, truncating=truncating)

    print('time elapsed %.1f' % (timeit.default_timer()-start_time))

    return X, Xtest, tokenizer


def get_embedding_matrix(embedding_index, word_index, max_features=None, oov='normal'):

    print('prepare embedding_matrix')

    embedding_size = len(list(embedding_index.values())[0])

    if max_features is not None:
        nb_words = min(max_features, len(word_index))
    else:
        nb_words = len(word_index)+1

    if oov=='normal':
        embeddings = np.stack(embedding_index.values())
        emb_mean, emb_std = embeddings.mean(), embeddings.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
    elif oov=='zero':
        embedding_matrix = np.zeros((nb_words, embedding_size))

    oov_count = 0
    for word, i in word_index.items():
        if max_features is not None and i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        embedding_vector_capitalized = embedding_index.get(word.capitalize())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif embedding_vector_capitalized is not None:
            embedding_matrix[i] = embedding_vector_capitalized
        else:
            oov_count += 1

    print('Out-of-vocabulary count: %d' % oov_count)

    return embedding_matrix


class TextBlobRoutines():


    def __init__(self, tokenizer=True, lemmatizer=False, progress_bar=False):

        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.progress_bar = progress_bar

    def textblob_tokenizer(self, text):
        if self.progress_bar:
            tqdm.pandas(desc='textblob tokenizer')
            return text.progress_map(lambda x: ' '.join(list(TextBlob(x).words)))
        else:
            return text.map(lambda x: ' '.join(list(TextBlob(x).words)))

    def textblob_lemmatizer(self, text):
        if self.progress_bar:
            tqdm.pandas(desc='textblob lemmatizer')
            return text.progress_map(lambda x: ' '.join([w.lemmatize() for w in TextBlob(x).words]))
        else:
            return text.map(lambda x: ' '.join([w.lemmatize() for w in TextBlob(x).words]))

    def transform(self, text):

        if self.tokenizer:
            text = self.textblob_tokenizer(text)
        if self.lemmatizer:
            text = self.textblob_lemmatizer(text)

        return text


# ========================================================================
# TRAINING
# ========================================================================


def run_train(nn_model, X, y, Xtest, num_epochs=20, batch_size=512, patience=2,
            random_state=42, verbose=1, clr_params=None, sgdr_params=None):

    pred_test = []

    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    tmp_file = 'tmp/weights %s.h5' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    w_train = None

    earlyStopping = EarlyStopping(monitor='val_loss', patience=patience,
                                  verbose=1, mode='auto')
    if clr_params is not None:
        lr_policy = CyclicLRScheduler(**clr_params)
        callbacks = [earlyStopping, lr_policy]
    elif sgdr_params is not None:
        lr_policy = SGDRScheduler(**sgdr_params)
        callbacks = [earlyStopping, lr_policy]
    else:
        checkpointer = ModelCheckpoint(filepath=tmp_file, verbose=0,
                            save_best_only=True, save_weights_only=True)
        callbacks = [earlyStopping, checkpointer]

    model = nn_model()
    model.fit([X], [y], validation_data=None,
                      epochs=num_epochs, batch_size=batch_size, sample_weight=w_train,
                      callbacks=callbacks, verbose=verbose)

    if clr_params is not None or sgdr_params is not None:

        for i, weights in enumerate(lr_policy.weights):
            model.set_weights(weights)
            pred_test.append(model.predict([Xtest], batch_size=2048, verbose=0).squeeze())
        pred_test = pred_test[1:]

    else:

        model.load_weights(tmp_file)
        pred_test.append(model.predict([Xtest], batch_size=512, verbose=0).squeeze())

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    return pred_test


def calculate_metrics(y, pred):

    metrics = {}
    metrics['logloss'] = log_loss(y, pred)
    metrics['roc_auc'] = roc_auc_score(y, pred)
    f1_search = f1_threshold_search(y, pred)
    metrics['threshold'], metrics['f1_score'] = f1_search['threshold'], f1_search['f1_score']
    return metrics


def f1_threshold_search(y_true, y_proba, plot=False):

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)  # not equal length of precision/recall and threshold
    f1_score = 2 / (1/precision + 1/recall)
    best_score = np.max(f1_score)
    best_threshold = thresholds[np.argmax(f1_score)]
    if plot:
        plt.plot(thresholds, f1_score, '-b')
        plt.plot([best_threshold], [best_score], '*r')
        plt.show()
    return {'threshold': best_threshold , 'f1_score': best_score}


def f1_expectation_approx(p):
    p = p.sort_values(ascending=False)
    F1 = 2*np.cumsum(p)/(np.sum(p) + np.arange(1,len(p)+1))
    return F1

def f1_expectation_max_approx(p):
    p = p.sort_values(ascending=False)
    F1 = f1_expectation_approx(p)
    F1_max = F1.max()
    threshold = p.iloc[np.argmax(F1.values)]
    return F1_max, threshold
    
    
class CyclicLRScheduler(Callback):

    def __init__(self, iteration_type='batch', min_lr=0.001, max_lr=0.01,
                 step_size=1000, num_cycles=5, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):

        self.iteration_type = iteration_type
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.num_cycles = num_cycles
        self.mode = mode
        self.gamma = gamma

        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.iteration = 0
        self.history = {}
        self.weights = []
        self.weight_iterations = []

    def on_train_begin(self, logs={}):

        if self.iteration == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, batch, logs=None):

        if self.iteration_type == 'batch':
            self._update(logs)

    def on_epoch_end(self, epoch, logs=None):

        if self.iteration_type == 'epoch':
            self._update(logs)

    def _update(self, logs):

        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iteration', []).append(self.iteration)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.iteration % (self.step_size*2) == 0:
            self.weights.append(self.model.get_weights())
            self.weight_iterations.append(self.iteration)

        if self.iteration >= self.num_cycles*self.step_size*2:
            self.model.stop_training = True

        K.set_value(self.model.optimizer.lr, self.clr())

    def clr(self):

        cycle = np.floor(1+self.iteration/(2*self.step_size))
        x = np.abs(self.iteration/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.min_lr + (self.max_lr-self.min_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.min_lr + (self.max_lr-self.min_lr)*np.maximum(0, (1-x))*self.scale_fn(self.iteration)


# ========================================================================
# MODELS
# ========================================================================


def CNN_model(embedding_matrix, max_len, num_filters, kernel_sizes,
              spatial_dropout=0, dense_units=0, trainable_embedding=False):

    inp = Input(shape=(max_len,), dtype='int32')
    x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix], input_length=max_len,
                  mask_zero=False, trainable=trainable_embedding)(inp)
    if spatial_dropout > 0:
        x = SpatialDropout1D(spatial_dropout)(x)
    conv = []
    for size in kernel_sizes:
        conv0 = Conv1D(filters=num_filters, kernel_size=size, activation='relu')(x)
        conv0 = GlobalMaxPooling1D()(conv0)
        conv.append(conv0)
    concat = concatenate(conv)
    if dense_units > 0:
        concat = Dense(dense_units, activation='relu')(concat)
    outp = Dense(1, activation="sigmoid")(concat)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def RNN_model(embedding_matrix, max_len, rnn_type='GRU', num_cells=50,
              dense_units=0, spatial_dropout=0, trainable_embedding=False):

    rnn_layer = get_rnn_layer(rnn_type)

    inp = Input(shape=(max_len,), dtype='int32')
    x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix], input_length=max_len,
                  mask_zero=False, trainable=trainable_embedding)(inp)
    if spatial_dropout > 0:
        x = SpatialDropout1D(spatial_dropout)(x)
    x = Bidirectional(rnn_layer(num_cells, return_sequences=True))(x)
    x = Bidirectional(rnn_layer(num_cells, return_sequences=False))(x)
    if dense_units > 0:
        x = Dense(dense_units, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

    
def RNN_GMP_model(embedding_matrix, max_len, rnn_type='GRU', num_cells=50,
              dense_units=0, spatial_dropout=0, avg_pool=False,
              trainable_embedding=False, embed_dim_reduction=None,
              optimizer=optimizers.Adam, lr=1e-3,
              clipnorm=-1, loss='binary_crossentropy'):

    rnn_layer = get_rnn_layer(rnn_type)

    inp = Input(shape=(max_len,), dtype='int32')
    x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix], input_length=max_len,
                  mask_zero=False, trainable=trainable_embedding)(inp)
    if embed_dim_reduction is not None:
        x = Conv1D(filters=embed_dim_reduction, kernel_size=1, activation='linear')(x)
    if spatial_dropout > 0:
        x = SpatialDropout1D(spatial_dropout)(x)
    x = Bidirectional(rnn_layer(num_cells, return_sequences=True))(x)
    if rnn_type == 'GRU':
        x, state1, state2 = Bidirectional(rnn_layer(num_cells, return_sequences=True, return_state=True))(x)
    elif rnn_type == 'LSTM':
        x, state1, state_c1, state2, state_c2 = Bidirectional(rnn_layer(num_cells, return_sequences=True, return_state=True))(x)
    max_pool = GlobalMaxPooling1D()(x)
    if avg_pool:
        avg_pool = GlobalAveragePooling1D()(x)
        concat = concatenate([max_pool, avg_pool, state1, state2])
    else:
        concat = concatenate([max_pool, state1, state2])
    if dense_units > 0:
        concat = Dense(dense_units, activation='relu')(concat)
    outp = Dense(1, activation="sigmoid")(concat)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=loss, optimizer=optimizer(lr=lr, clipnorm=clipnorm))

    return model
    
    
def get_rnn_layer(rnn_type):
    if rnn_type == 'GRU':
        return CuDNNGRU
    elif rnn_type == 'LSTM':
        return CuDNNLSTM
        
        
def Dense_model(input_shape):
    
    inp = Input(shape=(input_shape,), dtype='float32', sparse=True)
    x = Dense(256, activation='relu')(inp)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model 
        
        

# ======================================================================================
# ======================================================================================
# ======================================================================================


def prepare_embeddings(embeddings):
    
    embedding_matriсes = {}
    for embedding in embeddings:
        if embedding == 'google':
            embedding_index = read_gensim_embeddings(EMBEDDING_PATH[embedding])
        else:
            embedding_index = read_text_embeddings(EMBEDDING_PATH[embedding])
        embedding_matriсes[embedding] = get_embedding_matrix(embedding_index, tokenizer.word_index,
                                                max_features=config['preprocess_params']['max_features'], 
                                                oov=config['preprocess_params']['oov'])
        
    embedding_matriсes['all'] = np.concatenate([embedding_matriсes['glove'],
                                            embedding_matriсes['wiki'],
                                            embedding_matriсes['paragram']], axis=1)
    embedding_matriсes['glove_wiki'] = np.concatenate([embedding_matriсes['glove'],
                                            embedding_matriсes['wiki']], axis=1)
    embedding_matriсes['glove_paragram'] = np.concatenate([embedding_matriсes['glove'],
                                            embedding_matriсes['paragram']], axis=1)
    embedding_matriсes['wiki_paragram'] = np.concatenate([embedding_matriсes['wiki'],
                                            embedding_matriсes['paragram']], axis=1)
    
    return embedding_matriсes
    
    
def run(embedding):
    model_function = globals()[config['model_name']]
    nn_model = partial(model_function, embedding_matrix=embedding_matriсes[embedding],
                       max_len=max_len, **config['model_params'])
    pred_test = run_train(nn_model, X, y, Xtest, **config['train_params'], 
                                         clr_params=config['clr_params'])
    return pred_test
    
    
# settings

EMBEDDING_PATH = {'glove': '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
                  'paragram': '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt',
                  'wiki': '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
                  'google': '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
                  }

config = {}
config['preprocess_params'] = {'max_features': 100000,
                     'max_len': 50,
                     'truncating': 'post',
                     'fit_on_test': True,
                     'oov': 'normal',
                     'lower': True,
                     'tokenizer': 'textblob'}
max_len = config['preprocess_params']['max_len']
config['train_params'] = {'num_epochs': 20,
                'batch_size': 1024,
                'patience': 3,
                'random_state': 42,
                'verbose': 2}
config['clr_params'] = {'iteration_type': 'batch',
                        'min_lr': 1e-3,
                        'max_lr': 4e-3,
                        'step_size': 500,
                        'num_cycles': 4}
                        

# read data  

train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)
y = train.target

# stage 2 check
# test = pd.concat([test]*7)


# tfidf char model

print('start tfidf')
vectorizer = TfidfVectorizer(max_features=50000, analyzer='char', ngram_range=(3,4), 
                              sublinear_tf=True, strip_accents='unicode', min_df=10)
vectorizer.fit(train.question_text)
X_tfidf = vectorizer.transform(train.question_text)
Xtest_tfidf = vectorizer.transform(test.question_text)
print('tfidf calculated')

model_function = Dense_model
config['clr_params'] = {'iteration_type': 'batch',
                        'min_lr': 5e-4,
                        'max_lr': 2e-3,
                        'step_size': 250,
                        'num_cycles': 4}
nn_model = partial(model_function, input_shape=X_tfidf.shape[1])
pred_test_tfidf = run_train(nn_model, X_tfidf, y, Xtest_tfidf, **config['train_params'], 
                                     clr_params=config['clr_params'])
pred_test_tfidf = np.array(pred_test_tfidf).mean(axis=0)



# Deep learning models

X, Xtest, tokenizer = preprocess(train.question_text, test.question_text,
                embedding_index=None, **config['preprocess_params'])
embeddings = ['glove', 'wiki', 'paragram']
embedding_matriсes = prepare_embeddings(embeddings)
gc.collect()                                   
   
   
config['clr_params'] = {'iteration_type': 'batch',
                        'min_lr': 1e-3,
                        'max_lr': 4e-3,
                        'step_size': 500,
                        'num_cycles': 4}
                        
config['model_name'] = 'RNN_GMP_model'
config['model_params'] = {'rnn_type': 'LSTM',
                'num_cells': 128,
                'spatial_dropout': 0,
                'dense_units': 0,
                'avg_pool': False,
                'trainable_embedding': False,
                'clipnorm': 1
}
pred_test1 = run(embedding='glove_wiki')
gc.collect()

config['model_name'] = 'RNN_GMP_model'
config['model_params'] = {'rnn_type': 'GRU',
                'num_cells': 128,
                'spatial_dropout': 0,
                'dense_units': 0,
                'avg_pool': False,
                'trainable_embedding': False,
                'clipnorm': 1
}
pred_test2 = run(embedding='all')
gc.collect()

config['model_name'] = 'RNN_GMP_model'
config['model_params'] = {'rnn_type': 'GRU',
                'num_cells': 128,
                'spatial_dropout': 0,
                'dense_units': 0,
                'avg_pool': False,
                'trainable_embedding': False,
                'clipnorm': 1
}
pred_test3 = run(embedding='glove_wiki')
gc.collect()

config['model_name'] = 'RNN_GMP_model'
config['model_params'] = {'rnn_type': 'GRU',
                'num_cells': 128,
                'spatial_dropout': 0,
                'dense_units': 0,
                'avg_pool': False,
                'trainable_embedding': False,
                'clipnorm': 1
}
pred_test4 = run(embedding='wiki_paragram')
gc.collect()

# config['model_name'] = 'CNN_model'
# config['model_params'] = 	{'num_filters': 128, 
#                              'kernel_sizes': [1, 2, 3, 4, 5], 
#                              'spatial_dropout': 0, 
#                              'dense_units': 128, 
#                              'trainable_embedding': False}
# pred_test5 = run(embedding='glove_wiki')
# gc.collect()


pred_test = np.array(pred_test1+pred_test2+pred_test3+pred_test4).mean(axis=0)
# pred_test = np.array(pred_test1+pred_test2+pred_test3+pred_test4+pred_test5).mean(axis=0)
# pred_test = np.array(pred_test1+pred_test2+pred_test4+pred_test5).mean(axis=0)


import gc
del embedding_matriсes
del X, Xtest, tokenizer
gc.collect()

                                         

# final blend

pred_test = pred_test*0.85 + pred_test_tfidf*0.15


# threshold choice and submission

F1_max, threshold = f1_expectation_max_approx(pd.Series(pred_test))
print('threshold', threshold)
print('F1 expectation max', F1_max)

submission = pd.Series(pred_test, 
                       index=test.index, name='prediction').reset_index()
submission['prediction'] = (submission['prediction'] > threshold).astype(int)                 
submission.to_csv("submission.csv", index=False)


