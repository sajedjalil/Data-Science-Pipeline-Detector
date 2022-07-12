# References
# This Kernel is heavily borrowing from several Kernels:
# [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold  (most of it)
# [2] https://www.kaggle.com/artgor/cnn-in-keras-on-folds (preprocessing)
# [3] https://www.kaggle.com/taindow/simple-cudnngru-python-keras
# [4] https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480 (logger)
# [5] https://www.kaggle.com/thousandvoices/simple-lstm/ (embeddings)
# [6] https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution/ (pipline)
# [7] https://www.kaggle.com/thousandvoices/simple-lstm (pipline)

##
import logging
import datetime
import os
import gc
import warnings
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from multiprocessing import Pool, cpu_count

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from keras import layers as L
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.losses import binary_crossentropy

COMMENT_TEXT_COL = 'comment_text'
EMB_MAX_FEAT = 300
MAX_LEN = 220
MAX_FEATURES = 100000
BATCH_SIZE = 512
NUM_EPOCHS = 2
LSTM_UNITS = 64
GRU_UNITS_1 = 256
GRU_UNITS_2 = 128
DENSE_HIDDEN_UNITS_1 = 512
DENSE_HIDDEN_UNITS_2 = 128
NFOLDS = 4

# adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
PUNCT_MAPPING = {"_": " ", "`": " "}
PUNCT = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'


def get_logger():
    display_format = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=display_format)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()


def clean_special_chars(text, punct=PUNCT, mapping=PUNCT_MAPPING):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    return text


def perform_preprocessing(df):
    n_cores = cpu_count()
    logger.info('data preprocessing {} with {}'.format(df.shape, n_cores))

    data = df[COMMENT_TEXT_COL].astype(str).tolist()
    p = Pool(n_cores)
    df[COMMENT_TEXT_COL] = list(p.map(clean_special_chars, data))
    p.close()
    p.join()
    return df


def run_proc_and_tokenizer(train, test):
    logger.info('Running processing')
    
    '''
    credits go to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/ 
    '''
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim',
        'black', 'white', 'psychiatric_or_mental_illness']

    train_target_pos = (train['target'].values >= 0.5).astype(bool).astype(np.int)
    train_target_neg = (train['target'].values < 0.5).astype(bool).astype(np.int)

    train_identity_pos = (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)
    train_identity_neg = (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)

    # Overall
    weights = np.ones((len(train), )) / 4

    # Subgroup
    weights += train_identity_pos / 4

    # Background Positive, Subgroup Negative
    weights += ((train_target_pos + train_identity_neg) > 1 ).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += ((train_target_neg + train_identity_pos) > 1).astype(bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()
    
    #y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
    y_train = (train['target'].values >= 0.5).astype(np.int)
    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
    
    logger.info('Fitting tokenizer')
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
    word_index = tokenizer.word_index
    
    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))
    X_train = pad_sequences(X_train, maxlen=MAX_LEN)

    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))    
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    
    with open('temporary.pickle', mode='wb') as f:
        pickle.dump(X_test, f) # use temporary file to reduce memory
        del identity_columns, tokenizer, train, test
        gc.collect()
    
    return X_train, y_train, y_aux_train, word_index, weights, loss_weight
                                                                                                                    

class Embeddings():
    def __init__(self, emb_paths=EMB_PATHS, emb_max_feat=EMB_MAX_FEAT):
        self.emb_paths = emb_paths
        self.emb_max_feat = emb_max_feat

    @staticmethod
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(self, path):
        with open(path) as f:
            return dict(self.get_coefs(*line.strip().split(' ')) for line in f)

    def build_embedding_matrix(self, word_index, path):
        embedding_index = self.load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, self.emb_max_feat))

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

    def build(self, word_index):
        logger.info('Load and build embeddings')
        embedding_matrix = np.concatenate(
            [self.build_embedding_matrix(word_index, f) for f in self.emb_paths], axis=-1)

        return embedding_matrix


# https://www.kaggle.com/ezietsman/simple-keras-model-with-data-generator
class DataGenerator():
    def __init__(self, X, y, y_aux=None, w=None, random_state=None):
        self.X = X
        self.y = y
        self.y_aux = y_aux
        self.w = w
        self.rng = random_state

    def gen_data(self, batch_size=16, ):
        while True:

            queue = np.arange(0, self.X.shape[0], 1)
            self.rng.shuffle(queue)

            for i in range(0, len(queue), batch_size):

                idx = queue[i: i + batch_size]
                im = self.X[idx]
                label = self.y[idx]

                ret = list()
                ret.append(im)
                
                if self.y_aux is not None:
                    y_aux = self.y_aux[idx, :]
                    ret.append([label, y_aux])

                else:
                    ret.append(label)
                    
                if self.w is not None:
                    w = self.w[idx]
                    
                    if self.y_aux is None:
                        ret.append(w)
                    else:
                        ret.append([w, w])

                yield tuple(ret)


class Attention(Layer):
    def __init__(
            self, step_dim,
            W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):

        self.supports_masking = True

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = None
        super(Attention, self).__init__(**kwargs)

        self.param_W = {
            'initializer': initializers.get('glorot_uniform'),
            'name': '{}_W'.format(self.name),
            'regularizer': regularizers.get(W_regularizer),
            'constraint': constraints.get(W_constraint)
        }
        self.W = None

        self.param_b = {
            'initializer': 'zero',
            'name': '{}_b'.format(self.name),
            'regularizer': regularizers.get(b_regularizer),
            'constraint': constraints.get(b_constraint)
        }
        self.b = None

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]
        self.W = self.add_weight((input_shape[-1],), **self.param_W)

        if self.bias:
            self.b = self.add_weight((input_shape[1],), **self.param_b)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        step_dim = self.step_dim
        features_dim = self.features_dim

        eij = K.reshape(
            K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
            (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    
def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:, 0], (-1, 1)), y_pred) * y_true[:, 1]


def build_model(embedding_matrix, num_aux_targets, loss_weight, max_len=MAX_LEN):
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    words = Input(shape=(max_len,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(GRU_UNITS_1, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(GRU_UNITS_2, return_sequences=True))(x)

    att = Attention(max_len)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x) 
    x = concatenate([att, avg_pool, max_pool])
    
    hidden = Dense(DENSE_HIDDEN_UNITS_1, activation='relu')(x)
    hidden = Dense(DENSE_HIDDEN_UNITS_2, activation='relu')(hidden)
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    #model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def run_model(X_train, y_train, y_aux_train, embedding_matrix, word_index, weights, loss_weight):
    # model.summary()
    logger.info('Prepare folds')
    folds = StratifiedKFold(n_splits=NFOLDS, random_state=42)
    oof_preds = np.zeros((X_train.shape[0]))
    sub_preds = None

    logger.info('Run model')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

        K.clear_session()
        file_weights = 'mod_{:02d}.hdf5'.format(fold_)
        check_point = ModelCheckpoint(file_weights, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1, cooldown=0, min_lr=0.0001)
        # lr_sched = LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

        model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)

        train_gen = DataGenerator(
            X_train[trn_idx], y_train[trn_idx], y_aux_train[trn_idx], weights[trn_idx],
            random_state=np.random.RandomState(seed=42))
        valid_gen = DataGenerator(
            X_train[val_idx], y_train[val_idx], y_aux_train[val_idx], weights[val_idx],
            random_state=np.random.RandomState(seed=42))
        
        model.fit_generator(train_gen.gen_data(batch_size=BATCH_SIZE),
                            steps_per_epoch= len(trn_idx) // BATCH_SIZE,
                            epochs=NUM_EPOCHS,
                            verbose=0,
                            callbacks=[early_stopping, check_point, reduce_lr],
                            validation_data=valid_gen.gen_data(batch_size=BATCH_SIZE),
                            validation_steps=len(val_idx) // BATCH_SIZE,
                            workers=cpu_count(),
                            use_multiprocessing=True,)

        model.load_weights(file_weights)
        oof_preds[val_idx] += model.predict(X_train[val_idx])[0][:, 0]
        with open('temporary.pickle', mode='rb') as f:
            X_test = pickle.load(f) # use temporary file to reduce memory
            if sub_preds is None:
                sub_preds = np.zeros((X_test.shape[0]))
            
            sub_preds += model.predict(X_test)[0][:, 0]
            del X_test
            gc.collect()

    sub_preds /= folds.n_splits
    print(roc_auc_score(y_train, oof_preds))
    logger.info('Complete run model')
    return sub_preds


def submit(sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv(os.path.join(JIGSAW_PATH, 'sample_submission.csv'), index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)


def main():
    logger.info('Load train data')
    train = pd.read_csv(os.path.join(JIGSAW_PATH, 'train.csv'), index_col='id')
    train = perform_preprocessing(train)

    logger.info('Load test data')
    test = pd.read_csv(os.path.join(JIGSAW_PATH, 'test.csv'), index_col='id')
    test = perform_preprocessing(test)

    train_x, train_y, train_aux_y, word_index, weights, loss_weight = run_proc_and_tokenizer(train, test)
    
    embedding_matrix = Embeddings().build(word_index)
    sub_preds = run_model(train_x, train_y, train_aux_y, embedding_matrix, word_index, weights, loss_weight)
    submit(sub_preds)


if "__main__" == __name__:
    main()
