from collections import namedtuple
import logging as log
import random
import re
import sys

import gensim
import keras
import keras.layers as K
from nltk.metrics import edit_distance
import numba
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn


class Paths:
    WORD_2_VEC = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    GLOVE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    WIKI_NEWS = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    META = 'meta'
    
    
class Models:
    CNN = 0
    GRU = 1
    BOTH = 2
    

class Options:
    LOG_LEVEL = log.INFO
    TRAIN_CAP = None  # set a number for a quicker debug
    WORD_EMBEDDING = Paths.META
    MODEL_TYPE = Models.BOTH
    OPTIMIZER = keras.optimizers.Adam(lr=0.001, amsgrad=True)
    NUM_EPOCHS = 3
    NUM_MINI_EPOCHS = 5
    ENABLE_PUNCT = True
    ENABLE_NO_MATH = True
    ENABLE_OOV_INFERENCE = True
    OOV_THRESHOLD = 4
    OOV_INFERENCE_EPOCHS = 20
    OOV_INFERENCE_OPTIMIZER = keras.optimizers.Adam(lr=0.002, amsgrad=True)
    OOV_INFERENCE_LOSS = keras.losses.mean_squared_error  # e.g. MSE, MAE, cos


###################################################################
# I/O module
###################################################################


InputData = namedtuple('InputData', 'X_train,y_train,X_val,y_val,X_test,qid_test')


def load_data(train_path, test_path, val_size):
    log.info('Loading input data from %r, %r', train_path, test_path)
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.question_text.values
    X_test = test_data.question_text.values
    
    if Options.ENABLE_NO_MATH:
        log.info('Scrubbing math')
        math_pattern = re.compile(r'\[\/?math\]|\\\w+')
        for corpus in [X_train, X_test]:
            for i in range(len(corpus)):
                corpus[i] = math_pattern.sub('', corpus[i])
    
    log.debug('Loaded: X_train = %r, X_test = %r', type(X_train), type(X_test))
    log.debug('X_train.shape = %r', X_train.shape)
    
    y_train = train_data.target.values.astype('float32')
    
    X_train_train, X_train_val, \
    y_train_train, y_train_val = train_test_split(
        X_train, y_train,
        stratify=y_train, test_size=val_size, shuffle=True,
    )
    
    if Options.TRAIN_CAP:
        X_train_train = X_train_train[:Options.TRAIN_CAP]
        y_train_train = y_train_train[:Options.TRAIN_CAP]
    
    return InputData(
        X_train=X_train_train,
        y_train=y_train_train,
        X_val=X_train_val,
        y_val=y_train_val,
        X_test=X_test,
        qid_test=test_data.qid,
    )


###################################################################
# preprocessing module
###################################################################


def load_text_embedding(path, skip_header=False):
    """Loads an embedding's text file.
    """
    log.info('Loading the embedding from %r, skip_header = %s', path, skip_header)
    
    with open(path, encoding='latin1') as embedding_file:
        split_lines = (
            line.split(' ')
            for line in embedding_file
        )
        
        if skip_header:
            next(split_lines)
            
        return {
            line[0]: np.fromiter(map(float, line[1:]), dtype='float32')
            for line in split_lines
        }
        
        
class MetaEmbedding:
    def __init__(self, paths):
        log.info('Creating a meta-embedding')
        
        self.dicts = [
            load_embedding(path)
            for path in paths
        ]
        
    def __contains__(self, word):
        return any(
            word in embedding_dict
            for embedding_dict in self.dicts
        )
        
    def __getitem__(self, word):
        total = None
        num_dicts = 0
        for embedding_dict in self.dicts:
            if word in embedding_dict:
                total = embedding_dict[word] if total is None else total + embedding_dict[word]
                num_dicts += 1
        return None if num_dicts == 0 else total / num_dicts
        
        
        
def load_embedding(path):
    if path == Paths.META:
        return MetaEmbedding([Paths.WORD_2_VEC, Paths.GLOVE, Paths.WIKI_NEWS])
    elif path == Paths.WORD_2_VEC:
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        return load_text_embedding(path, skip_header=(path == Paths.WIKI_NEWS))


class TextPreprocessor:
    NUM_EMBED_FEATURES = 300
    if Options.ENABLE_PUNCT:
        # TODO: masked swear words
        PATTERN = re.compile(r"[\w\']+|[.,;:!?\"]")
    else:
        PATTERN = re.compile(r"[\w\']+")
    
    def __init__(self):
        self.word_to_id = dict()
        self.oov_vocab = set()
        self.index = 1
    
    def fit(self, sets):
        self._initial_fit(sets)
        if Options.ENABLE_OOV_INFERENCE:
            self._fit_oov()
        
    def _initial_fit(self, sets):
        log.info('Training the text preprocessor')
        
        w2v = load_embedding(Options.WORD_EMBEDDING)
        word_to_id = self.word_to_id
        oov_vocab = self.oov_vocab
        finder = self.PATTERN.finditer
        
        index = 1
        known_words = 0
        unknown_words = 0
        
        log.info('Mapping known words to indices')
        
        for corpus in sets:
            for text in corpus:
                for word_match in finder(text):
                    word = word_match.group(0).strip("'")
                    if word in w2v:
                        if word_to_id.setdefault(word, index) == index:
                            index += 1
                        known_words += 1
                    else:
                        if len(word) >= Options.OOV_THRESHOLD:
                            oov_vocab.add(word)
                        unknown_words += 1
                    
        self.index = index
                    
        log.info('Generating features')
        features = np.zeros((self.index, self.NUM_EMBED_FEATURES), dtype='float32')
        for word, index in word_to_id.items():
            features[index, :] = w2v[word]
        self.features = features
                    
        log.info(
            'Training complete, len(word_to_id) = %s, len(oov_vocab) = %s, known_words = %s, unknown_words = %s',
            len(word_to_id),
            len(oov_vocab),
            known_words,
            unknown_words
        )
    
    def _fit_oov(self):
        log.info('Learning to correct misspellings from the training corpus')
        word_to_id = self.word_to_id
        words = sorted(word_to_id.keys(), key=lambda word: word_to_id[word])
        
        words_train, words_val, \
        features_train, features_val = train_test_split(
            words,
            self.features[1:, :],
            test_size=0.1,
        )
        
        def transform(p_erase=0, p_duplicate=0, p_swap=0):
            # TODO: add swaps
            # TODO: dedup sequence completely? count removed characters as a feature?
            @numba.jit(nopython=True)
            def augment(ids):
                new_ids = np.zeros((2 * len(ids),), dtype=np.int32)
                new_i = 0
                for i in range(len(ids)):
                    new_ids[new_i] = ids[i]
                    if np.random.random_sample() < p_erase:
                        continue
                    else:
                        new_i += 1
                        if np.random.random_sample() < p_duplicate:
                            new_ids[new_i] = ids[i]
                            new_i += 1
                if new_i >= 4:  # 4 == Options.OOV_THRESHOLD
                    if np.random.random_sample() < p_swap:
                        i_swap = np.random.randint(0, new_i - 1)
                        tmp = new_ids[i_swap]
                        new_ids[i_swap] = new_ids[i_swap + 1]
                        new_ids[i_swap + 1] = tmp
                return new_ids[:new_i]
            
            should_augment = (p_erase > 0 or p_duplicate > 0 or p_swap > 0)
            def do_transform(word):
                sequence = np.fromiter(
                    (
                        ord(char)
                        for char in word
                        if ord(char) < 128
                    ),
                    dtype='int32'
                )
                if should_augment:
                    sequence = augment(sequence)
                return sequence
                
            return do_transform
        
        word_model = build_word_model()
        
        log.info('Training a word model')
        train_generator = DataGenerator(words_train, transform(p_erase=0.05, p_duplicate=0.05, p_swap=0.2), features_train)
        val_generator = DataGenerator(words_val, transform(), features_val, batch_size=256)
        word_model.fit_generator(
            train_generator,
            validation_data=val_generator,
            epochs=Options.OOV_INFERENCE_EPOCHS,
            use_multiprocessing=True,
            workers=2,
            verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=2, min_delta=0.0001, verbose=1)
            ],
        )
        
        log.info('Applying the word model')
        words_oov = list(self.oov_vocab)
        oov_generator = DataGenerator(words_oov, transform(), shuffle=False, batch_size=256)
        oov_features = word_model.predict_generator(
            oov_generator,
            use_multiprocessing=True,
            workers=2,
        )
        
        log.info('Augmenting the vocabulary with predictions')
        oov_index = self.index
        for word in words_oov:
            word_to_id[word] = oov_index
            oov_index += 1
            
        self.features = np.r_[self.features, oov_features]
        self.index = oov_index
        
    def transform(self, text):
        word_to_id = self.word_to_id
        list_ids = []
        for word_match in self.PATTERN.finditer(text):
            word = word_match.group(0).strip("'")  # TODO: abstract away
            index = word_to_id.get(word)
            if index is not None:
                list_ids.append(index)  # TODO: optimize?
        return list_ids


###################################################################
# DataGenerator module
# - a flexible sequence input generator for CNNs and RNNs
# - some util functions
###################################################################

def div_up(a, b):
    """Integer division, rounding up.
    """
    return (a + b - 1) // b


def batch_slice(index, size):
    """Generates a slice corresponding to the the index and the batch size.
    """
    return slice(index * size, (index + 1) * size)


def pad_samples(list_list_ids, legnth_multiple=8):
    """Pads sequence samples with zeros so that they're the same length.
    """
    num_words = legnth_multiple * div_up(max(map(len, list_list_ids)), legnth_multiple)
    samples = np.zeros((len(list_list_ids), num_words), dtype='int32')
    for i, list_ids in enumerate(list_list_ids):
        samples[i, 0:len(list_ids)] = list_ids
    return samples


# TODO: generate more data by splicing together more offensive texts
# TODO: mini-epochs for more precise early stopping
class DataGenerator(keras.utils.Sequence):
    """Generates batches of sequences by padding them with zeros.
    To reduce the variation of lengths, makes sure they are always a multiple of 8.
    """
    def __init__(
        self,
        X,
        transform,
        labels=None,
        batch_size=64,
        num_mini_epochs=1,
        shuffle=True,
    ):
        self.labels = None if labels is None else labels.astype('float32')
        self.X = X
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(X)))
        self.num_mini_epochs = num_mini_epochs
        self.mini_epoch = -1  # see on_epoch_end
        self.on_epoch_end()
        
    def __len__(self):
        return div_up(len(self.mini_epoch_indices), self.batch_size)
    
    def __getitem__(self, index):
        batch_indices = self.mini_epoch_indices[batch_slice(index, self.batch_size)]
        
        samples = pad_samples([self.transform(self.X[i]) for i in batch_indices])
        if self.labels is None:
            return samples
        else:
            return samples, self.labels[batch_indices]
    
    def on_epoch_end(self):
        self.mini_epoch = (self.mini_epoch + 1) % self.num_mini_epochs
        if self.mini_epoch == 0 and self.shuffle:
            np.random.shuffle(self.indices)
            
        mini_epoch_size = div_up(len(self.indices), self.num_mini_epochs)
        self.mini_epoch_indices = self.indices[batch_slice(self.mini_epoch, mini_epoch_size)]


###################################################################
# Model generation module
###################################################################

def build_word_model():
    input_layer = K.Input((None,), dtype='int32')
    layer = K.Embedding(128, 60)(input_layer)
    
    layer = K.CuDNNGRU(200, return_sequences=True)(layer)
    layer = K.CuDNNGRU(300)(layer)
    layer = K.Dense(300)(layer)
    
    model = keras.models.Model(input_layer, layer)
    model.compile(
        optimizer=Options.OOV_INFERENCE_OPTIMIZER,
        loss=Options.OOV_INFERENCE_LOSS,
    )
    return model


def build_cnn_layers(embed_layer):
    cnn_layers = []
    for width in (2, 4, 6):
        layer = K.Conv1D(256, width, padding='same')(embed_layer)
        layer = K.BatchNormalization()(layer)
        layer = K.LeakyReLU(0.01)(layer)

        layer = K.Conv1D(256, width, padding='same')(layer)
        layer = K.BatchNormalization()(layer)
        layer = K.LeakyReLU(0.01)(layer)

        cnn_layers.append(K.GlobalMaxPooling1D()(layer))
        
    layer = K.concatenate(cnn_layers)

    layer = K.Dense(128)(layer)
    layer = K.BatchNormalization()(layer)
    layer = K.LeakyReLU(0.01)(layer)
    layer = K.Dropout(0.3)(layer)
    
    return layer


def build_gru_layers(layer):
    layer = K.CuDNNGRU(300)(layer)
    return layer
    

def build_model(features):
    log.info('Building the model, features shape = %r', features.shape)
    
    input_layer = K.Input((None,), dtype='int32')
    
    embed_layer = K.Embedding(
        features.shape[0],
        features.shape[1],
        weights=[features],
        trainable=False,
    )(input_layer)

    layer = None
    if Options.MODEL_TYPE == Models.CNN:
        layer = build_cnn_layers(embed_layer)
    elif Options.MODEL_TYPE == Models.GRU:
        layer = build_gru_layers(embed_layer)
    elif Options.MODEL_TYPE == Models.BOTH:
        cnn_layer = build_cnn_layers(embed_layer)
        gru_layer = build_gru_layers(embed_layer)
        layer = K.concatenate([cnn_layer, gru_layer])
        
    layer = K.Dense(1, activation='sigmoid')(layer)
    
    model = keras.models.Model(input_layer, layer)
    if Options.LOG_LEVEL == log.DEBUG:
        log.debug('Model summary:')
        model.summary()
        
    log.info('Compiling the model')
        
    model.compile(
        optimizer=Options.OPTIMIZER,
        loss='binary_crossentropy',
    )
        
    return model


###################################################################
# Main module
###################################################################


def fit_model(model, data, transform, num_epochs, num_mini_epochs):
    log.info('Training the model')
    
    train_generator = DataGenerator(data.X_train, transform, data.y_train, num_mini_epochs=num_mini_epochs)
    val_generator = DataGenerator(data.X_val, transform, data.y_val, batch_size=256)
    
    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs * num_mini_epochs,
        use_multiprocessing=True,
        workers=2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        ],
        verbose=2,
    )
    
    log.info('Generating validation probabilities')
    
    eval_val_generator = DataGenerator(data.X_val, transform, data.y_val, shuffle=False, batch_size=256)
    prob_val = model.predict_generator(
        eval_val_generator,
        use_multiprocessing=True,
        workers=2,
    )
    
    log.info('Computing the best F1 threshold')

    thresh_options = np.linspace(0.01, 0.5, 100)
    f1_values = np.array([
        f1_score(data.y_val, (prob_val > thresh).astype('int32'))
        for thresh in thresh_options
    ])
    best_thresh_index = np.argmax(f1_values)
    best_thresh = thresh_options[best_thresh_index]
    log.info('best_thresh = %s, f1_score = %s', best_thresh, f1_values[best_thresh_index])

    return best_thresh


def predict(X_test, model, transform, thresh):
    log.info('Generating test predictions')
    test_generator = DataGenerator(X_test, transform, shuffle=False, batch_size=256)
    prob_test = model.predict_generator(
        test_generator,
        use_multiprocessing=True,
        workers=2,
    )
    
    return (prob_test > thresh).astype('int32')
    
    
def save_result(qid, y, path):
    log.info('Saving the predictions')
    
    submission = pd.DataFrame({'qid': qid, 'prediction': y.ravel()})
    submission.to_csv(path, index=False)


def main():
    log.basicConfig(level=Options.LOG_LEVEL, stream=sys.stdout, format='[%(levelname)s] %(message)s')
    data = load_data('../input/train.csv', '../input/test.csv', val_size=0.1)
    
    preprocessor = TextPreprocessor()
    preprocessor.fit([data.X_train, data.X_val, data.X_test])
    model = build_model(preprocessor.features)
    
    # TODO: filter the training set, drop noisy samples (like the ones with zero words)
    thresh = fit_model(
        model,
        data,
        transform=preprocessor.transform,
        num_epochs=Options.NUM_EPOCHS,
        num_mini_epochs=Options.NUM_MINI_EPOCHS,
    )
    
    y_test = predict(data.X_test, model, transform=preprocessor.transform, thresh=thresh)
    save_result(data.qid_test, y_test, 'submission.csv')
    
    
if __name__ == '__main__':
    main()