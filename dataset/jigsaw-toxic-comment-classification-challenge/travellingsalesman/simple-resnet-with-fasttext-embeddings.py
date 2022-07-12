# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import gc
import random
import gensim
import pandas as pd
from keras.preprocessing.text import Tokenizer
random.seed(666)

from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Input, Dense, Activation, Embedding, \
    GlobalMaxPool1D, Flatten, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers.merge import add
from keras.models import Model
from gensim.models.keyedvectors import KeyedVectors


EMBEDDING_FILE_FASTTEXT = '../input/fasttext-pretrained-wordvec/crawl-300d-2M.vec'
MAX_FEATURES = 18000
VEC_SIZE = 400

def get_keras_data_words(dataset, padding=100, nosingle=False):
        ret_dataset = {
            'comment_text_seq_glove': pad_sequences(dataset.comment_text_seq, maxlen=padding, padding='post')
        }
        return ret_dataset

def get_embedding_weights(dataset, tokenizar, file_path, binary=True):
    model = None
    model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=binary) # Much faster if read from binary...

    EMB_SIZE = len(model.word_vec('hello'))

    word_index = tokenizar.word_index
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(0.0, 0, (nb_words, EMB_SIZE))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        try:
            embedding_vector = model.word_vec(word) # Gensim 3.2.0 model.wv[word]
            embedding_matrix[i] = embedding_vector
        except:
            print(word + ' not in voc')
    return embedding_matrix, EMB_SIZE

def text_preprocessing(dataset):
    print('\tRemoving stopwords...')
    dataset['comment_text'] = dataset['comment_text'].str.lower()
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"http\S+", "")
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"http", "")
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"@\S+", "")
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"@", "at")
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"'s", "")
    dataset['comment_text'] = dataset['comment_text'].str.replace(r"'", "")

    dataset['comment_text'].fillna('__NA__', inplace=True)
    dataset['comment_text_prep'] = [str(p) for p in dataset['comment_text']]
    print('\tDone.')
    return dataset

def tokenization(dataset):
    tokenizar = Tokenizer(num_words=MAX_FEATURES,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    print('\tFit on texts...')
    tokenizar.fit_on_texts(np.array(dataset['comment_text_prep']))
    print('\tDone.')
    print('\tTokenization...')
    dataset['comment_text_seq'] = tokenizar.texts_to_sequences(np.array(dataset['comment_text_prep']))
    print('\tDone.')
    return dataset, tokenizar

def keras_preprocessing_words(train, test):
    full_dataset = pd.DataFrame(pd.concat([train, test]))
    tokenizar = None

    print(full_dataset.columns)
    print('Done.')

    print('Text preprocessing...')
    full_dataset = text_preprocessing(full_dataset)
    print('Done.')

    print('Text tokenization...')
    full_dataset, tokenizar = tokenization(full_dataset)
    print('Done.')

    # Embedding weights
    print('Get embedding weights...')

    embedding_weights = None
    EMB_SIZE = None
    embedding_weights, EMB_SIZE = get_embedding_weights(full_dataset, tokenizar, EMBEDDING_FILE_FASTTEXT,
                                                        binary=False)
    print('Done.')
    return full_dataset, embedding_weights, EMB_SIZE

def get_keras_target_words(dataset):
    ret_dataset = {
        'target': dataset.as_matrix()
    }
    return ret_dataset

def get_model_resnet(X_train, X_test, emb_size
                     , embedding_weights=None
                     , activation_general='relu'
                     , dr_rate=0.1
                     , trainable_embeddings=False
                     ):
    model = 0

    MAX_VOCAB = np.max([
        np.max(X_train)
        , np.max(X_test)
    ])

    inp = Input(shape=[len(X_train[0])], name='comment_text_seq_glove')
    main = Embedding(MAX_VOCAB + 1, emb_size, weights=[embedding_weights], name='embedding',
                     trainable=trainable_embeddings)(inp)
    main = SpatialDropout1D(dr_rate)(main)
    i = 0
    main = Conv1D(filters=64, kernel_size=3, padding='same')(main)
    i_l1 = MaxPooling1D(pool_size=2)(main)

    main = Conv1D(filters=64, kernel_size=3, padding='same')(i_l1)
    main = Activation(activation_general)(main)
    main = Conv1D(filters=64, kernel_size=3, padding='same')(main)
    main = add([main, i_l1])
    main = Activation(activation_general)(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = SpatialDropout1D(dr_rate)(main)

    i_l1 = Conv1D(filters=128, kernel_size=1, padding='same')(main)

    main = Conv1D(filters=128, kernel_size=3, padding='same')(main)
    main = Activation(activation_general)(main)
    main = Conv1D(filters=128, kernel_size=3, padding='same')(main)
    main = add([main, i_l1])
    main = Activation(activation_general)(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = SpatialDropout1D(dr_rate)(main)

    i_l1 = Conv1D(filters=256, kernel_size=1, padding='same')(main)

    main = Conv1D(filters=256, kernel_size=3, padding='same')(main)
    main = Activation(activation_general)(main)
    main = Conv1D(filters=256, kernel_size=3, padding='same')(main)
    main = add([main, i_l1])
    main = Activation(activation_general)(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = SpatialDropout1D(dr_rate)(main)

    i_l1 = Conv1D(filters=512, kernel_size=1, padding='same')(main)

    main = Conv1D(filters=512, kernel_size=3, padding='same')(main)
    main = Activation(activation_general)(main)
    main = Conv1D(filters=512, kernel_size=3, padding='same')(main)
    main = add([main, i_l1])
    main = Activation(activation_general)(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = SpatialDropout1D(dr_rate)(main)

    main = GlobalMaxPool1D()(main)

    main = Dense(1024, activation=activation_general)(main) # Should be 4096
    main = Dense(512, activation=activation_general)(main) # Should be 2048
    main = Dense(6, activation="sigmoid", name="dense2")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# Read input data
def train_and_predict(train, test):
    TARGETS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print('Reading dataset...')

    sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

    ntrain = train.shape[0]
    ntest = test.shape[0]

    # Get Keras dataset for words network
    full_dataset, embedding_weights, EMB_SIZE \
        = keras_preprocessing_words(train, test)

    print('Split dataset...')
    train = full_dataset.iloc[:ntrain, ]
    test = full_dataset.iloc[ntrain:, ]

    X_train = get_keras_data_words(train, nosingle=True, padding=VEC_SIZE)['comment_text_seq_glove']
    X_test = get_keras_data_words(test, nosingle=True, padding=VEC_SIZE)['comment_text_seq_glove']

    y_train = get_keras_target_words(train[TARGETS])['target']
    gc.collect()
    print('Done.')

    # FITTING THE MODEL
    BATCH_SIZE = 512
    EPOCHS = 2 # better CV results with ~5 (should be trained with 10-folds CV)

    model = get_model_resnet(X_train, X_test
                             , emb_size=EMB_SIZE
                             , embedding_weights=embedding_weights
                             , activation_general='relu'
                             , dr_rate=0.15)

    model.fit(X_train, y_train
              , epochs=EPOCHS
              , batch_size=BATCH_SIZE
              , verbose=2)
    test_predicts = model.predict(X_test, batch_size=128, verbose=1)

    sample_submission[TARGETS] = test_predicts
    sample_submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    NROWS = None
    train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv', nrows=NROWS)
    test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv', nrows=NROWS)
    train_and_predict(train, test)
