import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Reshape, Permute
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from keras.losses import binary_crossentropy
from keras import backend as K

EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 256


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix
    

def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]


    

def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

x_train = preprocess(train['comment_text'])

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
# Overall
weights = np.ones((len(x_train),)) / 4
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
x_test = preprocess(test['comment_text'])

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

import pickle
import gc

with open('temporary.pickle', mode='wb') as f:
    pickle.dump(x_test, f) # use temporary file to reduce memory

del identity_columns, weights, tokenizer, train, test, x_test
gc.collect()

checkpoint_predictions = []
weights = []

def build_model(embedding_matrix, num_aux_targets, loss_weight):
    K.clear_session()
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    c1_3 = Conv1D(filters=LSTM_UNITS, kernel_size=3, padding='same', activation='relu', name='c1_3')(x)
    c1_3 = MaxPooling1D(2)(c1_3)
    c1_4 = Conv1D(filters=LSTM_UNITS, kernel_size=4, padding='same', activation='relu', name='c1_4')(x)
    c1_4 = MaxPooling1D(2)(c1_4)
    c1_5 = Conv1D(filters=LSTM_UNITS, kernel_size=5, padding='same', activation='relu', name='c1_5')(x)
    c1_5 = MaxPooling1D(2)(c1_5)
    c1_6 = Conv1D(filters=LSTM_UNITS, kernel_size=6, padding='same', activation='relu', name='c1_6')(x)
    c1_6 = MaxPooling1D(2)(c1_6)

    c1_concat = concatenate([
        c1_3, c1_4, c1_5, c1_6
    ])
    c2_2 = Conv1D(filters=LSTM_UNITS, kernel_size=2, padding='same', activation='relu', name='c2_2')(c1_concat)
    c2_2 = MaxPooling1D(2)(c2_2)
    c2_3 = Conv1D(filters=LSTM_UNITS, kernel_size=3, padding='same', activation='relu', name='c2_3')(c1_concat)
    c2_3 = MaxPooling1D(2)(c2_3)
    c2_concat = concatenate([concatenate([c2_2, c2_3], axis=1), c1_5, c1_6])

    c3_2 = Conv1D(filters=LSTM_UNITS, kernel_size=2, padding='same', activation='relu', name='c3_2')(c2_concat)
    c3_2 = MaxPooling1D(2)(c3_2)
    c3_3 = Conv1D(filters=LSTM_UNITS, kernel_size=3, padding='same', activation='relu', name='c3_3')(c2_concat)
    c3_3 = MaxPooling1D(2)(c3_3)
    c3_concat = concatenate([c3_2, c3_3, c2_2, c2_3])

    c4_2 = Conv1D(filters=LSTM_UNITS, kernel_size=1, padding='same', activation='relu', name='c4_2')(c3_concat)
    hidden = c4_2
    hidden = Permute((2,1))(hidden)
    hidden = Conv1D(filters=4, kernel_size=1, padding='same', name='squish', activation='relu')(hidden)
    hidden = Reshape((DENSE_HIDDEN_UNITS,))(hidden)
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    hidden = concatenate([hidden, aux_result])
    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')

    return model

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            callbacks=[
                LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
            ]
        )
        with open('temporary.pickle', mode='rb') as f:
            x_test = pickle.load(f) # use temporary file to reduce memory
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        del x_test
        gc.collect()
        weights.append(2 ** global_epoch)
    del model
    gc.collect()

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
df_submit.prediction = predictions
df_submit.to_csv('submission.csv', index=False)
