import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
from keras.preprocessing import text, sequence
from keras import backend as K



BATCH_SIZE = 1000
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 3
MAX_LEN = 230
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

#train_df = pd.read_csv('../input/train.csv')
#test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

#y_train1 = train_df[TARGET_COLUMN].values

train_df = train_df[:1300000]

print("data read into memory")

train_df[TARGET_COLUMN] = np.where(train_df[TARGET_COLUMN] >= 0.5, 1, 0)

x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

print("pre processing done")


def precision(y_true, y_pred):

    """Precision metric.



    Only computes a batch-wise average of precision.



    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def recall(y_true, y_pred):

    """Recall metric.



    Only computes a batch-wise average of recall.



    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
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

EMBEDDING_PATH ='../input/glove840b300dtxt/glove.840B.300d.txt'


embedding_matrix = build_matrix(tokenizer.word_index, EMBEDDING_PATH)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(embedding_matrix.shape[0], 300, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.LSTM(LSTM_UNITS, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.LSTM(LSTM_UNITS)),
    tf.keras.layers.Dense(DENSE_HIDDEN_UNITS, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(DENSE_HIDDEN_UNITS, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(DENSE_HIDDEN_UNITS//4, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

class_weight = {0: 1.,
                1: 4.}

optimizer = tf.keras.optimizers.Adam(lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics = ['binary_accuracy'])
              
print("starting training")

history = model.fit(x = x_train, y =y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,class_weight=class_weight)

print("predicting")
                    
predictions = model.predict(x_test,batch_size=BATCH_SIZE)

predictions = list(np.array(predictions).reshape(-1,))

print("number of ones : ", sum(predictions))

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})


submission.to_csv('submission.csv', index=False)