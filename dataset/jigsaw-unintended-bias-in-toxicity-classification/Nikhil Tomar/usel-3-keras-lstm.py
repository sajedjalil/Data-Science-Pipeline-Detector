# https://www.kaggle.com/tunguz/tensorflow-hub-sentence-embeddings/notebook

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

import tensorflow as tf
import tensorflow_hub as hub

import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, Add, Concatenate, Multiply, Subtract, Dot, Reshape, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D

from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D

from keras.preprocessing import text, sequence
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer

def use_encode(path, data, bs=512):
    """Universal Sentence Encoder"""
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module(path)
        embedded_text = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    embedding = np.array([])
    sess = tf.Session(graph=g)
    sess.run(init_op)

    for i in tqdm(range(0, len(data), bs)):
        x = data[i : min(i+bs, len(data))]
        x_embed = sess.run(embedded_text, feed_dict={text_input: x})
        embedding = np.append(embedding, x_embed)

    sess.close()
    embedding = np.reshape(embedding, (-1, 512))
    return embedding

class Generator(keras.utils.Sequence):
    """Generating the data for training"""
    def __init__(self, x, y, y_aux, batch_size=128, path=None):
        self.x = x
        self.y = y
        self.y_aux = y_aux
        self.batch_size = batch_size
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.x)//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = self.x[index*self.batch_size:(index+1)*self.batch_size]
        batch_y = self.y[index*self.batch_size:(index+1)*self.batch_size]
        batch_y_aux = self.y_aux[index*self.batch_size:(index+1)*self.batch_size]
        #word_batch_x = use_encode(self.path, batch_x, self.batch_size)

        return [batch_x], [batch_y, batch_y_aux]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))

def build_model(num_aux_targets):
    word_input = Input(shape=(512,), name="word_input")
    embed = Reshape((1, 512))(word_input)
    embed = SpatialDropout1D(0.2)(embed)

    x = Bidirectional(CuDNNGRU(2*LSTM_UNITS, return_sequences=True))(embed)
    x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)

    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)

    hidden = Concatenate()([avg_pool1, max_pool1])
    hidden = Add()([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = Add()([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid', name="result")(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid', name="aux")(hidden)

    model = Model(inputs=[word_input], outputs=[result, aux_result])
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr=1e-3, decay=1e-5), metrics=["acc"])

    return model


if __name__ == "__main__":
    ## Path
    DATA_PATH = "../input/jigsaw-unintended-bias-in-toxicity-classification/"
    USE_PATH = "../input/usel3/usel3/"
    
    os.listdir(USE_PATH)

    ## CSV files
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    ## Columns
    IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    TEXT_COLUMN = 'comment_text'
    TARGET_COLUMN = 'target'

    ## Train and Test values
    x_train = train_df[TEXT_COLUMN].values
    y_train = np.where(train_df[TARGET_COLUMN] >= 0.5, 1, 0)
    y_aux_train = train_df[AUX_COLUMNS].values
    x_test = test_df[TEXT_COLUMN].values
    
    x_train = x_train[:1500000]

    ## USE Encode
    use_x_test = use_encode(USE_PATH, x_test, bs=768)
    use_x_train = use_encode(USE_PATH, x_train, bs=768)

    ## Hyperparameters
    BATCH_SIZE = 512
    LSTM_UNITS = 128
    DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
    EPOCHS = 6

    ## Generator
    train_gen = Generator(use_x_train, y_train, y_aux_train, batch_size=BATCH_SIZE, path=USE_PATH)

    ## Model
    model = build_model(y_aux_train.shape[-1])
    model.summary()

    ## Training
    checkpoint_predictions = []

    for global_epoch in range(EPOCHS):
        model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.__len__(),
            epochs=1,
            callbacks=[LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))]
        )
        p = model.predict([use_x_test], batch_size=2048, verbose=1)[0].flatten()
        checkpoint_predictions.append(p)

    ## Prediction
    lstm_pred = np.average(checkpoint_predictions, axis=0)

    ## Submission
    submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': lstm_pred
    })

    ## Submission to CSV
    submission.to_csv('submission.csv', index=False)
    print(submission.head())
