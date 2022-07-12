# Simple keras kernel exploring CNN before the usual RNN (LSTM or GRU). EMB - CONV - MAXP - CONV - MAXP - GRU - DENSE
# Base script ideas from https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051
# Load weights from file, etc.

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint



MAX_FEATURES = 20000
MAX_TEXT_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 4
VALIDATION_SPLIT = 0.1
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def get_X_train_X_test(train, test):
    train_raw_text = train["comment_text"].fillna("MISSINGVALUE").values
    test_raw_text = test["comment_text"].fillna("MISSINGVALUE").values
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)

def get_Y(train):
    return train[CLASSES_LIST].values

def get_model():
    embed_size = 128
    inp = Input(shape=(MAX_TEXT_LENGTH, ))
    main = Embedding(MAX_FEATURES, embed_size)(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()     
    return model

def train_fit_predict(model, X_train, X_test, y):
    file_path="weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early]
    model.fit(X_train, y, batch_size=BATCH_SIZE, epochs=EPOCHS,  verbose=0, validation_split=VALIDATION_SPLIT, callbacks=callbacks_list)
    model.load_weights(file_path)
    return model.predict(X_test)

def submit(y_test):
    sample_submission = pd.read_csv("../input/sample_submission.csv")
    sample_submission[CLASSES_LIST] = y_test
    sample_submission.to_csv("baseline.csv", index=False)


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train, X_test = get_X_train_X_test(train, test)
y = get_Y(train)

y_test = train_fit_predict(get_model(), X_train, X_test, y)

submit(y_test)