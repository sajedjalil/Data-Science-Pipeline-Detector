# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (Input, Dropout, Dense, Concatenate,
    BatchNormalization, Activation, concatenate, GRU, LSTM, 
    Embedding, Flatten, Conv1D, GlobalMaxPooling1D, Reshape)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from tensorflow.keras.utils import plot_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

def rmsle_keras(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def preprocessing():
    train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv')
    test = pd.read_table('../input/mercari-price-suggestion-challenge/test_stg2.tsv')
    train.fillna('missing', inplace=True)
    test.fillna('missing', inplace=True)
    print('カテゴリーデータを数値化...')
    for label in ['brand_name', 'category_name']:
        le = LabelEncoder()
        le.fit(np.hstack([train[label], test[label]]))
        train[label] = le.transform(train[label])
        test[label] = le.transform(test[label])
    print('テキストデータをシーケンス化...')
    raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(raw_text)
    
    word_index = tokenizer.word_index
    train.item_description = tokenizer.texts_to_sequences(train.item_description.str.lower())
    test.item_description = tokenizer.texts_to_sequences(test.item_description.str.lower())
    train.name = tokenizer.texts_to_sequences(train.name.str.lower())
    test.name = tokenizer.texts_to_sequences(test.name.str.lower())
    max_name = np.max([np.max(train.name.apply(lambda x: len(x))), 
                       np.max(test.name.apply(lambda x: len(x)))])
    max_item_description = np.max([np.max(train.item_description.apply(lambda x: len(x))),
                                   np.max(test.item_description.apply(lambda x: len(x)))])
    max_text = len(tokenizer.word_index) + 1
    max_category = np.max([train.category_name.max(), test.category_name.max()]) + 1
    max_brand_name = np.max([train.brand_name.max(), test.brand_name.max()]) + 1
    max_item_condition_id = np.max([train.item_condition_id.max(), test.item_condition_id.max()]) + 1
    x_train = {'name': pad_sequences(train.name, maxlen=max_name),
               'item_description': pad_sequences(train.item_description, maxlen=max_item_description),
               'brand_name': np.array(train.brand_name),
               'category_name': np.array(train.category_name),
               'item_condition_id': np.array(train.item_condition_id),
               'shipping': np.array(train.shipping)}
    x_test = {'name': pad_sequences(test.name, maxlen=max_name),
              'item_description': pad_sequences(test.item_description, maxlen=max_item_description),
              'brand_name': np.array(test.brand_name),
              'category_name': np.array(test.category_name),
              'item_condition_id': np.array(test.item_condition_id),
              'shipping': np.array(test.shipping)}
    y_train = np.array(train.price)
    y_train = np.log1p(y_train)
    return x_train, x_test, y_train, max_name, max_item_description, max_text, \
           max_category, max_brand_name, max_item_condition_id

class Mercari_Model:
    def __init__(self):
        x_train, x_test, y_train, max_name, max_item_description, max_text, \
        max_category, max_brand_name, max_item_condition_id = preprocessing()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.max_name = max_name
        self.max_item_description = max_item_description
        self.max_text = max_text
        self.max_category = max_category
        self.max_brand_name = max_brand_name
        self.max_item_condition_id = max_item_condition_id
        self.model = None

    def create_model(self):
        # Input Layers
        name_input = Input(shape=(self.x_train['name'].shape[1],), name='name')
        item_description_input = Input(shape=(self.x_train['item_description'].shape[1],), name='item_description')
        brand_name_input = Input(shape=(1,), name='brand_name')
        category_name_input = Input(shape=(1,), name='category_name')
        item_condition_id_input = Input(shape=(1,), name='item_condition_id')
        shipping_input = Input(shape=(1,), name='shipping')

        # Embedding Layers
        embed_name = Embedding(self.max_text, 32)(name_input)
        embed_item_decription = Embedding(self.max_text, 32)(item_description_input)
        embed_brand_name = Embedding(self.max_brand_name, 16)(brand_name_input)
        embed_category_name = Embedding(self.max_category, 16)(category_name_input)
        embed_item_condition_id = Embedding(self.max_item_condition_id, 16)(item_condition_id_input)
        embed_shipping = Embedding(2, 4)(shipping_input)

        # TextCNN Layers (You can switch to RNN)
        conv_item_decription1 = Conv1D(filters=64, 
                                      kernel_size=3, activation='relu')(embed_item_decription)
        conv_item_decription2 = Conv1D(filters=64, 
                                      kernel_size=4, activation='relu')(embed_item_decription)
        conv_item_decription3 = Conv1D(filters=64, 
                                      kernel_size=5, activation='relu')(embed_item_decription)
        pool_item_decription1 = GlobalMaxPooling1D()(conv_item_decription1)
        pool_item_decription2 = GlobalMaxPooling1D()(conv_item_decription2)
        pool_item_decription3 = GlobalMaxPooling1D()(conv_item_decription3)
        concat_item_decription = Concatenate(axis=1)([pool_item_decription1, pool_item_decription2, pool_item_decription3])
        
        conv_name1 = Conv1D(filters=16, 
                           kernel_size=3, activation='relu')(embed_name)
        conv_name2 = Conv1D(filters=16, 
                           kernel_size=4, activation='relu')(embed_name)
        conv_name3 = Conv1D(filters=16, 
                           kernel_size=5, activation='relu')(embed_name)
        pool_name1 = GlobalMaxPooling1D()(conv_name1)
        pool_name2 = GlobalMaxPooling1D()(conv_name2)
        pool_name3 = GlobalMaxPooling1D()(conv_name3)
        concat_name = Concatenate(axis=1)([pool_name1, pool_name2, pool_name3])

        # concatenate
        concat_layer = concatenate([
            Flatten()(embed_brand_name),
            Flatten()(embed_category_name),
            Flatten()(embed_item_condition_id),
            Flatten()(embed_shipping),
            concat_item_decription,
            concat_name
        ])
        concat_layer = Activation('relu')(concat_layer)
        bn_concat = BatchNormalization()(concat_layer)
        
        # Fully Connected Layer
        bn1 = BatchNormalization()(Dense(256, activation='relu', use_bias=False)(bn_concat))
        bn2 = BatchNormalization()(Dense(128, activation='relu', use_bias=False)(bn1))
        fc3 = Dense(64, activation='relu')(bn2)
        output = Dense(1, activation='linear')(fc3)

        # Model
        self.model = Model([name_input, item_description_input, brand_name_input,
                        category_name_input, item_condition_id_input, shipping_input],
                        output)
        self.model.compile(loss='logcosh', optimizer='adam', metrics=['mae', rmsle_keras])
        
    def train_model(self, batch_size=64, epochs=3):
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict_testset(self, filename='submission.csv', batch_size=64):
        preds = self.model.predict(self.x_test, batch_size=batch_size)
        preds = np.expm1(preds)
        submission = pd.DataFrame({'test_id': range(len(self.x_test['name']))})
        submission["price"] = preds
        submission.to_csv('./'+filename, index=False)
        
if __name__ == '__main__':
    model = Mercari_Model()
    model.create_model()
    model.train_model()
    model.predict_testset()