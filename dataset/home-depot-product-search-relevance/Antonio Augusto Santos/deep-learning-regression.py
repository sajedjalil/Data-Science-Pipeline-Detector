# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:45:41 2016

@author: ur57
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.constraints import nonneg
from keras.layers.advanced_activations import LeakyReLU

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import theano
theano.config.openmp = True

import time
start_time = time.time()

stop = stopwords.words('english')
stem = SnowballStemmer('english').stem


def rmse(y_true, y_pred):
    from keras import backend as k
    from keras.objectives import mean_squared_error

    return k.sqrt(mean_squared_error(y_true, y_pred))


def stemmer(sent):
    return ' '.join([stem(word)
                    for word in word_tokenize(sent.lower())
                    if word not in stop])


def data_gen(df_all, y_train=None, n_batch=5, loop_forever=True):
    res = []
    y_res = []
    while True:
        pos = 0
        for row in df_all.itertuples():
            search_term = str(row.search_term)
            bag = str(row.bag)

            arr = np.ndarray((2, 64, 4800), np.float32)
            arr[0].fill(128)
            arr[1].fill(255)
            for i in range(len(search_term)):
                letter_i = search_term[i]
                val_i = ord(letter_i)

                for j in range(len(bag)):
                    letter_j = bag[j]
                    val_j = ord(letter_j)

                    arr[0, i, j] = val_i - val_j
                    arr[1, i, j] = val_i + val_j

            arr = arr.reshape((2, 640, 480))
            res.append(arr)
            if y_train is not None:
                y_res.append(y_train[pos])
                pos += 1

            if len(res) == n_batch:
                res = np.asarray(res)
                if y_train is not None:
                    y_res = np.asarray(y_res)
                    yield (res, y_res)
                    y_res = []
                else:
                    yield (res)

                res = []

        if not loop_forever:
            if len(res) > 0:
                res = np.asarray(res)
                if y_train is not None:
                    y_res = np.asarray(y_res)
                    yield (res, y_res)
                    y_res = []
                else:
                    yield (res)

            break


def batch_test(model, data_x, data_y, n_batch=5):
    foo = data_gen(data_x, data_y, n_batch=n_batch, loop_forever=False)
    res = []
    for x, y in foo:
        ans = model.test_on_batch(x, y)
        res.extend(ans)
        print('{:5.4f} - {:5.4f}'.format(np.mean(res), np.std(res)))
    return res


def batch_predict(model, data, n_batch=10):
    foo = data_gen(data, n_batch=n_batch, loop_forever=False)
    total = data.shape[0]
    res = []
    for x in foo:
        ans = model.predict(x).ravel().tolist()
        res.extend(ans)
        print('{:d} de {:d}'.format(len(res), total))
    return res

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1", nrows=250)
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1", nrows=100)
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr = pd.read_csv('../input/attributes.csv')
df_attr.dropna(inplace=True)

print("--- Loading data: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()

df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

material = dict()
df_attr['about_material'] = df_attr['name'].str.lower().str.contains('material')
for row in df_attr[df_attr['about_material']].iterrows():
    r = row[1]
    product = r['product_uid']
    value = r['value']
    material.setdefault(product, '')
    material[product] = material[product] + ' ' + str(value)
df_material = pd.DataFrame.from_dict(material, orient='index')
df_material = df_material.reset_index()
df_material.columns = ['product_uid', 'material']

color = dict()
df_attr['about_color'] = df_attr['name'].str.lower().str.contains('color')
for row in df_attr[df_attr['about_color']].iterrows():
    r = row[1]
    product = r['product_uid']
    value = r['value']
    color.setdefault(product, '')
    color[product] = color[product] + ' ' + str(value)
df_color = pd.DataFrame.from_dict(color, orient='index')
df_color = df_color.reset_index()
df_color.columns = ['product_uid', 'color']

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')

print("--- Processing columns: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()

num_train = df_train.shape[0]
id_test = df_test['id']
y_train = df_train['relevance'].values

del df_color, df_attr, df_brand, df_material, df_train, df_test

df_all.fillna('', inplace=True)

df_all['search_term'] = df_all['search_term'].map(lambda x: stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: stemmer(x))
df_all['brand'] = df_all['brand'].map(lambda x: stemmer(x))
df_all['material'] = df_all['material'].map(lambda x: stemmer(x))
df_all['color'] = df_all['color'].map(lambda x: stemmer(x))

df_all['bag'] = df_all['product_title'].str.pad(df_all['product_title'].str.len().max(), side='right') + ' ' + \
                df_all['product_description'].str.pad(df_all['product_description'].str.len().max(), side='right') + ' ' + \
                df_all['brand'].str.pad(df_all['brand'].str.len().max(), side='right') + ' ' + \
                df_all['material'].str.pad(df_all['material'].str.len().max(), side='right') + ' ' + \
                df_all['color'].str.pad(df_all['color'].str.len().max(), side='right')

print("--- Stemming columns: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()

df_train = df_all[:num_train]
df_test = df_all[num_train:]

model = Sequential()
model.add(BatchNormalization(input_shape=(2, 640, 480)))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(16, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# More layers make for larger run time (and, in theory, better prediction)
'''
model.add(Convolution2D(24, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(24, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(32, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(40, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(40, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU())
'''
model.add(Dense(256))
model.add(LeakyReLU())
model.add(Dropout(0.50))
'''

model.add(Dense(1, W_constraint=nonneg()))
model.add(Activation('relu'))

model.compile(loss=rmse, optimizer='adam')

print("--- Compiling model: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()

# Try to work around class inbalance, increases the training size a LOT
'''
df_train2 = pd.DataFrame()
df_train2 = df_train2.append(df_train.ix[np.repeat(df_train.index[(y_train >= 1.0) & (y_train < 1.5)].tolist(), 7)])
df_train2 = df_train2.append(df_train.ix[np.repeat(df_train.index[(y_train >= 1.5) & (y_train < 2.0)].tolist(), 5)])
df_train2 = df_train2.append(df_train.ix[np.repeat(df_train.index[(y_train >= 2.0) & (y_train < 2.5)].tolist(), 1)])
df_train2 = df_train2.append(df_train.ix[np.repeat(df_train.index[(y_train >= 2.5) & (y_train <= 3)].tolist(), 1)])
df_train = df_train2

ybin1 = np.repeat(y_train[(y_train >= 1.0) & (y_train < 1.5)], 7, axis=0)
ybin2 = np.repeat(y_train[(y_train >= 1.5) & (y_train < 2.0)], 5, axis=0)
ybin3 = np.repeat(y_train[(y_train >= 2.0) & (y_train < 2.5)], 1, axis=0)
ybin4 = np.repeat(y_train[(y_train >= 2.5) & (y_train <= 3)], 1, axis=0)

y_train = np.concatenate((ybin1, ybin2, ybin3, ybin4))

indexes = list(range(len(y_train)))
np.random.shuffle(indexes)

df_train = df_train.iloc[indexes]
y_train = y_train[indexes]
'''

model.fit_generator(data_gen(df_train, y_train, n_batch=5),
                    samples_per_epoch=df_train.shape[0],
                    nb_epoch=1,
                    nb_worker=4,
                    )

print("--- Fitting model: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()

id_test = df_test['id']
y_pred = batch_predict(model, df_test, n_batch=20)
y_pred = np.asarray(y_pred)
print(y_pred)
y_pred[y_pred > 3] = 3
y_pred[y_pred < 1] = 1
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

print("--- Prediction: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()
