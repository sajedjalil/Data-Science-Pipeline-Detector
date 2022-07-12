# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np
import pandas as pd
import time
from collections import Counter
import re
import math
from scipy import sparse
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

def make_model(vec_len):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=vec_len))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="msle", optimizer=optimizers.Adam())
    return model

########################################################################################
########################################################################################

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
# path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"
path = "/home/lisa/Documents/leren_beslissen/Merri/"
path = "../input/"


os.chdir(path)

start = time.time()

def fill_missing_items(data):
    data["brand_name"].fillna("missing", inplace=True)
    data["item_description"].fillna("missing", inplace=True)
    data["category_name"].fillna("missing", inplace=True)


def edit_description_column(text):
    words = re.sub("[^a-z0-9 .]", "", text.lower())
    words = re.sub(r'(?<!\d)\.(?!\d)', '', words)
    return [word for word in words.split() if len(word) > 3]


def lower_string(text):
    return text.lower()


def edit_data(data):
    fill_missing_items(data)
    data["item_description"] = data["item_description"].apply(edit_description_column)
    data["category_name"] = data["category_name"].apply(lower_string)
    data["category_name"] = data["category_name"].str.split("/")
    data["brand_name"] = data["brand_name"].apply(lower_string)


def make_dictionaries(data):
    words_list = []
    description_list = data["item_description"].tolist()
    for sen in description_list:
        for word in sen:
            words_list.append(word)
    count = Counter(words_list)
    des_list = list(set([w for w in words_list if count[w] > 20]))
    description_dict = {k: v for v, k in enumerate(des_list)}

    categories = []
    for cats in data["category_name"]:
        if cats == cats:
            categories += cats
    categories = list(set(categories))
    category_dict = {k: v for v, k in enumerate(categories)}

    brand_names = []
    for brand in data["brand_name"]:
        brand_names.append(brand)
    count = Counter(brand_names)
    brand_list = set(list([b for b in brand_names if count[b] > 2]))
    brand_dict = {k: v for v, k in enumerate(brand_list)}
    return description_dict, category_dict, brand_dict


def get_price_list(data):
    return np.array(data["price"].tolist())


# Replaces missing brand names with brand from item_description
def replace_missing_brands(data, brand_dict):
    for index, row in data.iterrows():
        if row["brand_name"] == 'missing':
            for word in row["item_description"]:
                if word in brand_dict:
                    data.at[data.index[index], "brand_name"] = word


def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))


def make_sparse_matrix(data, description_dict, categories_dict, brand_dict):
    sparse_matrix = sparse.lil_matrix((data.shape[0], len(description_dict) + len(categories_dict) + len(brand_dict) + 6), dtype=bool)
    des_len, cat_len, brand_len = len(description_dict), len(categories_dict), len(brand_dict)

    descriptions_list = data["item_description"].tolist()
    for i, sen in enumerate(descriptions_list):
        for word in sen:
            if word in description_dict:
                sparse_matrix[i, description_dict[word]] = True

    categories_list = data["category_name"].tolist()
    for i, categories in enumerate(categories_list):
        for category in categories:
            if category in categories_dict:
                sparse_matrix[i, categories_dict[category] + des_len] = True

    brand_list = data["brand_name"].tolist()
    for i, brand in enumerate(brand_list):
        if brand in brand_dict:
            sparse_matrix[i, brand_dict[brand] + des_len + cat_len] = True
    condition_list = data["item_condition_id"].tolist()
    for i, condition in enumerate(condition_list):
        sparse_matrix[i, des_len + cat_len + brand_len + condition - 1] = True
    shipping_list = data["shipping"].tolist()
    for i, shipping in enumerate(shipping_list):
        sparse_matrix[i, des_len + cat_len + brand_len + 5] = shipping
    return sparse_matrix
    
def preprocess_training(train, start):
    print("train size:", train.shape)
    edit_data(train)
    print("edited all the data", time.time() - start)
    description_dict, categories_dict, brand_dict = make_dictionaries(train)
    print("made dictionaries", time.time() - start, " \n dict lengths:",
          len(description_dict), len(categories_dict), len(brand_dict))
    sparse_matrix = make_sparse_matrix(train, description_dict, categories_dict, brand_dict)
    print("made sparse matrix:", time.time() - start)
    prices = get_price_list(train)
    print("made prices", time.time() - start)
    vec_len = len(description_dict) + len(categories_dict) + len(brand_dict) + 6
    return sparse_matrix, prices, vec_len, [description_dict, categories_dict, brand_dict]

def preprocess_test(test, dicts, start):
    print("test size:", test.shape)
    edit_data(test)
    print("edited all the test data", time.time() - start)
    sparse_matrix = make_sparse_matrix(test, dicts[0], dicts[1], dicts[2])
    print("made sparse test matrix:", time.time() - start)
    return sparse_matrix
    
def get_rows(mat, price, rng):
    return mat[rng].todense(), price[rng]

def iter_minibatches(mat, price, chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < mat.shape[0]:
        if (mat.shape[0] - chunkstartmarker) < chunksize:
            chunksize = mat.shape[0] - chunkstartmarker
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk, y_chunk = get_rows(mat, price, chunkrows)
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize

os.chdir(path)

start = time.time()

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] < 300]
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
train = train.loc[0:1000000]
start = start
print(start)

sparse_matrix, prices, vec_len, dicts = preprocess_training(train, start)
training_data, training_prices = get_rows(sparse_matrix, prices, range(0, 900000))
validation_data, test_prices = get_rows(sparse_matrix, prices, range(900000,1000000))
ridge_model = Ridge(alpha=.75, fit_intercept=True, normalize=False,
      copy_X=True, max_iter=None, tol=0.01, solver='auto', random_state=100)
print("fitting ridge model...", time.time() - start)
ridge_model.fit(training_data, training_prices)
print("model fitted")
print(prices.shape)

#sparse_test_matrix = preprocess_test(test, dicts, start)
print("sparse test matrix made")
predicted_prices = ridge_model.predict(validation_data)
predicted_prices = np.maximum(predicted_prices, 0)
print(prices.shape)
print(predicted_prices.shape)
print("The score is:", calc_score(prices, predicted_prices))


# Any results you write to the current directory are saved as output.