import os, time, re, math
import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

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

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(pre_price+1) - math.log(price+1))**2
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
    
def fit_neuralnet(neuralnet, sparse_matrix, prices, batchsize, amount_batches=1):
    valbatch = sparse_matrix[amount_batches * batchsize: amount_batches * batchsize + 10000].todense()
    #valbatchprices = prices[amount_batches * batchsize: amount_batches * batchsize + 10000]
    for t in range(amount_batches):
        print("this is batch", t)
        batch = sparse_matrix[t * batchsize:t * batchsize + batchsize].todense()
        batchprices = prices[t * batchsize:t * batchsize + batchsize]
        neuralnet.fit(batch, batchprices, batch_size=200)
        #predicted_prices = neuralnet.predict(valbatch)
        #print("The validation score is:", calc_score(valbatchprices, predicted_prices))
        
def predict_neuralnet(neuralnet, sparse_matrix, batchsize, test_len):
    batches = np.array([])
    count = 0
    t = 0 
    change = 0
    addedbatchsize = batchsize
    while True:
        if (test_len - count) < batchsize:
            addedbatchsize = test_len - count
            change = 1
        batch = sparse_matrix[t * batchsize:t * batchsize + addedbatchsize].todense()
        batchprices = neuralnet.predict(batch)
        batches = np.append(batches, batchprices)
        count += batchsize
        print(t)
        t += 1
        if change == 1:
            break
    return np.array(batches)
    

def make_model(vec_len):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=vec_len))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="msle", optimizer=optimizers.Adam())
    return model
    
########################################################################################
########################################################################################

start = time.time()
train = pd.read_csv('../input/train.tsv', delimiter='\t', encoding='utf-8')
test = pd.read_csv('../input/test.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] < 300]
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
#train = train.loc[0:500000]

sparse_matrix, prices, vec_len, dicts = preprocess_training(train, start)
neuralnet = make_model(vec_len)
print("neural net set up:", time.time() - start)

fit_neuralnet(neuralnet, sparse_matrix, prices, 5000, amount_batches=200)
print("neural net fitted:", time.time() - start)

sparse_test_matrix = preprocess_test(test, dicts, start)
print("sparse test matrix made")
predicted_prices = predict_neuralnet(neuralnet, sparse_test_matrix, 100000, len(test))

print("predicted prices:", len(predicted_prices))
print(len(test[["test_id"]]))
submission = test[["test_id"]]
submission["price"] = predicted_prices
submission.to_csv("myNNsubmission.csv", index=False)