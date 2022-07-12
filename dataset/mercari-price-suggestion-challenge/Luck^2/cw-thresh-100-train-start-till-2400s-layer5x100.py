""" This program trains a regressor on the dataset. Running MLPRegressor
requires a sklearn version of 0.18 or higher.
    This program expects a file that cPickle can open at location PATH_TRAIN
cPickle needs to load a 2 dimensional numpy array, each row is a datapoint and
the last element of each row should be the label."""
from __future__ import print_function
from time import time
from sklearn.neural_network import MLPRegressor
import numpy as np
import csv as csv

start_time = time()

""" This program makes the dataset usable for machine learning."""
import re
from collections import Counter
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

PATH_TRAIN = "../input/mercari-price/train.tsv"
PATH_TEST = "../input/mercari-price/test.tsv"
BATCH_SIZE = 2000  # Don't make this number much larger than 100000
DATA_SIZE = sum(1 for line in open(PATH_TRAIN))  # Number of \n in train file
TEST_DATA_SIZE = sum(1 for line in open(PATH_TRAIN)) #Number of \n in train file
THRESH = 100  # How often needs a word to occur before it is used
TRAIN_RATIO = 1.0  # The ratio train validation

# The following constants are the columns in each "data" variable
COLUMN_NAME = 0
COLUMN_CONDITION = 1
COLUMN_CATEGORY = 2
COLUMN_BRAND = 3
COLUMN_PRICE = 4
COLUMN_SHIPPING = 5
COLUMN_DESCRIPTION = 6

TEST_COLUMN_ID = 0
TEST_COLUMN_NAME = 1
TEST_COLUMN_CONDITION = 2
TEST_COLUMN_CATEGORY = 3
TEST_COLUMN_BRAND = 4
TEST_COLUMN_SHIPPING = 5
TEST_COLUMN_DESCRIPTION = 6


# These characters will be ignored by the neural net when training data
IGN_CHAR = [',', ':', ';', '.', '(', ')', '\'', '"', '!', '?', '*', '&', '^']

# The output files
PATH_INPUT_TRAIN = "../input/processedmercaridata/input_train_100.tsv"
PATH_INPUT_SPEC = "../input/processedmercaridata/input_specs_100.tsv"
PATH_INPUT_TEST = "../input/processedmercaridata/input_test_100.tsv"
PATH_TEST_OUT = "test_thresh100_out_.csv"



with open(PATH_TEST_OUT, "a", newline='\n') as out_file:
    out_file.write("test_id" + "," + "price" + "\n")

def categorical_train_data(data):
    """Return a vectorized form of the category column"""
    cats = []
    for i in range(data.shape[0]):
        row = data[i]
        string = row[COLUMN_CATEGORY]
        string = re.compile(r'[^\s\w_]+').sub(' ',
                                              string)  # removes all non-alfanumeric characters
        string = string.lower()
        s = set(stopwords.words('english'))
        cat_arr = list(filter(lambda w: not w in s, string.split("\\")))
        # cat_arr = filter_words(cat_arr)
        cats.append(cat_arr)
    return np.array(cats)

def categorical_test_data(data):
    """Return a vectorized form of the category column"""
    cats = []
    for i in range(data.shape[0]):
        row = data[i]
        string = row[TEST_COLUMN_CATEGORY]
        string = re.compile(r'[^\s\w_]+').sub(' ',
                                              string)  # removes all non-alfanumeric characters
        string = string.lower()
        s = set(stopwords.words('english'))
        cat_arr = list(filter(lambda w: not w in s, string.split()))
        # cat_arr = filter_words(cat_arr)
        cats.append(cat_arr)
    return np.array(cats)


def count_words(words, word_count):
    """Keep track of word occurance in a counter."""
    for row in words:
        for word in row:
            word_count[word] += 1
    return word_count


def data_to_input(num_data, words, cats, labels, word_map, cat_map):
    """Return a version of the data that a neural network can take as input."""
    in_vec = []
    for i, word_row in enumerate(words):
        vec = ""

        for word in word_row:
            if word in word_map:
                vec += str(word_map[word]) + '\t'
        vec = vec[:-1]
        vec += ','

        for cat in cats[i]:
            if cat in cat_map:
                vec += str(cat_map[cat]) + '\t'
        vec = vec[:-1]
        vec += ','

        for j, _ in enumerate(num_data[i]):
            vec += str(num_data[i][j]) + ','
        vec += labels[i]

        in_vec.append(vec + '\n')
    return np.array(in_vec)

def test_to_input(num_data, words, cats, IDs, word_map, cat_map):
    """Return a version of the data that a neural network can take as input."""
    in_vec = []
    for i, word_row in enumerate(words):
        vec = ""

        for word in word_row:
            if word in word_map:
                vec += str(word_map[word]) + '\t'
        vec = vec[:-1]
        vec += ','

        for cat in cats[i]:
            if cat in cat_map:
                vec += str(cat_map[cat]) + '\t'
        vec = vec[:-1]
        vec += ','

        for j, _ in enumerate(num_data[i]):
            vec += str(num_data[i][j]) + ','
        vec += IDs[i]

        in_vec.append(vec + '\n')
    return np.array(in_vec)



def filter_words(word_arr):
    """Takes a array of words and removes some of them, then returns the
    smaller array."""
    word_arr = word_arr[word_arr != '']
    # This is a very ugly loop, if the nltk works, please remove it.
    for word in word_arr:
        if len(word) < 3:  # I chose 3 as an arbitrary length
            word_arr = word_arr[word_arr != word]
    return word_arr


def labels_from_data(data):
    """Return an array with the prizes matching the rows"""
    labels = data[:, COLUMN_PRICE]
    return labels

def ID_from_test_data(data):
    IDs = data[:, TEST_COLUMN_ID]
    return IDs


def load_train_data(start, size):
    """Return the trainingdata as an array."""
    data = []
    stop = False

    with open(PATH_TRAIN) as f:

        i = 0
        for row in f:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            row = row[:-1]
            row = row.split("\t")
            row = row[1:]
            data.append(np.array(row))
            i += 1

        if i < start + size:
            stop = True

    return stop, np.array(data)

def load_test_data(start, size):
    """Return the trainingdata as an array."""
    data = []
    stop = False

    with open(PATH_TEST) as f:

        i = 0
        for row in f:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            row = row[:-1]
            row = row.split("\t")
            data.append(np.array(row))
            i += 1

        if i < start + size:
            stop = True

    return stop, np.array(data)



def show_counter(word_count, thresh):
    """Count how many words occur at least 'thresh' times."""
    vec_len = 0
    avoid_len = 0

    for word in word_count.keys():
        if word_count[word] > thresh:
            vec_len += 1
        else:
            avoid_len += 1

    return vec_len, avoid_len


def make_maps():
    """Create a word map that assigns unique integers to words."""
    batch = 1
    pointer = 1
    word_count = Counter()
    cat_count = Counter()

    print("Making a wordmap:")
    stop = False
    while not stop:
        print("\nNow batch", batch, "currenlty",
              show_counter(word_count, THRESH), "words...")
        stop, data = load_train_data(pointer, BATCH_SIZE)
        print("... Substracting the words ...")
        words = words_from_train_data(data)
        print("... Substracting the categories ...")
        cats = categorical_train_data(data)
        print("... Counting")
        word_count = count_words(words, word_count)
        cat_count = count_words(cats, cat_count)

        batch += 1
        pointer += BATCH_SIZE

    print("\nDone!\n... Mapping words ...")
    word_map = map_words(word_count, dict())
    cat_map = map_words(cat_count, dict())
    print("... Currently", len(word_map.keys()), "words are in the map...")
    print("... And", len(cat_map.keys()), "categories are in the map")

    return word_map, cat_map


def map_words(word_count, word_map, thresh=THRESH):
    """Gives each word in the array a unique int in a dictionary"""
    n_words = 0
    for word in word_count.keys():
        if word_count[word] > thresh:
            word_map[word] = n_words
            n_words += 1
    return word_map


def numeric_train_data(data):
    """Return an array of floats for condition and shipping"""
    conditions = data[:, COLUMN_CONDITION]
    shipping = data[:, COLUMN_SHIPPING]
    num_data = np.array([conditions, shipping])
    return num_data.T

def numeric_test_data(data):
    """Return an array of floats for condition and shipping"""
    conditions = data[:, TEST_COLUMN_CONDITION]
    shipping = data[:, TEST_COLUMN_SHIPPING]
    num_data = np.array([conditions, shipping])
    return num_data.T


def split_train_val(data, ratio):
    """Return a set for training and a set for validation"""
    np.random.shuffle(data)
    index = int(len(data) * ratio)
    train = data[:index]
    val = data[index:]
    return train, val


def store(arr, path):
    """Store an array in a file."""
    with open(path, "a") as vec_file:
        for row in arr:
            vec_file.write(row)


def store_submission(predictions, IDs, path):
    """Store an array in a file."""
    with open(path, "a", newline='\n') as vec_file:
        for i in np.append(IDs, predictions, axis=0).T:
            print(i)
            vec_file.write(str(int(i[0])) + "," + str(i[1]) + "\n")




def store_specs(path, word_len, cat_len):
    """Store important information about the train and val files."""
    with open(path, 'w') as f:
        f.write(str(word_len) + '\t' + str(cat_len))


def words_from_train_data(data):
    """Gives each word in the array a unique int in a dictionary"""
    words = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[COLUMN_NAME] + ' ' + row[COLUMN_BRAND] + ' ' +
                  row[COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ',
                                              string)  # removes all non-alfanumeric characters
        string = string.lower()
        s = set(stopwords.words('english'))
        word_arr = list(filter(lambda w: not w in s, string.split()))
        # word_arr = filter_words(word_arr)
        words.append(word_arr)
    return np.array(words)

def words_from_test_data(data):
    """Gives each word in the array a unique int in a dictionary"""
    words = []
    for i in range(data.shape[0]):
        row = data[i]
        string = (row[TEST_COLUMN_NAME] + ' ' + row[TEST_COLUMN_BRAND] + ' ' +
                  row[TEST_COLUMN_DESCRIPTION])
        string = re.compile(r'[^\s\w_]+').sub(' ',
                                              string)  # removes all non-alfanumeric characters
        string = string.lower()
        s = set(stopwords.words('english'))
        word_arr = list(filter(lambda w: not w in s, string.split()))
        # word_arr = filter_words(word_arr)
        words.append(word_arr)
    return np.array(words)


def main_pre_processing():
    """The main function of the program"""
    word_map, cat_map = make_maps()
    store_specs(PATH_INPUT_SPEC, len(word_map), len(cat_map))
    batch = 1
    pointer = 1

    print("\n\nStoring the input data")
    print("Batch size =", BATCH_SIZE, "Data size =", DATA_SIZE)

    stop = False
    while not stop:
        """ preparing NN train input """
        print("\nNow loading batch", batch, "...")
        stop, data = load_train_data(pointer, BATCH_SIZE)

        print("... converting to input ...")
        num_train = numeric_train_data(data)
        words_train = words_from_train_data(data)
        cats_train = categorical_train_data(data)
        labels_train = labels_from_data(data)
        in_train = data_to_input(num_train, words_train, cats_train,
                                 labels_train, word_map, cat_map)

        print("... storing")
        store(in_train, PATH_INPUT_TRAIN)

        batch += 1
        pointer += BATCH_SIZE

    batch = 1
    pointer = 1

    print("\n\nStoring the input test data")
    print("Batch size =", BATCH_SIZE, "Data size =", TEST_DATA_SIZE)

    stop = False

    while not stop:
        """ preparing NN test input """
        print("\nNow loading batch", batch, "...")
        stop, data = load_test_data(pointer, BATCH_SIZE)

        print("... converting to input ...")
        num_test = numeric_test_data(data)
        words_test = words_from_test_data(data)
        cats_test = categorical_test_data(data)
        ID_test = ID_from_test_data(data)
        in_test = test_to_input(num_test, words_test, cats_test,
                               ID_test, word_map, cat_map)

        print("... storing")
        store(in_test, PATH_INPUT_TEST)

        batch += 1
        pointer += BATCH_SIZE


#main_pre_processing()
#VEC_LEN = 100  # This variable is used in simple_load



# DELTA_VAL_SIZE = sum(1 for line in open(PATH_DELTA_VAL))
# DELTA_TRAIN_SIZE = sum(1 for line in open(PATH_DELTA_TRAIN))
# VALIDATION_SIZE = sum(1 for line in open(PATH_VALIDATION))

MAX_TRIES = 5
CONVERGENCE = 0.01
TRAIN_SIZE = sum(1 for line in open(PATH_TRAIN))


def load_train(path, start, size):
    """Return the trainingdata as an array"""
    words_len, cats_len = load_specs(PATH_INPUT_SPEC)
    vec_len = words_len + cats_len + 2
    in_vecs = []
    labels = []
    with open(path) as in_file:
        i = 0
        for l in in_file:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            vec = np.zeros(vec_len)
            l = l.split(',')

            words = l[0].split('\t')
            for word in words:
                if word == '':  # Sometimes a word is empty.
                    continue
                vec[int(word)] = 1.0

            cats = l[1].split('\t')
            for cat in cats:
                if cat == '':  # Sometimes a category is empty.
                    break
                vec[int(cat) + words_len] = 1.0

            nums = l[-3:-1]
            for j, num in enumerate(nums):
                vec[j + words_len + cats_len] = float(num)
            label = float(l[-1][:-1])

            in_vecs.append(vec)
            labels.append(label)
            i += 1
    return np.array(in_vecs), np.array(labels)


def load_test(path, start, size):
    """Return the trainingdata as an array"""
    words_len, cats_len = load_specs(PATH_INPUT_SPEC)
    vec_len = words_len + cats_len + 2
    in_vecs = []
    IDs = []
    with open(path) as in_file:
        i = 0
        for l in in_file:
            if i < start:
                i += 1
                continue
            elif i >= start + size:
                break

            vec = np.zeros(vec_len)
            l = l.split(',')

            words = l[0].split('\t')
            for word in words:
                if word == '':  # Sometimes a word is empty.
                    continue
                vec[int(word)] = 1.0

            cats = l[1].split('\t')
            for cat in cats:
                if cat == '':  # Sometimes a category is empty.
                    break
                vec[int(cat) + words_len] = 1.0

            nums = l[-3:-1]
            for j, num in enumerate(nums):
                vec[j + words_len + cats_len] = float(num)
            label = float(l[-1][:-1])
            
            i += 1
            in_vecs.append(vec)
            IDs.append(label)
    return np.array(in_vecs), np.array(IDs)


#Todo: write testfile variant.
def load_simple_train(path, start, size):
    """This load function is used for the delta_words.py"""
    in_vecs = []
    labels = []
    with open(path, 'rb') as in_file:

        for i, row in enumerate(in_file):
            if i < start:
                continue
            elif i >= start + size:
                break

            row = row[:-1]
            row = row.split('\t')
            label = row[-1]
            row = row[:-1]

            in_vecs.append(np.array(row, dtype="float"))
            labels.append(float(label))
    return np.array(in_vecs), np.array(labels)


#Todo: write testfile variant.
def load_simple_val(path, batch_size, file_size):
    """Return some random rows from the validation file made by delta_words."""
    if batch_size > file_size:
        batch_size = file_size

    indexes = np.arange(file_size)
    np.random.shuffle(indexes)
    indexes = set(indexes[:batch_size])

    in_vecs = []
    labels = []
    with open(path) as f:
        for i, row in enumerate(f):
            if i not in indexes:
                continue

            row = row[:-1]
            row = row.split('\t')
            label = row[-1]
            row = row[:-1]

            in_vecs.append(np.array(row, dtype="float"))
            labels.append(float(label))
    return np.array(in_vecs), np.array(labels)


def load_specs(path):
    """Return the number of words and number of categories in the file."""
    words_len = 0
    cats_len = 0
    with open(path) as f:
        l = f.readline()
        l = l.split('\t')
        words_len = int(l[0])
        cats_len = int(l[1])
    return words_len, cats_len

#Todo rewrite so it loads & predicts from testfile input
def load_val(path, batch_size, file_size):
    """Return some random rows from the validation file."""
    if batch_size > file_size:
        batch_size = file_size

    indexes = np.arange(file_size)
    np.random.shuffle(indexes)
    indexes = set(indexes[:batch_size])
    words_len, cats_len = load_specs(PATH_INPUT_SPEC)
    vec_len = words_len + cats_len + 2

    val_data = []
    val_labels = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i not in indexes:
                continue

            vec = np.zeros(vec_len)
            l = l.split(',')

            words = l[0].split('\t')
            for word in words:
                if word == '':  # Sometimes a word is empty.
                    continue
                vec[int(word)] = 1.0

            cats = l[1].split('\t')
            for cat in cats:
                if cat == '':  # Sometimes a category is empty.
                    break
                vec[int(cat) + words_len] = 1.0

            nums = l[-3:-1]
            for j, num in enumerate(nums):
                vec[j + words_len + cats_len] = float(num)

            label = float(l[-1][:-1])

            val_data.append(vec)
            val_labels.append(label)
    return np.array(val_data), np.array(val_labels)


def test_regressor(regr, data, labels):
    """Return the average squared error of the regression neural network"""
    pred = regr.predict(data)
    return np.sum(np.power(np.log(pred+1) - np.log(labels+1), 2)) / len(labels)


def main(layers):
    """The main function of the program"""
    num_batches = int(TRAIN_SIZE / BATCH_SIZE)
    num_test_batches = int(TEST_DATA_SIZE / BATCH_SIZE)
    #  num_batches = int(DELTA_TRAIN_SIZE / BATCH_SIZE)
    print("The program will run in", num_batches)

    past_err = 999999
    new_err = 0
    i = 0
    #Todo: still need to change name when using other algorithm

    regr = MLPRegressor(hidden_layer_sizes=layers)
    
    while time() - start_time < 2400.0:
        for batch in range(num_batches):
            print("\nCurrent loop =", i + 1)
            print("Current layers:", layers)
            print("Starting batch", batch + 1, "of", num_batches)

            print("Loading data...")
            train_data, train_labels = load_train(PATH_INPUT_TRAIN, batch * BATCH_SIZE, BATCH_SIZE)
            # train_data, train_labels = load_simple_train(PATH_DELTA_TRAIN,batch * BATCH_SIZE,BATCH_SIZE)

            print("Training neural network...")
            regr.partial_fit(train_data, train_labels)
            i += 1
        print("\nTesting the network...")
        #Todo write function to test the neural network
        #val_data, val_labels = load_val(PATH_VALIDATION, BATCH_SIZE, VAL_SIZE)
        # val_data, val_labels = load_simple_val(PATH_DELTA_VAL, BATCH_SIZE,DELTA_VAL_SIZE)


        # store(new_err, PATH_OUTPUT)
        # store(new_err, PATH_DELTA_OUTPUT)
    for batch in range(num_test_batches):
        print("Current layers:", layers)
        print("Starting batch", batch + 1, "of", num_test_batches)

        print("Loading test data...")
        data, IDs = load_test(PATH_INPUT_TEST, batch * BATCH_SIZE, BATCH_SIZE)
        try:
            prediction = regr.predict(data)
        except ValueError:
            break
        
        tru_IDs = np.expand_dims(IDs, axis=0)
        tru_prediction = np.expand_dims(prediction, axis=0)
        
        #print(prediction)
        #print(IDs)
        store_submission(tru_prediction, tru_IDs, PATH_TEST_OUT)
        



    print("Done!\nThe regression has an average error of:", np.sqrt(new_err))




main(layers=[100,100,100,100,100])