# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from array import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, CuDNNLSTM, GRU, CuDNNGRU, Conv1D, MaxPool1D, Dropout, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
gloveFile = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

dictionary = {"'cause": 'because',
 "'s": 'is',
 "'tis": 'it is',
 "'twas": 'it was',
 "I'd": 'I had',
 "I'll": 'I shall',
 "I'm": 'I am',
 "I'm'a": 'I am about to',
 "I'm'o": 'I am going to',
 "I've": 'I have',
 "ain't": 'am not',
 "amn't": 'am not',
 "aren't": 'are not',
 "cain't": 'cannot',
 "can't": 'cannot',
 "could've": 'could have',
 "couldn't": 'could not',
 "couldn't've": 'could not have',
 "daren't": 'dare not',
 "daresn't": 'dare not',
 "dasn't": 'dare not',
 "didn't": 'did not',
 "doesn't": 'does not',
 "don't": 'do not',
 "e'er": 'ever',
 "everyone's": 'everyone is',
 'finna': 'fixing to',
 'gimme': 'give me',
 "gon't": 'go not',
 'gonna': 'going to',
 'gotta': 'got to',
 "hadn't": 'had not',
 "hasn't": 'has not',
 "haven't": 'have not',
 "he'd": 'he had',
 "he'll": 'he shall',
 "he's": 'he has',
 "he've": 'he have',
 "how'd": 'how did',
 "how'll": 'how will',
 "how're": 'how are',
 "how's": 'how has',
 "isn't": 'is not',
 "it'd": 'it would',
 "it'll": 'it shall',
 "it's": 'it has',
 "let's": 'let us',
 "may've": 'may have',
 "mayn't": 'may not',
 "might've": 'might have',
 "mightn't": 'might not',
 "must've": 'must have',
 "mustn't": 'must not',
 "mustn't've": 'must not have',
 "ne'er": 'never',
 "needn't": 'need not',
 "o'clock": 'of the clock',
 "o'er": 'over',
 "ol'": 'old',
 "oughtn't": 'ought not',
 'rarely': 'cannot',
 "shalln't": 'shall not',
 "shan't": 'shall not',
 "she'd": 'she had',
 "she'll": 'she shall',
 "she's": 'she has',
 "should've": 'should have',
 "shouldn't": 'should not',
 "shouldn't've": 'should not have',
 "so're": 'so are',
 "somebody's": 'somebody has',
 "someone's": 'someone has',
 "something's": 'something has',
 "that'd": 'that would',
 "that'll": 'that shall',
 "that're": 'that are',
 "that's": 'that has',
 "there'd": 'there had',
 "there'll": 'there shall',
 "there're": 'there are',
 "there's": 'there has',
 "these're": 'these are',
 "they'd": 'they had',
 "they'll": 'they shall',
 "they're": 'they are',
 "they've": 'they have',
 "this's": 'this has',
 "those're": 'those are',
 "wasn't": 'was not',
 "we'd": 'we had',
 "we'd've": 'we would have',
 "we'll": 'we will',
 "we're": 'we are',
 "we've": 'we have',
 "weren't": 'were not',
 "what'd": 'what did',
 "what'll": 'what shall',
 "what're": 'what are',
 "what's": 'what has',
 "what've": 'what have',
 "when's": 'when has',
 "where'd": 'where did',
 "where're": 'where are',
 "where's": 'where has',
 "where've": 'where have',
 "which's": 'which has',
 "who'd": 'who would',
 "who'd've": 'who would have',
 "who'll": 'who shall',
 "who're": 'who are',
 "who's": 'who has',
 "who've": 'who have',
 "why'd": 'why did',
 "why're": 'why are',
 "why's": 'why has',
 "won't": 'will not',
 "would've": 'would have',
 "wouldn't": 'would not',
 "y'all": 'you all',
 "you'd": 'you had',
 "you'll": 'you shall',
 "you're": 'you are',
 "you've": 'you have'}

# Review the questions:

def mappingWords(questions,dictionary):
    return " ".join([dictionary.get(w,w) for w in questions.split()])

def review_questions(questions):
  questions = mappingWords(questions, dictionary)
  questions = re.sub(r"[^a-zA-Z0-9 ]", " ", questions)
  questions = re.sub(r'[0-9]+', "Number", questions)
  return questions


def loadGloveModel(gloveFile):
  print("Loading Glove Model")
  f = open(gloveFile,'r', encoding='utf8')
  embedding_index = {}
  print("Opened!")
  for line in f:
    splitLine = line.split(' ')
    word = splitLine[0]
    embedding = np.asarray(splitLine[1:], dtype='float32')
    embedding_index[word] = embedding
  print("Done.",len(embedding_index)," words loaded!")
  return embedding_index





print("reviewing training data")
# Review training data
question_list = []
for i in range(len(train_data["question_text"])):
  question_list.append(review_questions(train_data["question_text"][i]))

print("reviewing testing data")
# Review test data
question_list_test = []
for i in range(len(test_data["question_text"])):
  question_list_test.append(review_questions(test_data["question_text"][i]))

X = question_list 
y = train_data["target"]
X_test = question_list_test

print("getting embedding vectors!")
# Get embedding vector

embedding_index = loadGloveModel(gloveFile)

# split the data

data = [X, y]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.05)

print("Generating tokens")
# Generate the text sequence for RNN model
np.random.seed(1000)
NUM_MOST_FREQ_WORDS_TO_INCLUDE = 100000
MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 80           # Input for keras.
embedding_vector_length = 32

all_quesion = X_train + X_val

tokenizer = Tokenizer(num_words = NUM_MOST_FREQ_WORDS_TO_INCLUDE)
tokenizer.fit_on_texts(all_quesion)

word_index = tokenizer.word_index

#tokenising train data
train_question_tokenized = tokenizer.texts_to_sequences(X_train)      
X_train = pad_sequences(train_question_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)          # len(X_train) x 50

#tokenising validation data
val_question_tokenized = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(val_question_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)               # len(X_val) X 50 

#tokenizing test data
test_question_tokenized = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(test_question_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)

print("generating embedding matrix")
# Now, we need to create embedding matrix.
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embedding_index.get(word)
  if embedding_vector is not None:
    # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector


print("training model")
# Model with 1 dense layer.
import tensorflow.keras.backend as K
import time

dense_layers = [1]
filter_sizes = [128]
conv_layers = [1]

# Clearing tensorflow session right before creating the model
K.clear_session()

for dense in dense_layers:
  for filters in filter_sizes:
    for conv in conv_layers:

      NAME = "Model_GRU_CONV_{}-conv-{}-layer-{}-dense".format(conv, filters, dense)
      print(NAME)

      model = Sequential()

      model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
          input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN, trainable=False) )

      model.add(Dropout(0.2))

      model.add(Conv1D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu'))
      model.add(MaxPool1D(pool_size = 2))

      for l in range(conv - 1):
        model.add(Conv1D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(MaxPool1D(pool_size = 2))

      model.add( CuDNNGRU(128, return_sequences=True))
      model.add(Dropout(0.2))

      model.add( CuDNNGRU(64, return_sequences=False))
      model.add(Dropout(0.2))

      model.add(Flatten())

      for n in range(dense):
        model.add(Dense(filters, activation = "relu"))

      model.add(Dense(1, activation = "sigmoid"))
      
      model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

      model.summary()
      model.fit(X_train, y_train, batch_size = 32, epochs = 1, validation_data = [X_val, y_val] )

      model.save("{}.h5".format(NAME))
      
      K.clear_session()
'''
mymodel = load_model("Model_GRU_CONV_1-conv-128-layer-1-dense.h5")
y_test_prediction = mymodel.predict(X_test)
y_test_prediction_list = y_test_prediction.tolist()

y_test = []
for i in range(len(test_data)):
	y_test.append(int(round(y_test_prediction_list[i][0])))

print("creainting submission file")
output = pd.DataFrame(data = { "qid" : test_data["qid"], "prediction" : y_test} )
output.to_csv("submission.csv", index = False)
'''

##############

os.chdir("/kaggle/working")
if os.path.exists("submission.csv"):
    os.remove("submission.csv")

from array import *

NAME1 = "Model_LSTM_CONV_1-conv-64-layer-1-dense.h5"
NAME2 = "Model_GRU_CONV_1-conv-128-layer-1-dense.h5"
uploadmodel = load_model(NAME2)
y_test_prediction = uploadmodel.predict(X_test)
    
threshold_GRU = 0.299999
y_test = (y_test_prediction > threshold_GRU).astype(int)

print("creainting submission file")
output = pd.DataFrame(data = { "qid" : test_data["qid"]} )
output["prediction"] = y_test
output.to_csv("submission.csv", index = False)


