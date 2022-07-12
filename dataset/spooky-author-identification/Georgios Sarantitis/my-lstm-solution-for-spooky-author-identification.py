#Import libraries
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
import itertools
import pandas as pd
import numpy as np
import copy

#Import data
print('Importing data sets...')
df_train = pd.read_csv("../input/train.csv")
df_train = df_train.drop(['id'], axis=1)

df_test = pd.read_csv("../input/test.csv")
ids = df_test['id']
df_test = df_test.drop(['id'], axis=1)

#Tokenize sentences
print("Tokenizing in progress...")
#1. Train set
text_list_train = list(df_train['text'])
#text_list_train_lower = [word.lower() for word in text_list_train]
tokenized_text_train = [word_tokenize(i) for i in text_list_train]

#2. Test set
text_list_test = list(df_test['text'])
#text_list_test_lower = [word.lower() for word in text_list_test]
tokenized_text_test = [word_tokenize(i) for i in text_list_test]

#Create vocabulary from train set only
list_of_all_words = list(itertools.chain.from_iterable(tokenized_text_test))
vocabulary =sorted(list(set(list_of_all_words)))

#Remove stopwords (I found out that it makes no difference but you can try on your own)
#vocabulary = [word for word in vocabulary if word not in stopwords.words('english')]

#--------------------------Pre-processing train and test sets----------------------------------
print('Pre-processing Train set...')
tokenized_numbers_train = copy.deepcopy(tokenized_text_train)

i=-1
for list in tokenized_numbers_train:
    i=i+1
    j=-1
    for number in list:
        j = j + 1
        if tokenized_numbers_train[i][j] in vocabulary:
            tokenized_numbers_train[i][j]= vocabulary.index(number)
        else:
            tokenized_numbers_train[i][j] = 0

tokens_train = pd.DataFrame(tokenized_numbers_train, dtype='int32')
tokens_train = tokens_train.fillna(0)
tokens_train = tokens_train.astype(int)

print('Pre-processing Test set...')
tokenized_numbers_test = copy.deepcopy(tokenized_text_test)

i=-1
for list in tokenized_numbers_test:
    i=i+1
    j=-1
    for number in list:
        j = j + 1
        if tokenized_numbers_test[i][j] in vocabulary:
            tokenized_numbers_test[i][j] = vocabulary.index(number)
        else:
            tokenized_numbers_test[i][j] = 0

tokens_test = pd.DataFrame(tokenized_numbers_test, dtype='int32')
tokens_test = tokens_test.fillna(0)
tokens_test = tokens_test.astype(int)

print('Making some more pre-processing to train and test sets...')

#Bring both sets to same shape (Choose how many words to use)
max_words_in_sentence=30

#Shorten or extend Train set to reach selected length
if tokens_train.shape[1]>max_words_in_sentence:
    tokens_train = tokens_train.drop(tokens_train.columns[[range(max_words_in_sentence,tokens_train.shape[1])]], axis=1)
else:
    for col in range(tokens_train.shape[1],max_words_in_sentence):
        tokens_train[col]=0

#Shorten or extend Test set to reach selected length
if tokens_test.shape[1] > max_words_in_sentence:
    tokens_test = tokens_test.drop(tokens_test.columns[[range(max_words_in_sentence, tokens_test.shape[1])]],
                                     axis=1)
else:
    for col in range(tokens_test.shape[1], max_words_in_sentence):
        tokens_test[col] = 0

#------------------------------End of Pre-processing----------------------------------------------------

#Define train and Test sets
train_x = np.array(tokens_train)
train_y = np.array(df_train['author'])

test_x = np.array(tokens_test)

encoder1 = LabelEncoder()
encoder1.fit(train_y)
encoded_train_Y = encoder1.transform(train_y)
dummy_train_y = np_utils.to_categorical(encoded_train_Y)
dummy_train_y.astype(int)

l=len(vocabulary)+1
inp=train_x.shape[1]

#Build an LSTM model
model = Sequential()
model.add(Embedding(l, 64,input_length=inp))
model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.1))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(train_x, dummy_train_y, epochs=3, batch_size=16, verbose=2)

# Predict and write to file
results = model.predict(test_x)
results = pd.DataFrame(results, columns=['EAP', 'HPL','MWS'])
results.insert(0, "id", ids)
results.to_csv("my_submission.csv", index=False)
