# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer   

#settings
eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
color = sns.color_palette()

APPOS = { "aren't" : "are not", "can't" : "cannot", "couldn't" : "could not", "didn't" : "did not", "doesn't" : "does not", "don't" : "do not", "hadn't" : "had not", "hasn't" : "has not", "haven't" : "have not", "he'd" : "he would", "he'll" : "he will", "he's" : "he is", "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am", "isn't" : "is not", "it's" : "it is", "it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not", "shan't" : "shall not", "she'd" : "she would", "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "that's" : "that is", "there's" : "there is", "they'd" : "they would", "they'll" : "they will", "they're" : "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not", "we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is", "what've" : "what have", "where's" : "where is", "who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have", "won't" : "will not", "wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will", "you're" : "you are", "you've" : "you have", "'re": " are","wasn't": "was not", "we'll":" will", "didn't": "did not"
}

def clean(comment):
    comment = comment.lower()
    # remove new line character
    comment=re.sub('\\n','',comment)
    # remove ip addresses
    comment=re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', comment)
    # remove usernames
    comment=re.sub('\[\[.*\]', '', comment)
    # split the comment into words
    words = tokenizer.tokenize(comment)
    # replace that's to that is by looking up the dictionary
    words=[APPOS[word] if word in APPOS else word for word in words]
    # replace variation of a word with its base form
    words=[lem.lemmatize(word, "v") for word in words]
    # eliminate stop words
    words = [w for w in words if not w in eng_stopwords]
    # now we will have only one string containing all the words
    clean_comment=" ".join(words)
    # remove all non alphabetical characters
    clean_comment=re.sub("\W+"," ",clean_comment)
    clean_comment=re.sub("  "," ",clean_comment)
    return (clean_comment)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import Dense, Embedding, LSTM, GRU ,Input ,GlobalMaxPool1D, Dropout, Bidirectional, Activation
from keras.layers.embeddings import Embedding

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("train and test shape:-")
print(train.shape,test.shape)
train_val_x = train.loc[:,'comment_text'].apply(lambda comment: clean(comment))
train_val_y = train.loc[:,'target'].values
print(train.head())
fet = ['severe_toxicity', 'obscene','identity_attack', 'insult', 'threat']
train_fet = train.loc[:,fet]
print(train_fet.shape)
test_val_x = test.loc[:,'comment_text'].apply(lambda comment: clean(comment))

print(train_val_x.shape)
print(train_val_y.shape)
train_yy = []

for i in range(train_val_x.shape[0]):
    train_yy.append(train_val_y[i])
    
print(len(train_yy))

print("tokenizing start")
tokenizer_obj = Tokenizer()
total_comment = np.concatenate((train_val_x, test_val_x))
print(total_comment.shape)
tokenizer_obj.fit_on_texts(total_comment)
max_length = max([len(s.split()) for s in total_comment])
vocab_size = len(tokenizer_obj.word_index)
x_train_token = tokenizer_obj.texts_to_sequences(train_val_x)
x_test_token = tokenizer_obj.texts_to_sequences(test_val_x)
print("padding start")
x_train_pad = pad_sequences(x_train_token, maxlen=max_length,padding="post")
x_test_pad = pad_sequences(x_test_token, maxlen=max_length,padding="post")

print(x_train_pad.shape)

inp = Input(shape=(max_length, )) #maxlen=200 as defined earlier

# size of the vector space
embed_size = 128
x = Embedding(vocab_size, embed_size)(inp)

output_dimention = 60
x =  Bidirectional(LSTM(output_dimention, return_sequences=True,name='lstm_layer'))(x)
# reduce dimention
x = GlobalMaxPool1D()(x)
# disable 10% precent of the nodes
x = Dropout(0.1)(x)
# pass output through a RELU function
x = Dense(128, activation="relu")(x)
# another 10% dropout
x = Dropout(0.1)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
# pass the output through a sigmoid layer, since 
# we are looking for a binary (0,1) classification 
x = Dense(5, activation="sigmoid")(x)



model = Model(inputs=inp, outputs=x)
# we use binary_crossentropy because of binary classification
# optimise loss by Adam optimiser
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("on lstm model")
print(x_train_pad.shape,train_val_y.shape)
model.fit(x_train_pad,train_fet.values,batch_size=4096,epochs=5,validation_split=0.1,verbose=2)


# model1 = Sequential()
# model1.add(Dense(128,input_dim=5))
# model1.add(Activation('relu'))
# model1.add(Dropout(.4))
# model1.add(Dense(512))
# model1.add(Activation('relu'))
# model1.add(Dropout(.4))
# model1.add(Dense(1))
# model1.add(Activation('sigmoid'))
# model1.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
              
# model1.fit(train_fet.values,train_yy,batch_size=32,epochs=8,validation_split=0.1)



prediction = model.predict(x_test_pad,verbose=1,batch_size=2048)
print(prediction.shape)

print(prediction[0])


pred = np.zeros((prediction.shape[0]),dtype=float)

for i in range(prediction.shape[0]):
    pred[i]=max(prediction[i])
    

submission = pd.DataFrame.from_dict({
    'id': test.id,
    'prediction': pred
})
submission.to_csv('submission.csv', index=False)

print(pred.shape)

print(submission.values[12])