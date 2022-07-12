# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os


# Any results you write to the current directory are saved as output.

import pandas as pd
import re
import numpy as np

train=pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

test=pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

df=train["comment_text"]
Y=train["target"]


test0=test["comment_text"]
test_ids=test["id"]

import unicodedata
from unidecode import unidecode

def deEmojify(inputString):
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                     returnString += "[" + unicodedata.name(character) + "]"
                except ValueError:
                     returnString += "[x]"

    return returnString

text_train1=df.apply(deEmojify)

test1=test0.apply(deEmojify)


from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def cleanText(inputString):
	review=re.sub(r"http\S+",'' , inputString)
	review=re.sub(r'\W',' ',review) # remove punchations 
	review=review.lower()
	review=re.sub(r'\s+[a-z]\s+',' ',review) # remove single characters which have space in starting and end of the characters
	review=re.sub(r'^[a-z]\s+',' ',review) # remnove single characters which have at starting position of the sentence 
	review=re.sub(r'\s+',' ',review) # remove extra spaces.
	review=[lemmatizer.lemmatize(word) for word in review.split()]
	review =' '.join(review)
	return review
	
text_train2=text_train1.apply(cleanText)
test2=test1.apply(cleanText)
#-----------------------------------
from string import digits
def removedigits(text):
    trans = str.maketrans('','',digits)
    return text.translate(trans)
text_train3=text_train2.apply(removedigits)
test3=test2.apply(removedigits)
#------------------------------------
#def clean_numbers(x):
#	x = re.sub('[0-9]{5,}', '#####', x)
#	x = re.sub('[0-9]{4}', '####', x)
#	x = re.sub('[0-9]{3}', '###', x)
#	x = re.sub('[0-9]{2}', '##', x)
#	return x

#text_train3=text_train2.apply(clean_numbers)
#test3=test2.apply(clean_numbers)
#---------------------------------
to_remove = ['am','is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',\
			   'had', 'having','do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',\
			   'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for','to', 'from', 'up',\
			   'down', 'in', 'out', 'on', 'off', 'over', 'under','so','s', 't','can',\
			   'will', 'just','now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain','ma']

#import nltk
#from nltk.tokenize import sent_tokenize, word_tokenize
sentences=[]
listOfWord=[]
for s in text_train3:
	sentence=[]
	for word in s.split():		
		if word not in to_remove:
			listOfWord.append(word)
			sentence.append(word) 
	#review=' '.join(sentence)
	sentences.append(' '.join(sentence))
	
	

test4=[]
for s in test3:
	sentence=[]
	for word in s.split():		
		if word not in to_remove:
		#	listOfWord.append(word)
			sentence.append(word) 
	#review=' '.join(sentence)
	test4.append(' '.join(sentence))
	
#----------------------
sent_train=[ 1 if sent>=.5 else 0 for sent in Y]
#--------------------------
listOfWord.append("ENDPAD")

uniquelistwords=list(set(listOfWord))

max_len = 50
word2idx = {w: i for i, w in enumerate(uniquelistwords)}
#------------------------------

train_sentences1=[sentence.split() for sentence in sentences]

test5=[sentence.split() for sentence in test4]

X=[]
for s in train_sentences1:
	intW=[]
	for w in s:
		intW.append(word2idx[w])
	X.append(intW)

test6=[]
for s in test5:
	intW=[]
	for w in s:
		if w in word2idx.keys():
			intW.append(word2idx[w])
	test6.append(intW)

#--------------------------------------------------
padingValue = uniquelistwords.index('ENDPAD')
from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=padingValue)

test7 = pad_sequences(maxlen=max_len, sequences=test6, padding="post", value=padingValue)
#------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed,Bidirectional,Dropout
from keras.layers.embeddings import Embedding

model1 = Sequential()
model1.add(Embedding(input_dim=len(uniquelistwords), output_dim=100, input_length=max_len))
model1.add(Dropout(0.2))
model1.add(LSTM(100))
model1.add(Dropout(0.2))
#model1.add(Dense(100,activation='relu')) # New hidden layer with 4 params
model1.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
#model.add(Dense(1, kernel_initializer='normal',activation='linear'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model1.summary())

model1.fit(X_train, sent_train, epochs=2, batch_size=50,validation_split=0.1, verbose=2)

#---------------------------------------------
test_result=model1.predict(test7)

sample= pd.DataFrame()
sample["id"]=test["id"]
sample["prediction"]=pd.DataFrame(test_result)

sample.to_csv('submission.csv',index=False)


