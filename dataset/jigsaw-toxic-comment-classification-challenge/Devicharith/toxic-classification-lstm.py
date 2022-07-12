import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

train.isnull().any()
test.isnull().any()

train_text = train.iloc[:, 1]
test_text = test.iloc[:, 1]

#texts to sequences
max_features = 50000
from keras.preprocessing.text import Tokenizer
tokens = Tokenizer(num_words=max_features)
tokens.fit_on_texts(train_text)
train_tokens = tokens.texts_to_sequences(train_text)
test_tokens = tokens.texts_to_sequences(test_text)

#padding
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
tra_x = pad_sequences(train_tokens, maxlen=100)
te_x = pad_sequences(test_tokens, maxlen=100)

#creating a model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GlobalMaxPool1D
from keras.layers import Bidirectional
model = Sequential()
#Embedding
model.add(Embedding(input_dim = 50000,output_dim = 64,input_length=100))
#Lstm
model.add(Bidirectional(LSTM(64,return_sequences=True)))
#Maxpooling
#model.add(GlobalMaxPool1D())
#Dense
model.add(Dense(50,activation = 'relu'))
#Dropout
model.add(Dropout(0.1))
#Dense
model.add(Dense(6,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

toxic_types = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
y = train[toxic_types].values
model.fit(tra_x,y,batch_size=64, epochs=1)

submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')
score = model.predict_proba(te_x,batch_size=256)
submission[toxic_types]=score
submission.to_csv('submission.csv', index=False)