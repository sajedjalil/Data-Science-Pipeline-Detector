from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Model

import pandas as pd

# load dataset
# egitim_seti = train_set

# test seti = test_set
# Veri setlerini yukleyelim.
train_set = pd.read_csv('../input/train.csv', index_col=False)
test_set = pd.read_csv('../input/test.csv', index_col=False)

# Explore Data

# Veriyi arastiralim.
# print egitim_seti.head()

# Concatenate all test.txt and train.txt
# butun_cumleler = all_texts 

# Test ve Egitim Setindeki cumleleri birlestirelim.
all_sentences = pd.concat([train_set['text'], test_set['text']])

# size of train_set

# egitim setinin buyuklugu
m = len(train_set)  # 19579

# Set the target and predictor variables for Model Training.
# Modelin Egitimi icin gerekli hedef ve ongorucu degiskenleri ayarlayalim.

# Target Variable = Authors
# Hedef Degiskenimiz = Yazarlar

# Encode authors as binary. 
# We encode as binary 'cause we must submit a csv file with the id, 
# and a probability for each of the three classes.
# yazarlari binary olarak kodlayalim.
# bunun amaci, submission dosyasi olustururken,
# tahminlerimizi her bir yazara gore olasilik dagilimi istenmesi.
# EAP = 1 0 0
# HPL = 0 1 0
# MWS = 0 0 1
labelbinarizer = LabelBinarizer()
labelbinarizer.fit(train_set['author'])
y = labelbinarizer.fit_transform(train_set['author'])

# Predictor Variable: Sentences
# These are text, we can not use directly.
# Let's extract some features for machine could understand.
# Transforms each text in texts in a sequence of integers.

# Ongorucu Degiskenimiz: Yazarlarin Kurdugu Cumleler
# Bu cumleler, text oldugu icin direkt kullanamayiz.
# Cumlelerden makinenin anlayabilecegi ozellikler cikartalim.
# Yazarlarin kurdugu cumleleri bir dizi tamsayiya donusturelim.
# texts to sequences islemi dogal olarak her bir cumle icin farkli uzunlukta
# bir tam sayi dizisi donecegi icin,
# pad sequences ile her diziyi en uzun tam sayi dizisinin uzunlugunda
# saklamamizi saglar. Boylece her cumle icin cikardigimiz
# ozellik dizisi ayni uzunluktadir.

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_sentences)
X = tokenizer.texts_to_sequences(train_set['text'])
X = pad_sequences(X)

# X_egitim = X_train
# y_egitim = y_train
X_train = X
y_train = y

# sozluk_boyutu = size of dictionary

# hangi kelimeden kac tane gectigini hesapladigimizda toplam map'in boyutu
# modelimizi olustururken kullanacagiz.
dict_size = len(tokenizer.word_index)  # 29451

# X_test 

# submission dosyasini olusturmak icin kullanacagimiz test seti
# ayni sekilde test setindeki cumleleri kullanarak her biri icin
# ozellik dizilerini olusturalim.
X_test = tokenizer.texts_to_sequences(test_set['text'])
X_test = pad_sequences(X_test)

# Create our model
# our model has four layers

# modelimizi olusturalim
# modelimiz dort katmandan olusuyor

model = Sequential()
model.add(Embedding(input_dim=dict_size + 1, output_dim=30))
model.add(Dropout(0.5))
model.add(LSTM(30))
model.add(Dense(15, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(X_train, y_train, batch_size=32, epochs=5)

model.summary()

# tahminler = predictions

predictions = model.predict(X_test, batch_size=16)
test_set['EAP'] = predictions[:, 0]
test_set['HPL'] = predictions[:, 1]
test_set['MWS'] = predictions[:, 2]

test_set[['id', 'EAP', 'HPL', 'MWS']].to_csv('submission.csv', index=False)
