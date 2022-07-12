# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "."]).decode("utf8"))

from sklearn.preprocessing import LabelEncoder


import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
   
import time
from keras import metrics
from keras.layers import Conv2D, MaxPooling2D, Input,Conv1D,MaxPooling1D,Flatten,Dense
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import itertools
import matplotlib.pyplot as plt
print('import done')
# Any results you write to the current directory are saved as output.

mercari_train= pd.read_csv('../input/train.tsv',sep='\t')
mercari_test= pd.read_csv('../input/test.tsv',sep='\t',engine='python')
print('data read complete')


category_list= pd.unique(mercari_train['category_name'])

with_category= mercari_train[mercari_train["category_name"].notnull()]  
no_category=  mercari_train[mercari_train["category_name"].isnull()]  

with_category['category_name']= with_category['category_name'].astype(str)
with_category['name']= with_category['name'].astype(str)
with_category['item_description']= with_category['item_description'].astype(str)
with_category['brand_name']= with_category['brand_name'].astype(str)

mercari_test['category_name']= mercari_test['category_name'].astype(str)
mercari_test['name']= mercari_test['name'].astype(str)
mercari_test['item_description']= mercari_test['item_description'].astype(str)
mercari_test['brand_name']= mercari_test['brand_name'].astype(str)



train_df= with_category
le1= LabelEncoder()
le2= LabelEncoder()
le1.fit(train_df.category_name)
test_category= mercari_test.category_name.map(lambda s: 'other' if s not in le1.classes_ else s)
le1_classes = le1.classes_.tolist()
import bisect
bisect.insort_left(le1_classes, 'other')
le1.classes_ = le1_classes
train_df= train_df.assign(category= lambda x:le1.transform(x.category_name))
mercari_test= mercari_test.assign(category= le1.transform(test_category))

le2.fit(train_df.brand_name)
test_brand= mercari_test.brand_name.map(lambda s: 'other' if s not in le2.classes_ else s)
le2_classes = le2.classes_.tolist()
import bisect
bisect.insort_left(le2_classes, 'other')
le2.classes_ = le2_classes
train_df= train_df.assign(brand= lambda x:le2.transform(x.brand_name))
mercari_test= mercari_test.assign(brand= le2.transform(test_brand))

max_len=20
num_words= 90000

raw_text = np.hstack([train_df.category_name.str.lower(), 
                      train_df.item_description.str.lower(), 
                      train_df.name.str.lower()])

tok_raw = Tokenizer(num_words=num_words)
tok_raw.fit_on_texts(raw_text)
category_name_seq = tok_raw.texts_to_sequences(train_df.category_name.str.lower())
item_description_seq = tok_raw.texts_to_sequences(train_df.item_description.str.lower())
name_seq = tok_raw.texts_to_sequences(train_df.name.str.lower())

data_category_name= pad_sequences(category_name_seq, maxlen=max_len)
data_item_description = pad_sequences(item_description_seq, maxlen=max_len)
data_name= pad_sequences(name_seq, maxlen=max_len)

brand= np.array(train_df.brand)
item_condition= np.array(train_df.item_condition_id)
shipping=  np.array(train_df.shipping)

category_name_seq_test = tok_raw.texts_to_sequences(mercari_test.category_name.str.lower())
item_description_seq_test = tok_raw.texts_to_sequences(mercari_test.item_description.str.lower())
name_seq_test = tok_raw.texts_to_sequences(mercari_test.name.str.lower())

data_category_name_test= pad_sequences(category_name_seq_test, maxlen=max_len)
data_item_description_test = pad_sequences(item_description_seq_test, maxlen=max_len)
data_name_test= pad_sequences(name_seq_test, maxlen=max_len)

brand_test= np.array(mercari_test.brand)
item_condition_test= np.array(mercari_test.item_condition_id)
shipping_test=  np.array(mercari_test.shipping)

X_train={
        "data_category_name":data_category_name,
        "item_condition": item_condition,
        "data_item_description":data_item_description,
        "data_name":data_name,
        "brand":brand,
        "shipping":shipping
        }
X_test={
        "data_category_name":data_category_name_test,
        "item_condition": item_condition_test,
        "data_item_description":data_item_description_test,
        "data_name":data_name_test,
        "brand":brand_test,
        "shipping":shipping_test
        }
price= np.array(train_df.price)
print('Keras Model Building!')

category_name_input= Input(shape=[max_len],name="data_category_name")
item_description_input= Input(shape=[max_len],name="data_item_description")
name_input= Input(shape=[max_len],name="data_name")

item_condition_input= Input(shape=[1],name="item_condition")
brand_input= Input(shape=[1],name="brand")
shipping_input=Input(shape=[1],name="shipping")

emb_category_name= Embedding(num_words, 3)(category_name_input)
emb_item_description = Embedding(num_words, 20)(item_description_input)
emb_name= Embedding(num_words, 10)(name_input)

#emb_item_condition = Embedding(6, 5)(item_condition_input)
dense_item_condition= Dense(64)(item_condition_input)
dense_brand= Dense(16)(brand_input)
dense_shipping= Dense(16)(shipping_input)


rnn_layer1 = GRU(16) (emb_category_name)
rnn_layer2 = GRU(16) (emb_item_description)
rnn_layer3 = GRU(16) (emb_name)


main_l = concatenate([
    dense_item_condition,
    dense_brand,
    dense_shipping,
    rnn_layer1,
    rnn_layer2,
    rnn_layer3
])
main_l = Dropout(0.3)(Dense(64,activation='relu') (main_l))


output = Dense(1,activation="linear") (main_l)

#model
model = Model([category_name_input,item_description_input,name_input,
               item_condition_input,brand_input,shipping_input], output)
#optimizer = optimizers.RMSprop()
optimizer = optimizers.Adam()
model.summary()
model.compile(loss='mean_absolute_error', 
              optimizer=optimizer)

print('Model Ready!')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("./weights.{epoch:04d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
print('Model Training')
history= model.fit(X_train,price,batch_size=128,epochs=2,verbose=1,shuffle=True,
          validation_split=0.1,callbacks=callbacks_list)
print('Optimimum model')
#model.load_weights("weights.0001-11.29.hdf5")
print('Prediction')
pred= model.predict(X_test)
submission_df= pd.concat([mercari_test.test_id,pd.DataFrame(pred)],axis=1)
submission_df.columns=['test_id','price']
submission_df.to_csv("keras_initial_model.csv",index=False)
