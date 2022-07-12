
import numpy as np # linear algebra

import os
print(os.listdir("../input"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import re

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint

train=pd.read_json('../input/train.json')
test=pd.read_json('../input/test.json')

test['cuisine']='unknown'

# Merge train and test dataframe
all=pd.concat([train,test],ignore_index=True)
# Remove punctuation from the ingredients field
all.ingredients=all.ingredients.apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

# SPlit the train_df and test_df
train_df=all[all.cuisine !='unknown']
test_df=all[all.cuisine=='unknown']

# fit the tokenizer on all ingredients
tokenizer=Tokenizer()
tokenizer.fit_on_texts(all['ingredients'])

# tf-idf to encode train_df ingredients
train_encoded=tokenizer.texts_to_matrix(train_df['ingredients'],mode='tfidf')
# tf-idf to encode test_df ingredients
test_encoded=tokenizer.texts_to_matrix(test_df['ingredients'],mode='tfidf')

# categorical target field cuisine
y_df=train_df[['cuisine']]
le=LabelEncoder()
y_le=le.fit_transform(y_df)

y_encoded=to_categorical(y_le)

# train and validation dataset split
X_train,X_val,y_train,y_val=train_test_split(train_encoded,y_encoded,test_size=0.2,random_state=88)

# Create the MLP model
MLP=Sequential()
MLP.add(Dense(512,input_shape=(train_encoded.shape[1],),activation='relu'))
MLP.add(Dropout(0.5))

MLP.add(Dense(256,activation='relu'))
MLP.add(Dropout(0.5))

MLP.add(Dense(128,activation='relu'))
MLP.add(Dropout(0.3))

MLP.add(Dense(y_encoded.shape[1],activation='softmax'))


MLP.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
# optimimzer: adagrad 0.7985, Adadelta:0.8006, adadelta:0.7980

# train the model
check_point=ModelCheckpoint(filepath='mlp.hdf5',monitor='val_loss',save_best_only=True,save_weights_only=True)
early_stop=EarlyStopping(monitor='val_loss',patience=5)

MLP.fit(X_train,y_train,
                    batch_size=128,
                    epochs=50,
                    validation_data=(X_val,y_val),
                    callbacks=[check_point,early_stop])


MLP.load_weights(filepath='mlp.hdf5')
mlp_pred=MLP.predict(test_encoded).argmax(axis=1)
final=le.inverse_transform(mlp_pred)

submission=pd.DataFrame({'id':test.id,'cuisine':final},columns=['id','cuisine'])
submission.to_csv('submission.csv',index=False)

