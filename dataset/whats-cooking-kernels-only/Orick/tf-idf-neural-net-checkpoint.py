# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from numpy import argmax
import pandas as pd
import json

# Dataset Preparation
print ("Read Dataset ... ")
train = json.load(open('../input/train.json'))
test = json.load(open('../input/test.json'))

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train_text).astype('float16')
X_test = tfidf.transform(test_text).astype('float16')

# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
dummy_y_train = np_utils.to_categorical(y)

# Model Training 
print ("Create model ... ")
def build_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=3010, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

print("Fit model ...")
file_path="best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]
model = build_model()
model.fit(X, dummy_y_train, epochs=30, batch_size=128, validation_split=0.1, callbacks=callbacks_list)
model.load_weights(file_path)

# Predictions 
print ("Predict on test data ... ")
pred = argmax(model.predict(X_test), axis = 1)
y_pred = lb.inverse_transform(pred)

# Submission
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('nn_output.csv', index=False)















