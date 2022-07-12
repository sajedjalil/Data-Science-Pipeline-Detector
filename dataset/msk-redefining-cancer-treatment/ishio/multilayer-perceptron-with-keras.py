import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dropout, Activation
from keras.utils import np_utils

from subprocess import check_output

# training data
train_variant = pd.read_csv("../input/training_variants")
train_variant = train_variant.set_index('ID')
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_text = train_text.set_index('ID')
train = pd.concat([train_variant, train_text], axis=1)

# test data
test_variant = pd.read_csv("../input/test_variants")
test_variant = test_variant.set_index('ID')
test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = test_text.set_index('ID')
test = pd.concat([test_variant, test_text], axis=1)

sentences = pd.DataFrame(np.concatenate((train.Text, test.Text), axis=0), columns=["Text"]).Text

print(len(train))
print(len(test))
print(len(sentences))

# Calculate TF-IDF with scikitlearn
# use English stop word
vect = TfidfVectorizer(stop_words = 'english')
words_vectors = vect.fit_transform(sentences) 
#<8989x169129 sparse matrix of type '<type 'numpy.float64'>' with 14880440 stored elements in Compressed Sparse Row format>

# Singular value decomposition
# 169129 dim -> 1000 dim
svd = TruncatedSVD(1000)
words_vectors = svd.fit_transform(words_vectors)

# Keras Data set
train_x = words_vectors[0:3321]
test_x = words_vectors[3321:]
label = np.array(train_variant.Class)
train_y = np_utils.to_categorical(label-1)

# multi perceptron model
print("start moderl")
model = Sequential()
model.add(Dense(512, input_dim=1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=200, batch_size=100)

predict = model.predict_classes(test_x, batch_size=128) 
res = np_utils.to_categorical(predict)
np.save('..input/res.npy', res)

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

submisstion = np.load('..input/res.npy')
submit = pd.DataFrame(submisstion, columns=['class1','class2','class3','class4','class5','class6','class7','class8','class9'])
submit['ID']= range(5668)
submit = submit.set_index('ID')

submit.to_csv("..input/output.csv",index=True)
# Any results you write to the current directory are saved as output.