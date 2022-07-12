from sklearn import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

""" Read Data """
train_variant = pd.read_csv("../input/training_variants")
test_variant = pd.read_csv("../input/test_variants")
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train_variant, train_text, how='left', on='ID')
train_y = train['Class'].values
train_x = train.drop('Class', axis=1)
# number of train data : 3321

test_x = pd.merge(test_variant, test_text, how='left', on='ID')
test_index = test_x['ID'].values
# number of test data : 5668


all_data = np.concatenate((train_x, test_x), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]

sentences = all_data['Text']
sentences

vect = TfidfVectorizer(stop_words = 'english')
sentence_vectors = vect.fit_transform(sentences)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(200)
sentence_vectors = svd.fit_transform(sentence_vectors)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dropout

# define model
def baseline_model():
    model = Sequential()
    model.add(Dense(512, input_dim=200, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dense(9, init='normal', activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)

dummy_y = np_utils.to_categorical(encoded_y)
print(dummy_y.shape)

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=64)
estimator.fit(sentence_vectors[0:3321], dummy_y, validation_split=0.05)

y_pred = estimator.predict_proba(sentence_vectors[3321:])

""" Submission """
submission = pd.DataFrame(y_pred)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv("submission.csv",index=False)
