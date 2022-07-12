# Based on https://www.kaggle.com/fchollet/digit-recognizer/simple-deep-mlp-with-keras/code

import pandas as pd

import pandas as pd


# Columns with almost same value
mixCol = [8,9,10,11,12,18,19,20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45, 
          73, 74, 98, 99, 100, 106, 107, 108, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 180, 
          181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 202, 205, 206, 207, 
          208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 240, 371, 372, 373, 374,
          375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 
          396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 
          437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
          458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
          479, 480, 481, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
          510, 511, 512, 513, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 840]

#Columns with logical datatype
alphaCol = [283, 305, 325, 352, 353, 354, 1934]

#Columns with Places as entries
placeCol = [200, 274, 342]

#Columns with timestamps
dtCol = [75, 204, 217]

selectColumns = []
rmCol = mixCol+alphaCol+placeCol+dtCol
for i in range(1,1935):
    if i not in rmCol:
        selectColumns.append(i)

cols = [str(n).zfill(4) for n in selectColumns]
strColName = ['VAR_' + strNum for strNum in cols] 

nrows = 50000
trainData = pd.read_csv("../input/train.csv", usecols=strColName, nrows=nrows)
testData = pd.read_csv("../input/test.csv", usecols=strColName)
label = pd.read_csv("../input/train.csv", usecols=['target'], nrows=nrows)

numericFeatures = trainData._get_numeric_data()
numericFeatures_test = testData._get_numeric_data()

# filling na values
removeNA = numericFeatures.fillna(0)
removeNA_test = numericFeatures_test.fillna(0)

# remove all features that are either one or zero (on or off) in more than 80% of the samples
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(removeNA)
features_test = sel.fit_transform(removeNA_test)

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
y = np.array(label).ravel()

clf = ExtraTreesClassifier()
X_new = clf.fit(features, y).transform(features)
X_new_test = clf.fit(features_test, y).transform(features_test)

from sklearn import preprocessing
X_scaled = preprocessing.scale(X_new)
X_scaled_test = preprocessing.scale(X_new_test)
#display(pd.DataFrame(X_scaled).head())

normalizer = preprocessing.Normalizer().fit(X_scaled)
X_norm = normalizer.transform(X_scaled)
X_norm_test = normalizer.transform(X_scaled_test) 
#display(pd.DataFrame(X_norm).head())
X_test = X_norm_test

print(X_test)

# In[34]:

# Dividing Data into training and crossvalidation sets
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.0, random_state=666)

# In[35]:

from keras.utils import np_utils
y_train = np.array(y_train)
#print zip(range(0, len(X_train), 128), range(128, len(X_train), 128))
y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test) 

# print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# In[36]:
from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(input_dim, input_dim/2, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(input_dim/2, 256, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256, nb_classes, init='lecun_uniform'))
model.add(Activation('relu'))

# we'll use MSE (mean squared error) for the loss, and RMSprop as the optimizer
model.compile(loss='mse', optimizer='rmsprop')

print("Training...")
model.fit(X_train, y_train, nb_epoch=50, batch_size=128*2, validation_split=0.25, show_accuracy=True, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

submission = pd.DataFrame({ 'ID': test_df['PassengerId'],
                            'target': preds })
submission.to_csv("submission.csv", index=False)
