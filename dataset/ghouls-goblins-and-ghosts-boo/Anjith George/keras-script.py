

import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
from sklearn.preprocessing import StandardScaler
import keras.models as models
import keras.utils.np_utils as kutils
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.utils.np_utils import to_categorical

train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)



# Really simple data preparation
y_train = pd.get_dummies(train[["type"]], prefix="")
train.drop("type", inplace=True, axis=1)

train_test = pd.concat([train, test], axis=0)

# It looks like the color actually is just noise, and does not give any signal to the monster-class.
# Comment one of these lines.
#train_test = pd.get_dummies( train_test, columns=["color"], drop_first=False)
train_test.drop("color", inplace=True, axis=1)

X_train = train_test.iloc[:len(y_train)]
X_test  = train_test.iloc[len(y_train):]

# Clean up
del train_test
del train
del test




# (EDIT: It's much faster to convert the dataframes to numpy arrays and then iterate)
X = np.array(X_train, dtype=np.float32)
Y = np.array(y_train, dtype=np.float32)



model = Sequential()
model.add(Dense(32,input_dim=4,  init='he_uniform', activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

scaler= StandardScaler().fit(X)
X = scaler.transform(X)



## Error is measured as categorical crossentropy or multiclass logloss
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])


## Fitting the model on the whole training data
model.fit(X,Y,batch_size=128,
					nb_epoch=3000,verbose=2, validation_split=0.1)
					
test=np.array(X_test,dtype=np.float32)
yPred = model.predict(test)

## Converting the test predictions in a dataframe as depicted by sample submission



with open('submission_keras.csv', 'w') as f:
	f.write("id,type\n")
	for index, monster in X_test.iterrows():
		#print "monster.shape",monster.shape
		tda=np.array(monster, dtype=np.float32).ravel()
		tda=np.reshape(tda,(1,4))
		tda=scaler.transform(tda)
		
		

		
		probs = model.predict_proba( tda)
		f.write("{},{}\n".format(index, y_train.columns.values[np.argmax(probs)][1:]))
