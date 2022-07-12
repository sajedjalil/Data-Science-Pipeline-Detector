import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

x_train = train[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']]
target = train['type']
x_test = test[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']]
x_train = pd.get_dummies(x_train, columns=['color'])
target = pd.get_dummies(target)
x_test = pd.get_dummies(x_test, columns=['color'])

x_train = x_train.values
target = target.values
x_test = x_test.values

#x_train_color = x_train[:,4:]
#x_test_color = x_test[:,4:]
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#nor_train = scaler.fit_transform(x_train_color)
#nor_test = scaler.fit_transform(x_test_color)
#x_train[:,4:] = nor_train
#x_test[:,4:] = nor_test

n_feature = x_train.shape[1]
n_class = target.shape[1]

#from sklearn.model_selection import train_test_split
#X_train, X_cv, y_train, y_cv = train_test_split(x_train, target, test_size=0.1)
#X_train = X_train.values
#y_train = y_train.values.astype(np.uint8)


from keras.models import Sequential
from keras.layers import Dense, Dropout

##Neural Network Building
model = Sequential()
model.add(Dense(100, input_dim=n_feature, activation='relu', init='uniform'))
model.add(Dropout(0.3))
#model.add(Dense(10, activation='relu'))
#model.add(Dense(6, activation='relu'))
model.add(Dense(n_class, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, target, batch_size=15, nb_epoch=100)

pred_proba = model.predict(x_test)
pred = np_utils.categorical_probas_to_classes(pred_proba)

pred_class = pd.Series(pred).map({0:'Ghost', 1:'Ghoul', 2:'Goblin'})
Id = test.id
result = pd.DataFrame({
          'id' : Id,
          'type' : pred_class
          })
result.to_csv('./result.csv', index=False)