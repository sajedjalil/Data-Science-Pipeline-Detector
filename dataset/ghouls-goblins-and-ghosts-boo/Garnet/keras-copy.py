import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.utils import np_utils

from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def make_model(nb_inputs=10):
    inputs = Input(shape=(nb_inputs, ))
    dense1_1 = Dense(50, activation='sigmoid')(inputs)
    dropout_1 = Dropout(0.2)(dense1_1)
    output = Dense(3, activation='softmax')(dropout_1)
    
    nn_model = Model(input=inputs, output=output)
    nn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return nn_model
    
N_FOLDS = 10

raw_train = pd.read_csv('../input/train.csv', index_col='id')
raw_test = pd.read_csv('../input/test.csv', index_col='id')
data = pd.concat([raw_train, raw_test])
data = pd.concat([data, pd.get_dummies(data['color'])], axis=1)
data = pd.concat([data, pd.get_dummies(data['type'])], axis=1)
data = data.drop(['color'], axis=1)

feature = ['bone_length', 'hair_length', 'has_soul', 'rotting_flesh', 
            'black', 'blood', 'blue', 'clear', 'green', 'white']
classes = ['Ghost', 'Ghoul', 'Goblin']

X_data = data[~data['type'].isnull()][feature]
X_test = data[data['type'].isnull()][feature]
y_data = data[~data['type'].isnull()][classes]
              
kf = KFold(n_splits=N_FOLDS)
keras_predictions = np.zeros(data[data['type'].isnull()][classes].shape)
keras_accuracy = []

for i, (train_index, test_index) in enumerate(kf.split(X_data)):
#    print('i: ',i)
#    print('train_index: ', train_index, len(train_index))
#    print('test_index:', test_index, len(test_index))
    nn = make_model(len(feature))
    nn.fit(X_data.iloc[train_index].values, y_data.iloc[train_index].values, batch_size=16, nb_epoch=96, shuffle=True,
           validation_data=(X_data.iloc[test_index].values, y_data.iloc[test_index].values), verbose=0)
    cv_target = np.argmax(y_data.iloc[test_index].values, axis=1)
    cv_pred = np.argmax(nn.predict(X_data.iloc[test_index].values), axis=1)
    keras_accuracy.append(accuracy_score(cv_target, cv_pred))
    
print('Keras mean accuracy: ', np.mean(keras_accuracy))

keras_predictions = nn.predict(X_test.values)
pred = np_utils.categorical_probas_to_classes(keras_predictions)
pred_class = pd.Series(pred).map({0:'Ghost', 1:'Ghoul', 2:'Goblin'})

Id = raw_test.index
result = pd.DataFrame({
          'id' : Id,
          'type' : pred_class
          })
result.to_csv('./result.csv', index=False)