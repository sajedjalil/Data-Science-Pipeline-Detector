__author__ = "David Kazaryan"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from keras.layers import Dense, Dropout, Input
from keras.models import Model

# Here I prepare two hidden layers neural network with dropout.
# I used Keras Functional API.
def make_model(inps=10):
    inputs    = Input(shape=(inps, ))
    dense_1   = Dense(24, activation='sigmoid')(inputs)
    dropout_1 = Dropout(0.1)(dense_1)
    output    = Dense(3, activation='softmax')(dropout_1)
    
    nnet = Model(input=inputs, output=output)
    nnet.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return nnet

if __name__ == "__main__":

    N_FOLDS = 10
    
    # Reading data...
    data = pd.concat([pd.read_csv('../input/train.csv', index_col='id'), 
                      pd.read_csv('../input/test.csv', index_col='id')])
    
    # Getting one-hot encoding for categorical data ('color' column)
    data = pd.concat([data, pd.get_dummies(data['color'])], axis=1)
    data.drop('color', inplace=True, axis=1)
    
    # Getting one-hot encoding for classes
    data = pd.concat([data, pd.get_dummies(data['type'])], axis=1)
    
    # That's what we get here
    print(data.head())
    
    # Columns for the classification process
    cols = ['bone_length', 'hair_length', 'has_soul', 'rotting_flesh', 
            'blood', 'blue', 'clear', 'green', 'white', 'black']
    
    # Class names
    classes = ['Ghost', 'Ghoul', 'Goblin']
    
    # Preparing data for the training...
    X_data = np.array(data[data['type'].notnull()][cols])
    X_test = np.array(data[data['type'].isnull()][cols])
    y_data = np.array(data[data['type'].notnull()][classes])
    
    kf = KFold(n_splits=N_FOLDS)
    keras_predictions = np.zeros(np.array(data[data['type'].isnull()][classes]).shape)
    xgb_predictions = np.zeros(np.array(data[data['type'].isnull()][classes]).shape)
    
    print("X_data shape: ", X_data.shape)
    print("X_test shape: ", X_test.shape)
    print("predictions shape: ", keras_predictions.shape)
    
    keras_accuracy = []
    xgb_accuracy   = []
    
    # Here I used KFold for the creation of 10 models:
    # 5 NNs and 5 XGBs. I didn't tune them much though.
    for i, (train_index, test_index) in enumerate(kf.split(X_data)):
        nnet = make_model(len(cols))
        nnet.fit(X_data[train_index], y_data[train_index], batch_size=16, nb_epoch=96, shuffle=True,
                 validation_data=(X_data[test_index], y_data[test_index]), verbose=0)
        keras_accuracy.append(accuracy_score(np.argmax(y_data[test_index], axis=1), 
                                             np.argmax(nnet.predict(X_data[test_index]), axis=1)))
        
    
        xgb_clf = XGBClassifier(objective="multi:softprob", max_depth=6, learning_rate=0.01)    
        xgb_clf.fit(X_data[train_index], np.argmax(y_data[train_index], axis=1))
        xgb_accuracy.append(accuracy_score(np.argmax(y_data[test_index], axis=1), 
                                           np.argmax(xgb_clf.predict_proba(X_data[test_index]), axis=1)))
                                           
        print("{} fold accuracy (Keras):   {}".format(i+1, keras_accuracy[-1]))
        print("{} fold accuracy (XGBoost): {}".format(i+1, xgb_accuracy[-1]))     
        
        # Each model predicts classes probability, and we take the overall sum
        keras_predictions += nnet.predict(X_test, batch_size=1024)
        xgb_predictions   += xgb_clf.predict_proba(X_test)
        
    print("Mean accuracy (Keras): ", np.mean(keras_accuracy))
    print("Mean accuracy (XGBoost): ", np.mean(xgb_accuracy))
    
    prediction = (keras_predictions + xgb_predictions) / 2

    f = open('init.csv', 'w')
    f.write('type,' + ','.join(classes) + '\n')
    for i, p in zip(data[data['type'].isnull()].index, prediction):
        f.write(str(i) + ',' + ','.join([str(x) for x in p]) + '\n')
    f.close()
