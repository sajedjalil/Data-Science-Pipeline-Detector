import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,Dropout
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

train_df =  pd.read_csv("../input/train.csv")

#--Feature selection
features = [x for x in train_df.columns.values.tolist() if x.startswith("var_")]

#--Scaling data and store scaling values
scaler = preprocessing.StandardScaler().fit(train_df[features].values)

X = scaler.transform(train_df[features].values)
y = train_df['target'].values

#--training & test stratified split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y, test_size=0.10)
# reshape dataset
X_train=np.reshape(X_train,(180000,10,10,2))
X_valid=np.reshape(X_valid,(20000,10,10,2))
print(X_train.shape)
print(X_valid.shape)

#--Model training        

model = Sequential()
model.add(ZeroPadding2D((2, 2), input_shape=( 10, 10,2)))
model.add(Convolution2D(64, 2, 2, activation='relu', name='conv1_1'))
model.add(BatchNormalization())
model.add(Flatten(input_shape=model.output_shape[1:]))

model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])

earlystop = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)

history = model.fit(X_train, y_train, epochs=300, batch_size=180000, validation_data=(X_valid, y_valid), callbacks=[earlystop])

#--AUC
y_pred_keras = model.predict(X_valid).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_valid, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

roc_auc_score(y_valid, y_pred_keras)

#--Plot AUC
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#--fitting plot
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    #plt.plot([0, 1], [0, 1], 'k--')
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')

    plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.ylim([0, 0.4])
    plt.show()

plot_history([('baseline', history)])

#--make prediction
test_df =  pd.read_csv("../input/test.csv")

X_test = scaler.transform(test_df[features].values)
X_test = np.reshape(X_test,(200000,10,10,2)) 
prediction = model.predict(X_test)

result = pd.DataFrame({"ID_code": test_df.ID_code.values})
result["target"] = prediction
result.to_csv("submission.csv", index=False)
model.save('./my_model.h5')
