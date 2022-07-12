#Load standard libraries
import numpy as np
import pandas as pd




#Load data
df = pd.read_csv('../input/train.csv')
print(df.columns.values)


#Load Keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler


#Define network
M1 = 1000
leaf_layer = Sequential()
leaf_layer.add(Dense(M1,input_dim=64*3,init='uniform',activation='relu'))
leaf_layer.add(Dropout(0.2))
leaf_layer.add(Dense(M1,init='uniform',activation='relu'))
leaf_layer.add(Dropout(0.2))
leaf_layer.add(Dense(M1,init='uniform',activation='sigmoid'))
leaf_layer.add(Dropout(0.3))
leaf_layer.add(Dense(99,init='uniform',activation='softmax'))


#Load train data
data_train = df.filter(regex="margin*|shape*|texture*").values
data_train = StandardScaler().fit(data_train).transform(data_train)
le = LabelEncoder().fit(df['species'])
labels_train = le.transform(df['species'])
labels_train = to_categorical(labels_train)


#Train network
leaf_layer.compile(optimizer='rmsprop', loss='binary_crossentropy')
leaf_layer.fit(data_train, labels_train, nb_epoch=100, batch_size=50)


#Load test data
df_test = pd.read_csv("../input/test.csv")
data_test = df_test.filter(regex="margin*|shape*|texture*").values
data_test = StandardScaler().fit(data_test).transform(data_test)
test_ids = df_test.pop('id')


#Predict labels
predicted_labels = leaf_layer.predict(data_test)


#Save as csv
df_pred = pd.DataFrame(predicted_labels,index=test_ids,columns=le.classes_)
csv_test = open('submission1.csv','w')
csv_test.write(df_pred.to_csv())