import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as  xgb
from sklearn import preprocessing

os.listdir('../input')
data = pd.read_csv(os.path.join('../input','train.csv'))
data.head()
data.info()
data.columns
cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']
data = data.reindex(np.random.permutation(data.index))
df = data.copy()

def NormalizeData(dataset, cols):
    dataset[cols] = (dataset[cols] -  dataset[cols].min())/(dataset[cols].max() - dataset[cols].min())    
    return dataset
    
df = NormalizeData(df, cols)
df.head()
X_train = df.iloc[:,1:-1]
X_train.head()
y = data['Cover_Type'].copy()
lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(y)
y[1000]

model = Sequential()
model.add(Dense(768, input_dim=54, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(384, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0003), metrics=['accuracy'])
model.fit(X_train,y, epochs=256, batch_size=32, validation_split=0.3)

df_test = pd.read_csv('../input/test.csv')			#验证集参数传入
df_test.head()
X_test = df_test.copy()
X_test = NormalizeData(X_test, cols)
X_test.head()
X_test.drop(['Id'], axis=1, inplace=True)
X_test.head()
preds = model.predict(X_test)
sub = pd.DataFrame({"Id": df_test.iloc[:,0].values,"Cover_Type": lb.inverse_transform(preds)})
sub.to_csv("submission.csv", index=False) 
sub.head()
print("end!")