# This Python 3 environment comes with many helpful analytics libraries installed
import time
t0=time.time()
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import tensorflow as tf # Input data files are available in the "../input/" directory. 
from tensorflow.keras import datasets 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from scipy.interpolate import*
from tensorflow import keras
from tensorflow.keras import layers
#import keras
#from keras import optimizers
import sklearn 
from sklearn.neighbors import DistanceMetric
from scipy.stats import zscore
from random import* 
from pandas.compat.numpy import* 
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.pyplot import draw, show, plot 
from sklearn.preprocessing import MinMaxScaler 
from keras.utils import to_categorical
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory 
import os 
print(os.listdir("../input")) # Any results you write to the current directory are saved as output. 
#######
data_train =pd.read_csv("../input/train.csv") 
data_test  =pd.read_csv("../input/test.csv") 
print (data_train.head(5))
print (data_test.head(5))
####### CLEANING THE DATA
data_train.isna().sum()
data_test.isna().sum()
#data_train= data_train.dropna()  ### dropping NA data
#data_test = data_test.dropna()   ### dropping NA data
########### 
label_names =["feature_1","feature_2","feature_3"] 
target_names=["target"] 
label_target_names=["feature_1","feature_2","feature_3","target"]
########### INSPECTING THE DATA
sns.pairplot(data_train[["feature_1","feature_2","feature_3","target"]], diag_kind="kde")
sns.pairplot(data_test[["feature_1","feature_2","feature_3"]], diag_kind="kde")
######## STATISTIC
data_train_stats = data_train.describe()
data_test_stats  = data_test.describe()
#train_stats.pop("feature_1")
data_train_stats = data_train_stats.transpose()
data_test_stats  = data_test_stats.transpose()
#data_train_stats
#data_test_stats

#################################################
y_target_max=float(data_train[target_names].max())
y_target_min=float(data_train[target_names].min())
########## NORMALIZED DATA
normed_train_data = (data_train[label_target_names] - data_train[label_target_names].mean()) / (data_train[label_target_names].max()-data_train[label_target_names].min())
normed_test_data  = (data_test[label_names]  - data_test[label_names].mean()) / (data_test[label_names].max()-data_test[label_names].min())
#normed_train_data = (data_train[label_target_names] - float(data_train[label_target_names].mean())) / (float(data_train[label_target_names].max())-float(data_train[label_target_names].min()))
#normed_test_data  = (data_test[label_names]  - float(data_test[label_names].mean())) / (float(data_test[label_names].max())-float(data_test[label_names].min()))
print (normed_train_data)
print (normed_test_data)
##### HISTOGRAM OF ALL FEATURES
normed_train_data.hist()
plt.show()
sns.pairplot(normed_train_data[["feature_1","feature_2","feature_3","target"]], diag_kind="kde")
############# OUTLIER REMOVAL ############################################
data_train_zscore = normed_train_data[target_names].apply(zscore)  ## ZSCORE
threshold         = -0.1  ## Threshold
data              = normed_train_data[target_names]
#print (data)
#dd                = data
print('Mean of target data',float(np.mean(data)))
#print(data)
normed_train_data.hist()
############ remove and replacing outliers###########
def f(x):
    if x<threshold:
        x=0.0#float(dd.min())
    else:
        x=x
    return x 
data=data.any().apply(f)
#print(data)
############### DATA RECONVERSION
normed_data_train=data
normed_train_data.hist()
#print(normed_train_data[target_names])
###########################################################
train_features=normed_train_data[label_names]
train_labels  =normed_train_data[target_names]
######
test_features =normed_test_data[label_names]
test_features.hist()
#### BUILDING MODEL
def build_model():
  model = Sequential([
    Dense(64, activation=tf.nn.relu, input_shape=[len(train_features.keys())]),
    Dense(64, activation=tf.nn.relu),
    Dense(1)
  ])

  optimizer = RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
model.summary()
################## TRAIN THE MODEL 
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
##########
EPOCHS = 100  #####PLAY WITH THIS NUMBER FOR CONVERGENCE
early_stop =EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  train_features, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop,PrintDot()])
#plot_history(history)
################### HISTORY TAIL
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
##############################
###########PLOTTTING RESULTS
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [target]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,0.08])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [target^2]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,0.01])

plot_history(history)
####### MEAN ERROR OF VALIDATION SET
loss, mae, rmse = model.evaluate(train_features, train_labels, verbose=0)

print("Testing: {:5.2f}".format(rmse))
#########
test_predictions = model.predict(normed_test_data).flatten()
###### INTERPOLATION
test_range    =list(range(0,len(normed_test_data)))
train_range   =list(range(0,len(train_labels)))
train_labels_i=train_labels['target']
train_labels_i=train_labels_i.T
Train_values  =np.interp(test_range,train_range,train_labels_i)
print (len(test_range),len(train_range), Train_values.shape)
############### DATA RENORMALIZATION
Train_values     =float(y_target_max-y_target_min)*Train_values + float(y_target_max)
test_predictions =float(y_target_max-y_target_min)*test_predictions + float(y_target_max)
###############
plt.figure(figsize=(15,5))
plt.scatter (normed_test_data["feature_1"],Train_values,c='b',s=40,alpha=0.75, label='True Values [target]')
plt.scatter (normed_test_data["feature_1"],test_predictions,c='r',s=40,alpha=0.5, label='Predictions [target]')
plt.ylabel('target')
plt.xlabel('feature_1')
plt.legend()
plt.show()
    ############### RMSE ESTIMATION
ERROR=np.sqrt(np.mean((test_predictions-Train_values)**2))
print ("RSME",format(ERROR, '.3f'))

error1 = test_predictions - Train_values
plt.hist(error1, bins = 25)
plt.xlabel("Prediction Error [target]")
_ = plt.ylabel("Count")
######### SAVING THE DATA
data_test['target']=test_predictions
target = data_test['target']
card_id= data_test['card_id'] 
df3= pd.concat([card_id,target], axis=1)
DatatoSubmit=df3.to_csv('Submission_LN.csv',index=True)
#print(df3)
##### TOTAL TIME
t1   =time.time()
total=t1-t0
print ('Time spent is about:', np.round(total), 'seconds')

