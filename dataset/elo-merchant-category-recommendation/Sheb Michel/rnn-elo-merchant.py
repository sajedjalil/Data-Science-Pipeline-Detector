# This Python 3 environment comes with many helpful analytics libraries installed 
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python 
# For example, here's several helpful packages to load in 
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
#import keras
#from keras import optimizers
import sklearn 
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
#from __future__ import absolute_import, division, print_function 
#tf.__version__
#tf.keras.__version__
#tf.logging.set_verbosity(tf.logging.INFO) 
#######
data_train =pd.read_csv("../input/train.csv") 
data_test =pd.read_csv("../input/test.csv") 
data_train_header =list(data_train) 
#Elo.columns data_test_header =list(data_test) 
#print (data_train.shape) 
print (data_train.tail(3)) 
#data_train['first_active_month ']['target'].plot() 
#print (data_train.head('sample_submission.csv'))
L1=len(data_train)
label_names =["feature_1","feature_2","feature_3"] 
target_names=["target"] 
train_split = 0.9
L2 = int(train_split * L1)  ### 90% OF TRAINING DATA
L3= L1-L2
#print('L1=',L1,'L2=',L2,'L3=',L3)
#### TRAINING DATA WITHIN TRAINING SET
X_train =data_train[0:L2][label_names]  ### TRAINING DATA SET
X_test  =data_train[L2:][label_names]   ### TRAINING DATA SET
##### VALIDATION SET
YTrain  =data_train[target_names]   # FULL VALIDATION SET
Y_train =data_train[0:L2][target_names]
Y_test  =data_train[L2:][target_names] 
L4=len(X_train) + len(X_test)
### DETERMINE THE SCALE OF YOUR DATA 
#print (L4,'L1?') 
#print(type(X_train)) 
#print("Shape",X_train.shape) 
#print(type(Y_train)) 
#print("Shape",Y_train.shape) 
#print("Max X_train", np.max(X_train)) 
#print("Min X_train",np.min(X_train)) 
#print("Max Y_train", np.max(Y_train)) 
#print ("Min Y_train",np.min(Y_train)) 
########### INSPECT THE DATA
train_dataset = data_train#(frac=0.8,random_state=0)
#test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[["feature_1","feature_2","feature_3","target"]], diag_kind="kde")
train_stats = train_dataset.describe()
#train_stats.pop("feature_1")
train_stats = train_stats.transpose()
train_stats
############
### SINCE VALUES ARE NOT BOUNDED BETWEEN -1 AND 1, RESCALING IS REQUIRED 
x_scaler      =MinMaxScaler() 
X_train_scaled=x_scaler.fit_transform(X_train) 
X_test_scaled =x_scaler.transform(X_test) 
#print("Max X_train_scaled",np.max(X_train_scaled)) 
#print("Min X_train_scaled",np.min(X_train_scaled)) 
#print("Max X_test_scaled",np.max(X_test_scaled)) 
#print("Min X_test_scaled",np.min(X_test_scaled))  
#######RESCALING TEST DATA 
#print(type(X_test)) 
#print("Shape",X_test.shape) 
y_scaler       = MinMaxScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)
Y_test_scaled  = y_scaler.transform(Y_test)
#print(Y_train_scaled.shape)
#print(Y_test_scaled.shape)
#print("Max Y_train_scaled", np.max(Y_train_scaled)) 
#print ("Min Y_train_scaled",np.min(Y_train_scaled))
#print("Max Y_test_scaled", np.max(Y_train_scaled)) 
#print ("Min Y_test_scaled",np.min(Y_train_scaled))
################ FINAL TEST DATA
L5           =len(data_test)
XPred        =data_test[0:L5][label_names]  ### TRAINING DATA SET
XPred_scaled =x_scaler.transform(XPred)
#print('L5', L5)
#print(XPred_scaled.shape)
##### NEED TO FIND YPred=model.predict(XPred)
def batch_generator(batch_size, sequence_length): 
    ## Infinite loop of batch generator functions 
    while True: ### INPUT DATA BATCH 
       X_shape=(batch_size, sequence_length,num_X_signals) 
       X_batch=np.zeros(shape=X_shape, dtype=np.float16) 
       ### INPUT DATA BATCH 
       Y_shape=(batch_size,sequence_length,num_Y_signals) 
       Y_batch=np.zeros(shape=Y_shape, dtype=np.float16) 
       ###### FILL THE BATCH WITH RANDOM DATA 
       for i in range (batch_size): 
          idx = np.random.randint(len(X_train)-sequence_length) 
          ### COPY THE BATCH 
          X_batch[i]=X_train_scaled[idx:idx+sequence_length] 
          Y_batch[i]=Y_train_scaled[idx:idx+sequence_length] 
       yield (X_batch, Y_batch) 
########### PARAMETER DECLARATION
batch_size     =10#5  ### BATCH SIZES
sequence_length=15#0   ### 
num_X_signals  =3    ### Number of Features (Columns)
num_Y_signals  =1    ### NBRE TARGET
generator=batch_generator(batch_size=batch_size,sequence_length=sequence_length) 
X_batch,Y_batch =next(generator) 
#print (X_batch.shape) 
#print (Y_batch.shape) 
#print (X_batch) #### 
batch=0 # First sequence in the batch 
signal=0 # First signal in the 3 signals 
####################
## Validation on improved test set 
validation_data=(np.expand_dims(X_test_scaled, axis=0),np.expand_dims(Y_test_scaled, axis=0)) 
#### CLASSIFIER 
model=Sequential() 
model.add(GRU(units=512, return_sequences=True, input_shape=(None,num_X_signals))) 
model.add(Dense(num_Y_signals, activation='sigmoid'))
###
if False:
    from tensorflow.python.keras.initializers import RandomUniform
    init=RandomUniform(minval=-0.05,maxval=0.05)  ### Experiment this value until it works
    model.add(Dense(num_Y_signals, activation='sigmoid', kernel_initializer=init))
###########################
######## LOSS FUNCTION
warmup_steps=0
def loss_mse_warmup(Y_true,Y_pred):
    Y_true_slice=Y_true[:,warmup_steps:,]
    Y_pred_slice=Y_pred[:,warmup_steps:,]
########
    loss     =tf.losses.mean_squared_error(labels=Y_true_slice, predictions=Y_pred_slice)
    loss_mean=tf.reduce_mean(loss)
    return loss_mean
##### OPTIMIZER 
optimizer=RMSprop(lr=1e-3) 
model.compile(loss=loss_mse_warmup,optimizer=optimizer) 
model.summary()
path_checkpoint='23_checkpoint.keras'
callback_checkpoint=ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',verbose=1,save_weights_only=True,save_best_only=True)

callback_early_stopping=EarlyStopping(monitor='val_loss',patience=5, verbose=1)
callback_tensorboard =TensorBoard(log_dir='./23_logs/',histogram_freq=0,write_graph=False)
callback_reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,min_lr=1e-4,patience=0,verbose=1)
callbacks = [callback_early_stopping,callback_checkpoint,callback_tensorboard,callback_reduce_lr]
### MODEL FIT GENERATOR
model.fit_generator(generator=generator,epochs=20,steps_per_epoch=100,validation_data=validation_data, callbacks=callbacks)
try: 
    model.load_weights(path_checkpoint)
except Exception as error:
    print('Error trying to load checkpoint')
    print(error)
result = model.evaluate(x=np.expand_dims(X_test_scaled, axis=0),y=np.expand_dims(Y_test_scaled, axis=0))
#print("loss (test-set):", result)
# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))
        
#### GENERATE PREDICTION
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

def plot_comparison(XPred_scaled,YTrain):
    XPred_scaled  =np.expand_dims(XPred_scaled, axis=0)
    YPred_scaled  =model.predict(XPred_scaled)
    YPred         =y_scaler.inverse_transform(YPred_scaled[0])
    #LL1           =len(YPred)
    #LL2           =len(YTrain)
    #YPredc        =YPred  ### THIS IS USED TO CONCANATE RESULTS
    ###### INTERPOLATE DATA
    test_range    =list(range(0,L5))
    train_range   =list(range(0,L1))
    #YTrain        =YTrain  ###CONVERT TO 1D ARRAY
    YTrain        =YTrain['target']
    YTrain_int    =np.interp(test_range,train_range,YTrain)
    YTrain2       =YTrain_int.T 
    YPred=YPred[:,0]
    ######## FOR PLOTTING PURPOSE
    TRUE_signal=YTrain2#[0:10:10000]
    PRED_signal=YPred#[0:10:10000]
    test_range =test_range#[0:10:10000]
    ##### NORMALIZATION OF THE SIGNAL
    #TRUE_signal = (TRUE_signal - TRUE_signal.mean()) / (TRUE_signal.max() - TRUE_signal.min())
    #PRED_signal = (PRED_signal - PRED_signal.mean()) / (PRED_signal.max() - PRED_signal.min())
    #PRED_signal       =NCoef_TRUE_signal*PRED_signal
    #print(min(TRUE_signal),max(TRUE_signal))
    #print(min(PRED_signal),max(PRED_signal))
    # Plot and compare the two signals.
    plt.figure(figsize=(15,5))
    plt.scatter (test_range,TRUE_signal,c='b',s=40,alpha=0.5, label='True')
    plt.scatter (test_range,PRED_signal,c='r',s=40,alpha=0.5, label='Predicted')
    #plt.scatter(TRUE_signal, label='true')
    #plt.scatter(PRED_signal, label='pred')
    p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        # Plot labels etc.
    plt.ylabel(target_names[signal])
    plt.legend()
    plt.show()
    ############### RMSE ESTIMATION
    ERROR=np.sqrt(np.mean((TRUE_signal-PRED_signal)**2))
    print ("RSME",format(ERROR, '.3f'))
    #### SAVING DATA
    data_test['target']=PRED_signal
    target = data_test['target']
    card_id= data_test['card_id']
    df3= pd.concat([card_id,target], axis=1)
    DatatoSubmit=df3.to_csv('Submission_RNN.csv',index=False)
plot_comparison(XPred_scaled,YTrain)
##### TOTAL TIME
t1   =time.time()
total=t1-t0
print ('Time spent is about:', np.round(total), 'seconds')
