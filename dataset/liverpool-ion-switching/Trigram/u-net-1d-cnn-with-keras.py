# Credit to the original author. I HAVE CHANGED TO 100 EPOCHS AS WELL AS ADDING AN EXTRA CONV1-D LAYER.
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Reshape, Conv1D, Conv2D, BatchNormalization, Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply
from keras.models import Model
from keras.objectives import mean_squared_error
from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from keras.initializers import random_normal
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import KFold, train_test_split

df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
df_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
train_input = df_train["signal"].values.reshape(-1,4000,1)#number_of_data:1250 x time_step:4000
train_input_mean = train_input.mean()
train_input_sigma = train_input.std()
train_input = (train_input-train_input_mean)/train_input_sigma
test_input = df_test["signal"].values.reshape(-1,10000,1)#
test_input = (test_input-train_input_mean)/train_input_sigma
train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1,4000,11)#classification
idx = np.arange(train_input.shape[0])
train_idx, val_idx = train_test_split(idx, random_state = 111,test_size = 0.2)
val_input = train_input[val_idx]
train_input = train_input[train_idx] 
val_target = train_target[val_idx]
train_target = train_target[train_idx] 

print("train_input:{}, val_input:{}, train_target:{}, val_target:{}".format(train_input.shape, val_input.shape, train_target.shape, val_target.shape))

def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = GlobalAveragePooling1D()(x_in)
    x = Dense(layer_n//8, activation="relu")(x)
    x = Dense(layer_n, activation="sigmoid")(x)
    x_out=Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = Add()([x_in, x])
    return x  

def Unet(input_shape=(None,1)):
    layer_n = 64
    kernel_size = 7
    depth = 2

    input_layer = Input(input_shape)    
    input_layer_1 = AveragePooling1D(5)(input_layer)
    input_layer_2 = AveragePooling1D(25)(input_layer)
    input_layer_3 = AveragePooling1D(125)(input_layer)
    
    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n*2, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])    
    x = cbr(x, layer_n*3, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1)
    out_2 = x

    x = Concatenate()([x, input_layer_2])    
    x = cbr(x, layer_n*4, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1)
    
    x = UpSampling1D(5)(x)
    x = Concatenate()([x, out_2])
    x = cbr(x, layer_n*3, kernel_size, 1, 1)

    x = UpSampling1D(5)(x)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n*2, kernel_size, 1, 1)

    x = UpSampling1D(5)(x)
    x = Concatenate()([x, out_0])
    x = cbr(x, layer_n, kernel_size, 1, 1)    

    x = Conv1D(11, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = Activation("softmax")(x)
    
    model = Model(input_layer, out)
    
    return model

def augmentations(input_data, target_data):
    #flip
    if np.random.rand()<0.5:    
        input_data = input_data[::-1]
        target_data = target_data[::-1]

    return input_data, target_data

def Datagen(input_dataset, target_dataset, batch_size, is_train=False):
    x=[]
    y=[]
  
    count=0
    idx_1 = np.arange(len(input_dataset))
    #idx_2 = np.arange(len(input_dataset))
    np.random.shuffle(idx_1)
    #np.random.shuffle(idx_2)

    while True:
        for i in range(len(input_dataset)):
            input_data = input_dataset[idx_1[i]]
            target_data = target_dataset[idx_1[i]]
            #input_data_mix = input_dataset[idx_2[i]]
            #target_data_mix = target_dataset[idx_2[i]]

            if is_train:
                input_data, target_data = augmentations(input_data, target_data)
                #input_data_mix, target_data_mix = augmentations(input_data_mix, target_data_mix)
                
            x.append(input_data)
            y.append(target_data)
            count+=1
            if count==batch_size:
                x=np.array(x, dtype=np.float32)
                y=np.array(y, dtype=np.float32)
                inputs = x
                targets = y       
                x = []
                y = []
                count=0
                yield inputs, targets

class macroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis=2).reshape(-1)

    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)
        f1_val = f1_score(self.targets, pred, average="macro")
        print("val_f1_macro_score: ", f1_val)
                
def model_fit(model, train_inputs, train_targets, val_inputs, val_targets, n_epoch, batch_size=32):
    hist = model.fit_generator(
        Datagen(train_inputs, train_targets, batch_size, is_train=True),
        steps_per_epoch = len(train_inputs) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen(val_inputs, val_targets, batch_size),
        validation_steps = len(val_inputs) // batch_size,
        callbacks = [lr_schedule, macroF1(model, val_inputs, val_targets)],
        shuffle = False,
        verbose = 1
        )
    return hist

def lrs(epoch):
    if epoch<35:
        lr = learning_rate
    elif epoch<50:
        lr = learning_rate/10
    else:
        lr = learning_rate/100
    return lr

K.clear_session()
model = Unet()
learning_rate=0.0005
n_epoch=200
batch_size=8
lr_schedule = LearningRateScheduler(lrs)
model.compile(loss=categorical_crossentropy, 
              optimizer=Adam(lr=learning_rate), 
              metrics=["accuracy"])
hist = model_fit(model, train_input, train_target, val_input, val_target, n_epoch, batch_size)
pred = np.argmax((model.predict(val_input)+model.predict(val_input[:,::-1,:])[:,::-1,:])/2, axis=2).reshape(-1)
gt = np.argmax(val_target, axis=2).reshape(-1)
pred = np.argmax((model.predict(test_input)+model.predict(test_input[:,::-1,:])[:,::-1,:])/2, axis=2).reshape(-1)
df_sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
df_sub.open_channels = np.array(np.round(pred,0), np.int)
df_sub.to_csv("submission.csv",index=False)