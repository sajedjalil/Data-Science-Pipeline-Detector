# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import gc
import os
import numpy as np # linear algebra
np.random.seed(666)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,KFold
from subprocess import check_output
import random
#print(check_output(["ls", "/input"]).decode("utf8"))

#os.chdir("D:\work\DeepLearning\kaggle\StatoilC-CORE_Iceberg_Classifier\src")

#Load data
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
print("done!")

print(train)
#ytt code here
#original :InShape = 75
#          run_epoch = 25
#          run_batch = 32
#          loss
# output later dense 256-64
InShape = 75
run_epoch = 140
run_batch = 32


# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(InShape, InShape) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(InShape, InShape) for band in train["band_2"]])

#ytt

X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)

#x_band3 = np.divide(x_band1, x_band2, out=np.zeros_like(x_band1), where=x_band2!=0)
#X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
#                          , x_band2[:, :, :, np.newaxis]
#                         , x_band3[:, :, :, np.newaxis]], axis=-1)


#sar to rgb
#band = {'HH': 0, 'HV': 1}
#r = tif[:, :, band['HH']]
#g = tif[:, :, band['HV']]

#hh = r.astype(np.float64)
#hv = g.astype(np.float64)
#b = np.divide(hh, hv, out=np.zeros_like(hh), where=hv!=0)
#rgb = np.dstack((r, g, b.astype(np.uint16)))


X_angle_train = np.array(train.inc_angle)
#ytt
#X_angle_train = np.zeros(len(X_angle_train))

y_train = np.array(train["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(InShape, InShape) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(InShape, InShape) for band in test["band_2"]])

#ytt
#x_band3 = np.divide(x_band1, x_band2, out=np.zeros_like(x_band1), where=x_band2!=0)

#X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
#                          , x_band2[:, :, :, np.newaxis]
#                         , x_band3[:, :, :, np.newaxis]], axis=-1)

X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)

X_angle_test = np.array(test.inc_angle)

#ytt
#X_angle_test = np.zeros(len(X_angle_test))
#X_train = np.array(X_train, np.float32)/255.
#X_test  = np.array(X_test, np.float32)/255.


from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,AveragePooling2D,UpSampling2D
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,ReduceLROnPlateau
import keras.backend as K

from keras.applications.vgg16 import VGG16

def get_callbacks(filepath, patience=2,epsilon = 1e-4):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    rl =              ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=15,
                               verbose=1,
                               epsilon=epsilon,
                               min_lr = 1e-5,
                               mode='min')
    return [es, msave,rl]


#             ReduceLROnPlateau(monitor='val_dice_loss',
#                               factor=0.1,
#                               patience=4,
#                               verbose=1,
#                               epsilon=1e-4,
#                               mode='max'),


def get_model_fusion(automodel):
    # create a placeholder for an encoded (32-dimensional) input
    #input_img = Input(shape=(75, 75, 3), name="input_img")
    input_img = automodel.get_layer('input_img').output
    input_2 = Input(shape=[1], name="angle")
    # retrieve the last layer of the autoencoder model
    encoder_out = automodel.get_layer('encoded').output
    # 10* 10
    #img_1 = GlobalMaxPooling2D() (encoder_out)
    img_1 = GlobalAveragePooling2D() (encoder_out)
    #add global poool ??
    img_concat =  (Concatenate()([img_1,input_2 ]))
    dense_layer = Dense(512, activation='relu', name='fcc1')(img_concat )
    #dense_layer = BatchNormalization(momentum=0)(dense_layer )
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(256, activation='relu', name='fcc2')(dense_layer)
    #dense_layer = BatchNormalization(momentum=0)(dense_layer )
    dense_layer = Dropout(0.2)(dense_layer)    
    predictions = Dense(1, activation='sigmoid')(dense_layer)
    
    #model = Model(input=base_model.input, output=predictions)
    model = Model([input_img,input_2], predictions)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer=RMSprop(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def get_model_autoencoder():
    
    input_img = Input(shape=(75, 75, 3), name="input_img")  # adapt this if using `channels_first` image data format
    bn_model = 0
    p_activation = "elu"
    x = Conv2D(16, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')((BatchNormalization(momentum=bn_model))(input_img))
    x = Conv2D(16, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
#    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') ((BatchNormalization(momentum=bn_model))(input_img))
#    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
#    #img_1 = BatchNormalization()(img_1)
#    img_1 = MaxPooling2D((2,2)) (img_1)
#    img_1 = Dropout(0.3)(img_1)
#    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1) 
#    #img_1 = BatchNormalization()(img_1)
#    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
#    #img_1 = BatchNormalization()(img_1)
#
#    img_1 = MaxPooling2D((2,2)) (img_1)
#    img_1 = Dropout(0.3)(img_1)
#    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
#    #img_1 = BatchNormalization()(img_1)
#    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)

    encoded = MaxPooling2D((2, 2), padding='same', name="encoded")(x)

    x = Conv2D(128, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(encoded)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(16, (3, 3), activation='elu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='valid',name="decoded")(x)
    
    autoencoder = Model(input_img, decoded)
    optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy',)
    return autoencoder



def conv_block(x, nfilter=8, ksize=3, stride=1, nblock=2, p_act='elu'): 
     
     for i in range(nblock): 
         x = Conv2D(filters=nfilter, kernel_size=(ksize, ksize), strides=(stride, stride),   
                   activation=p_act, 
                   padding='same', kernel_initializer='he_uniform')(x) 
          
     return x 



    
def get_model():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(InShape, InShape, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")

#block test    
#    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
#    img_1 = conv_block(img_1, nfilter=16, ksize=3, stride=1, nblock=2, p_act=p_activation)
#    img_1 = MaxPooling2D((2,2)) (img_1)
#    img_1 = Dropout(0.2)(img_1)
#    img_1 = conv_block(img_1, nfilter=32, ksize=3, stride=1, nblock=2, p_act=p_activation)
#    img_1 = MaxPooling2D((2,2)) (img_1)
#    img_1 = Dropout(0.2)(img_1)
#    img_1 = conv_block(img_1, nfilter=64, ksize=3, stride=1, nblock=2, p_act=p_activation)
#    img_1 = MaxPooling2D((2,2)) (img_1)
#    img_1 = Dropout(0.2)(img_1)
#    img_1 = conv_block(img_1, nfilter=128, ksize=3, stride=1, nblock=2, p_act=p_activation)
#    img_1 = MaxPooling2D((2,2)) (img_1)
#    img_1 = Dropout(0.2)(img_1)
#    img_1 = conv_block(img_1, nfilter=128, ksize=3, stride=1, nblock=3, p_act=p_activation)  

#    img_1 = Dropout(0.2)(img_1)
#    img_1 = conv_block(img_1, nfilter=256, ksize=3, stride=1, nblock=3, p_act=p_activation)     

#original    
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') ((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #img_1 = BatchNormalization()(img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.3)(img_1)
    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    
    #img_1 = BatchNormalization()(img_1)
    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #img_1 = BatchNormalization()(img_1)

    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.3)(img_1)
    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #img_1 = BatchNormalization()(img_1)
    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #img_1 = BatchNormalization()(img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.3)(img_1)
    img_1 = Conv2D(128, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #img_1 = BatchNormalization()(img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.3)(img_1)
    #img_1 = GlobalMaxPooling2D() (img_1)
    img_1 = GlobalAveragePooling2D() (img_1)

    #auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
 
    img_2 = Conv2D(128, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') ((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dropout(0.3)(img_2)
    #img_2 = GlobalMaxPooling2D() (img_2)
    img_2 = GlobalAveragePooling2D() (img_2)
    
    #img_2 = Flatten()((BatchNormalization(momentum=bn_model))(input_1))
    
    #img_concat =  (Concatenate()([img_1,img_2 ]))
    img_concat =  (Concatenate()([img_1,input_2 ]))
    #img_concat =  (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))
    #img_concat =  (Concatenate()([img_1, img_2, input_2]))
    
    #img_concat =  (Concatenate()([img_1, BatchNormalization(momentum=bn_model)(input_1), BatchNormalization(momentum=bn_model)(input_2)]))
    
    dense_ayer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(512, activation=p_activation)(img_concat) ))
    dense_ayer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(256, activation=p_activation)(dense_ayer) ))


    output = Dense(1, activation="sigmoid")(dense_ayer)
    
    model = Model([input_1,input_2],  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer =Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    #optimizer=RMSprop(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model



#start loop here

num_folds = 4
count_folds = 0

#random: 123,456
kf = KFold(n_splits=num_folds, shuffle=True, random_state=123)

#X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
#                    , X_angle_train, y_train, random_state=123, train_size=0.75)


#ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)
#for ids_train_split, ids_valid_split in kf.split(ids_train):

#random.seed(48)
#random.shuffle(ids_train_split)
#random.shuffle(ids_valid_split)
#random.seed(123)
prediction = np.zeros(shape=(len(X_test),1) , dtype = np.float)
kfoldcount = 1
#model = get_model()   
#model.summary()

for train_index,valid_index in kf.split(X_train):

    random.seed(123)
    random.shuffle(train_index)
    random.shuffle(valid_index)



    X_train_in = X_train[train_index]
    
    X_valid_in =  X_train[valid_index]
    X_angle_train_in =  X_angle_train[train_index]
    X_angle_valid_in =  X_angle_train[valid_index]
    y_train_in = y_train[train_index]
    y_valid_in = y_train[valid_index]
     
    print('Training on {} samples'.format(len(y_train_in)))
    print('Validating on {} samples'.format(len(y_valid_in)))
    print("kfoldcount {}".format(kfoldcount))


    print("Load model in kfold count {}".format(kfoldcount))

    automodel = get_model_autoencoder()
    automodel.summary()
    
    
    #train auto encoder
    if(kfoldcount == 1):
       # X_all = np.concatenate([X_test,X_train],axis=0)
        #print(X_all.shape)
        automodel.fit(X_train,X_train,
                    epochs=40,
                    batch_size=64,
                    shuffle=True,
                    #validation_data=(X_train, X_train),
                    validation_split = 0.2,
                    #verbose = 2,
                    callbacks=[EarlyStopping('val_loss', patience=15, mode="min")])
        automodel.save_weights('auto_weights0.hdf5') 
    automodel.load_weights('auto_weights0.hdf5') 
    
    model = get_model_fusion(automodel)
    model.summary()
    file_path = 'model_weights{}.hdf5'.format(kfoldcount)    
    #load golden
    #model.load_weights(filepath="D:\work\python\DeepLearning\kaggle\StatoilC-CORE_Iceberg_Classifier\src\weights\kfold4_single_model_weights3_los1620.hdf5")    
    #file_path = "model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=30,epsilon = 1e-3)
    
    #model = get_model()
    
    #model =get_model_LB18()
    #YTT FAST REPEAT
    #if(kfoldcount>1):
    #    model.load_weights('weights/model_weights{}.hdf5'.format(kfoldcount-1))     
    #model.load_weights('weights/best_weights{}.hdf5'.format(kfoldcount-1))     
    
#    #YTT
    datagen = ImageDataGenerator(
                rotation_range = 20,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                zoom_range = 0.2,
                horizontal_flip = True,
                vertical_flip = True)


    #original from inverse xxx
    def generate_generator_multiple(generator,data1,data2 ,y_label, batch_size):
    
        genX1 = generator.flow(data1,y_label,batch_size =batch_size,shuffle=False)
        
        while True:
    
            for start in range(0, data2.size, batch_size):
                X1i = genX1.next()
                # 1203   1216           
                end = min(start + batch_size, data2.size)
                y_out_batch = data2[start:end]
                
                yield [X1i[0], y_out_batch ], X1i[1]
                #yield [X1i[0], X2i[0]], X1i[1]  #Yield both images and their mutual label
            #break;   
    
    
    inputgenerator=generate_generator_multiple(generator=datagen,
                                               data1=X_train_in,
                                               data2=X_angle_train_in,
                                               y_label = y_train_in,
                                               batch_size=run_batch) 
    
#    validgenerator=generate_generator_multiple(generator=datagen,
#                                               data1=X_test,
#                                               data2=X_angle_test,
#                                               y_label = y_valid_in,
#                                               batch_size=run_batch) 
    
    #K.set_value(model.optimizer.lr, 0.001)
    #basic
    model.fit_generator(inputgenerator,
                validation_data=([X_valid_in, X_angle_valid_in], y_valid_in), callbacks =callbacks,
                                #steps_per_epoch = len(X_train_in) / run_batch,
                steps_per_epoch = np.ceil(float(len(X_train_in))*5 / float(run_batch)),
                epochs =run_epoch, verbose = 2)
    
    
    model.load_weights(filepath=file_path)

    print("Train kfold split evaluate:")
    print(model.evaluate([X_train_in, X_angle_train_in], y_train_in, verbose=1, batch_size=200))
    print("####################")
    print("watch kfold split evaluate:")
    print(model.evaluate([X_valid_in, X_angle_valid_in], y_valid_in, verbose=1, batch_size=200))
    
    print("All Train list evaluate:")
    print(model.evaluate([X_train, X_angle_train], y_train, verbose=1, batch_size=200))
    print("####################")
          
    print("in prediction")
    cpred= model.predict([X_test, X_angle_test], verbose=1, batch_size=200) 
    print(type(cpred))
    print(cpred.shape) 
    print(cpred)    
    prediction =  prediction +cpred
    #prediction += model.predict([X_test, X_angle_test], verbose=1, batch_size=200)  
    print(type(prediction))
    print(prediction.shape)
    print(prediction)
 
    #output single model submission
    cpred[cpred <0.005] = 0.005 
    csubmission = pd.DataFrame({'id': test["id"], 'is_iceberg': cpred.reshape((cpred.shape[0]))})
    csubmission.head(10)
    csubmission.to_csv('SingleSubmission{}.csv'.format(kfoldcount), index=False)    
    
    #tf.reset_default_graph() 
    K.clear_session()
    gc.collect()
    kfoldcount = kfoldcount+1
    print("end") 
    #raise

#loop end
print("out prediction")
print(prediction)
print(prediction.shape[0])
prediction /= num_folds  
prediction[prediction <0.005] = 0.005 
#model.load_weights(filepath="D:\work\DeepLearning\kaggle\StatoilC-CORE_Iceberg_Classifier\src\weights\model_weights3.hdf5")    
    
#prediction = model.predict([X_test, X_angle_test], verbose=1, batch_size=200)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.head(10)

submission.to_csv("submission.csv", index=False)