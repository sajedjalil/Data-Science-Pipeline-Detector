# Original kernel: Keras Xception model [0.68++ on PL] + weights
#   https://www.kaggle.com/mihaskalic/keras-xception-model-0-68-on-pl-weights
# Original code from Miha Skalic

# This version avoids the need to install the sallamander multiGPU keras library (since the class is already included in the code)

# NOTE: This version has issues saving and loading models, and may only work when run from start to end.


##############################
###        IMPORTS         ###
##############################

import keras.backend as K


from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Concatenate, Lambda, Flatten, Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model, Model, save_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

import math
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

# MultiGPU model build on top of
# https://github.com/sallamander/multi-gpu-keras-tf/
#from multiGPU import MultiGPUModel



##############################
###        CONFIG          ###
##############################

TEST_MODE       = False

num_gpus_to_use = 2

model_name      = "xception_v2"
models_folder   = "./models/"

img_width       = 180
img_height      = 180


if TEST_MODE == True:
        
    batch_size      = 8 * num_gpus_to_use  # 258
    
    num_epochs      = [1, 1]
    learning_rate   = [0.1, 0.1, 0.1]

    sub_model_names  = ["trainhead_test", "blocks_11_test", "whole_model_test"]

    train_data_dir  = '../input/train'
    val_data_dir    = '../input/validation'
    classnames      = pickle.load(open("../input/class_order_test.pkl", "rb"))
else:
    
    batch_size      = 128 * num_gpus_to_use  # 258
    
    num_epochs      = [3, 8]
    learning_rate   = [0.001, 0.00025, 0.00025]
    
    sub_model_names  = ["trainhead", "blocks_11", "whole_model"]

    train_data_dir  = '../input/train'
    val_data_dir    = '../input/validation'
    classnames      = pickle.load(open("../input/class_order.pkl", "rb"))



##############################
###       MULTI GPU        ###
##############################
class MultiGPUModel(Model):
    def __init__(self, serial_model, gpu_ids, batch_size):
        self.serial_model = serial_model
        self._parallelize_model(serial_model, gpu_ids, batch_size)

    def __getattribute__(self, key):
        serial_attributes = {
            'load_weights', 'save_weights',
            'summary', 'to_yaml', 'save', 'to_json',
        }
        if key in serial_attributes:
            return getattr(self.serial_model, key)
        return super().__getattribute__(key)

    def _parallelize_model(self, model, gpu_ids, batch_size):
        all_sliced_outputs = []
        for gpu_id in gpu_ids:
            with tf.device('/gpu:{}'.format(gpu_id)):
                sliced_inputs = []
                for model_input in model.inputs:
                    idx_min = gpu_id * batch_size
                    idx_max = (gpu_id + 1) * batch_size
                    input_slice = Lambda(
                        lambda x: x[idx_min:idx_max],
                        lambda shape: shape
                    )(model_input)
                    sliced_inputs.append(input_slice)

                sliced_outputs = model(sliced_inputs)
                all_sliced_outputs.append(sliced_outputs)

        with tf.device('/cpu:0'):
            outputs = Concatenate(axis=0)(all_sliced_outputs)

            super().__init__(inputs=model.inputs, outputs=outputs)



##############################
###         MODEL          ###
##############################
model0 = Xception(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(img_width, img_height, 3))

for lay in model0.layers:
    lay.trainable = False
    
x = model0.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.2)(x)
x = Dense(len(classnames), activation='softmax', name='predictions')(x)
model0 = Model(model0.input, x)

# Train on 2GPUs
model = MultiGPUModel(model0, [0, 1], int(batch_size/num_gpus_to_use))




##############################
###       GENERATORS       ###
##############################
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = classnames,
        class_mode = 'categorical')

val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = classnames,
        class_mode = 'categorical')

os.makedirs("./models", exist_ok=True)






##############################
###                        ###
###         TRAIN          ###
###                        ###
##############################

sub_model_save_paths = []
callbacks = []
had_been_run = []

for idx,subm in enumerate(sub_model_names):

    sub_model_save_paths.append(models_folder + "mod_and_outp_" + model_name + "_" + subm + ".hdf5")

    callbacks.append([ModelCheckpoint(monitor='val_loss',
                             filepath= models_folder + model_name + "_" + subm + '_{epoch:03d}-{val_loss:.7f}.hdf5',
                             save_best_only=False,
                             save_weights_only=False,
                             mode='max'),
                                TensorBoard(log_dir='logs/{}'.format(model_name))])

    if os.path.isfile(sub_model_save_paths[idx]):
        had_been_run.append(True)
    else:
        had_been_run.append(False)



########################
#      train HEAD      #
########################

#sub_model_names  = ["trainhead", "blocks11", "whole_model"]

sub_model_order = 0  # Train head (base 0)

if had_been_run[sub_model_order]:
    print("\n We FOUND",sub_model_save_paths[sub_model_order],".\n\t Since it seems to have been already launched, we will skip this model.\n\n")

else:

    print("\n Starting optimization of model:",sub_model_names[sub_model_order])

    model.compile(loss='categorical_crossentropy', 
                    optimizer=SGD(lr=learning_rate[sub_model_order], momentum=0.9), 
                    metrics=[top_k_categorical_accuracy, 'accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=math.ceil(2000000 / batch_size),
                        verbose=0,
                        callbacks=callbacks[sub_model_order],
                        validation_data=validation_generator,
                        initial_epoch=0,
                        epochs = num_epochs[sub_model_order],
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers = 8,
                        validation_steps=math.ceil(10000 / batch_size))

    model.save(sub_model_save_paths[sub_model_order])




###############################
#       train BLOCKS 11+      #
###############################

sub_model_order = 1  # blocks 11+ (base 0)

if had_been_run[sub_model_order]:
    print("\n We FOUND",sub_model_save_paths[sub_model_order],".\n\t We will skip this model.\n\n")

else:

    if had_been_run[sub_model_order - 1]:
        model = load_model(sub_model_save_paths[sub_model_order - 1])  # returns a compiled model identical to the previous one

    print("\n Starting optimization of model:",sub_model_names[sub_model_order])
    print('\n\n\n')
    print('Debugging Blocks11 layer issue:')
    for lay in model.layers:
        print(lay)

    print('\n\n\n')
    for clayer in model.layers[num_gpus_to_use + 1].layers:
        print("trainable:", clayer.name)

        if clayer.name.split("_")[0] in ["block{}".format(i) for i in range(10, 15)]:
            clayer.trainable = True
    
            
    model.compile(loss='categorical_crossentropy', 
                    optimizer=Adam(lr=learning_rate[sub_model_order]), 
                    metrics=[top_k_categorical_accuracy, 'accuracy'])


    model.fit_generator(generator=train_generator,
                        steps_per_epoch=math.ceil(2000000 / batch_size),
                        verbose=1,
                        callbacks=callbacks[sub_model_order],
                        validation_data=validation_generator,
                        initial_epoch=3,
                        epochs = num_epochs[sub_model_order],
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers = 8,
                        validation_steps=math.ceil(10000 / batch_size))



    model.save(sub_model_save_paths[sub_model_order])
    #save_model(model, sub_model_save_paths[sub_model_order])




############################
#     train WHOLE model    #
############################

sub_model_order = 2  # whole model (base 0)

if had_been_run[sub_model_order]:
    print("\n We FOUND",sub_model_save_paths[sub_model_order],".\n\t We will skip this model.\n\n")

else:

    if had_been_run[sub_model_order - 1]:
        model = load_model(sub_model_save_paths[sub_model_order - 1])  # returns a compiled model identical to the previous one

    print("\n Starting optimization of model:",sub_model_names[sub_model_order])
    print('\n\n\n Debugging whole model model: ')
    for lay in model.layers:
        print(lay)
    print('\n\n\n')
    

    for clayer in model.layers[num_gpus_to_use + 1].layers:
        clayer.trainable = True

    # Note you need to recompile the whole thing. Otherwise you are not traing first layers    
    model.compile(loss='categorical_crossentropy', 
                    optimizer=Adam(lr=learning_rate[sub_model_order]), 
                    metrics=[top_k_categorical_accuracy, 'accuracy'])


    init_epochs = 8  # We pretrained the model already

    # Keep training for as long as you like.
    for i in range(100):
        # gradually decrease the learning rate
        K.set_value(model.optimizer.lr, 0.95 * K.get_value(model.optimizer.lr))
        start_epoch = (i * 2)
        epochs = ((i + 1) * 2)    
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=math.ceil(2000000 / batch_size),
                            verbose=1,
                            callbacks=callbacks[sub_model_order],
                            validation_data=validation_generator,
                            initial_epoch=start_epoch + init_epochs,
                            epochs=epochs + init_epochs,
                            use_multiprocessing=True,
                            max_queue_size=10,
                            workers = 8,
                            validation_steps=math.ceil(10000 / batch_size))


        model.save(model, models_folder + "mod_and_outp_" + model_name + "_" + subm + "_ep_" + str(i) + ".hdf5")