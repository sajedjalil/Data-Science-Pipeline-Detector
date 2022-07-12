########################################################################################################################
## Author: Amro Tork amtc2018@gmail.com
########################################################################################################################

from __future__ import print_function

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

from skimage.io import imread,imsave
from skimage.exposure import is_low_contrast

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, SGD

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy

from deepsense import neptune

from PIL import Image

import io
import bson
import os
import gc

import threading
import pickle

########################################################################################################################
## Global Variables
########################################################################################################################
## Total number of images in train = 12371293
## Total number of products in train = 7069896
## Total number of categories in train = 5270
## Total number of images in test = 3095080
## Total number of products in test = 1768182

batch_size = 250
test_batch_size = 500

im_size = 180

num_dicts = 7069896
num_dicts_test = 1768182

total_num_train_imgs = 12371293
total_num_test_imgs = 3095080


num_of_tuning_training = 20

model_name = "incep_resnet"
models_savename = "/output/models/" + model_name
model_file = "/input/inception_v2.hdf5"
train_metadata = "/input/all_images_categories.csv"
class_order_file = "/input/class_order.pkl"
input_train_file = '/public/Cdiscount/train.bson'
cat_names_file = '/public/Cdiscount/category_names.csv'

valid_size = 0.15
test_size = 0.15

ctx = neptune.Context()

TARGET_SZ = (im_size,im_size)

if K.image_data_format() == 'channels_first':
    input_shape = (3, im_size, im_size)
else:
    input_shape = (im_size, im_size, 3)
    
########################################################################################################################
## Procedures
########################################################################################################################
def get_image_array(fimg,img_gen=None):
    img = imread(fimg)
    if is_low_contrast(img):
        return None
    x = image.img_to_array(img)
    if img_gen is not None:
        x = img_gen.random_transform(x)
        x = img_gen.standardize(x)
    else:
        x = preprocess_input(x)
    return x

def make_top_n_layers_trainable(model,num_layers):
    if num_layers != "all":
        for clayer in model.layers[-num_layers:]:
            clayer.trainable = True
            print("Trainable:", clayer.name)
    else:
        for clayer in model.layers:
            clayer.trainable = True
        print("All layers trainable.")
    return model

def create_model(num_classes,input_shape):
    model0 = InceptionResNetV2(include_top=False,weights="imagenet",input_shape=input_shape,pooling="avg")
    for clayer in model0.layers:
        clayer.trainable = False
    x = model0.output
    x = Dropout(0.1)(x)
    x = Dense(5000,activation="relu",name="final_hidden")(x)
    x = Dropout(0.1)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(model0.input, x)

    for clayer in model.layers:
        clayer.trainable = False
    return model

def get_cat_name(cat_num):
    global cat_name_df
    #print(cat_name_df.head())
    #print(cat_name_df.dtypes)
    found_cat = cat_name_df[cat_name_df["category_id"] == int(cat_num)]
    #print(found_cat)
    cat_level1 = found_cat["category_level1"]
    cat_level2 = found_cat["category_level2"]
    cat_level3 = found_cat["category_level3"]
    return "{}_{}_{}".format(str(cat_level1),str(cat_level2),str(cat_level3))

def get_image_bytes(image_id, data_df, fh):
    img_info = data_df[data_df["img_id"] == image_id]
    item_loc = img_info["item_loc"].values[0]
    item_len = img_info["item_len"].values[0]
    pic_ind = img_info["pic_ind"].values[0]
    fh.seek(item_loc)
    item_data = fh.read(item_len)
    d = bson.BSON.decode(item_data)

    return io.BytesIO(d["imgs"][pic_ind]['picture'])

def get_image(image_id, data_df, fh,img_gen=None):
    picture = get_image_array(get_image_bytes(image_id, data_df, fh),img_gen)
    return picture

def train_finetune_model(model,mod_save_name,train,valid,train_filehandle,batch_size,class_order,lock):
    es = EarlyStopping(monitor='val_loss',
                  min_delta=0,
                  patience=1,
                  verbose=0, mode='auto')

    nep_follow_cb = NeptuneCallback(valid,batch_size,20,lock,class_order,train_filehandle,200)

    callbacks = [ModelCheckpoint(monitor='val_loss',
                                 filepath=mod_save_name + '_{epoch:03d}-{val_loss:.7f}.hdf5',
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='max'), es, nep_follow_cb]
    init_epochs = 5  # We pretrained the model already

    # Prepare generators
    train_datagen = ImageDataGenerator(
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    train_generator = BSONIterator(train_filehandle, train, class_order,train_datagen, lock, batch_size=batch_size)
    valid_generator = BSONIterator(train_filehandle, valid, class_order, None, lock, batch_size=batch_size)


    # Train head
    model = make_top_n_layers_trainable(model,5)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.9),
                  metrics=[top_k_categorical_accuracy, 'accuracy'])
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=np.ceil(float(len(train))/ batch_size),
                        verbose=1,
                        callbacks=callbacks,
                        #use_multiprocessing=True,
                        #workers=4,
                        validation_data=valid_generator,
                        initial_epoch=0,
                        epochs=init_epochs,
                        max_queue_size=10,
                        validation_steps=np.ceil(float(len(valid)) / batch_size))

    # Train the top blocks
    model = make_top_n_layers_trainable(model, 20)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01),metrics=[top_k_categorical_accuracy, 'accuracy'])
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=np.ceil(float(len(train)) / batch_size),
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        #use_multiprocessing=True,
                        #workers=4,
                        initial_epoch=0,
                        epochs=init_epochs,
                        max_queue_size=10,
                        validation_steps=np.ceil(float(len(valid)) / batch_size))

    # Train the whole model
    print("## Train whole model")
    model = make_top_n_layers_trainable(model, "all")

    # Note you need to recompile the whole thing. Otherwise you are not traing first layers
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.05),metrics=[top_k_categorical_accuracy, 'accuracy'])

    # Keep training for as long as you like.
    for i in range(num_of_tuning_training):
        # gradually decrease the learning rate
        K.set_value(model.optimizer.lr, 0.9 * K.get_value(model.optimizer.lr))
        start_epoch = (i * 2)
        epochs = ((i + 1) * 2)
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=np.ceil(float(len(train)) / batch_size),
                            verbose=1,
                            callbacks=callbacks,
                            #use_multiprocessing=True,
                            #workers=4,
                            validation_data=valid_generator,
                            initial_epoch=start_epoch + init_epochs,
                            epochs=epochs + init_epochs,
                            max_queue_size=10,
                            validation_steps=np.ceil(float(len(valid)) / batch_size))

    return model

class NeptuneCallback(Callback):
    def __init__(self,img_df,test_size,num_per_epoch,lock,class_order,file_handle,per_batch):
        self.epoch_id = 0
        self.batch_id = 0
        self.img_df = img_df
        self.test_size = test_size
        self.num_per_epoch = num_per_epoch
        self.lock = lock
        self.num_classes = len(class_order)
        self.file = file_handle
        self.class_order = class_order
        self.check_per_batch = per_batch

    def check_test_sample_images(self):
        # Predict the digits for images of the test set.
        test_sample = self.img_df.sample(n=self.test_size)
        batch_x = np.zeros((len(test_sample), im_size, im_size, 3), dtype=K.floatx())
        batch_y = np.zeros((len(test_sample), self.num_classes), dtype=K.floatx())

        b_ind = 0
        for ind, row in test_sample.iterrows():
            # Protect file and dataframe access with a lock.
            with self.lock:
                img_id = row["img_id"]
                #print(img_id)
                pic = get_image(img_id, self.img_df, self.file, None)

                if pic is None:
                    continue
                #print(pic.shape)

                img_cat = row["category"]
                img_ind = np.where(self.class_order == str(img_cat))[0]
                #print(img_ind)
                # print(img_ind)

                # Add the image and the label to the batch (one-hot encoded).
                batch_x[b_ind] = pic
                batch_y[b_ind, img_ind] = 1.0

                b_ind += 1

        preds = model.predict(batch_x[:b_ind, :, :, :])
        pred_labels = np.argmax(preds, axis=1)
        actual_labels = np.argmax(batch_y, axis=1)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        disp_count = 0
        for ind, (p, a) in enumerate(zip(pred_labels, actual_labels)):
            if p != a:
                p_cat_id = self.class_order[p]
                a_cat_id = self.class_order[a]

                p_cat = get_cat_name(p_cat_id)
                a_cat = get_cat_name(a_cat_id)

                ctx.channel_send('false_predictions', neptune.Image(
                    name='[{}] pred_id: {}  true_id: {} ------ pred: {} true: {}'.format(self.batch_id, p_cat_id,
                                                                                         a_cat_id, p_cat, a_cat),
                    description="Actual Score : {:.5f}\nPredicted Score : {:.5f}".format(preds[ind,a],preds[ind,p]),
                    data=Image.open(get_image_bytes(test_sample.iloc[ind]["img_id"], test_sample, self.file))))
                disp_count += 1

                if disp_count > self.num_per_epoch:
                    break

    def on_batch_end(self, batch, logs={}):
        self.batch_id +=1
        ctx.channel_send('Log-loss training', self.batch_id, logs['loss'])
        ctx.channel_send('Accuracy training', self.batch_id, logs['acc'])
        print("## Batch {:d} is complete.".format(self.batch_id))

        if self.batch_id % self.check_per_batch == 0:
            self.check_test_sample_images()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # logging numeric channels
        ctx.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        ctx.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])
        self.check_test_sample_images()



class BSONIterator(Iterator):
    def __init__(self, bson_file_handle, meta_df, class_order,
                 image_data_generator, lock,with_labels=True, batch_size=32, shuffle=False, seed=None):

        self.file = bson_file_handle
        self.meta_df = meta_df
        self.class_order = class_order
        self.with_labels = with_labels
        self.samples = len(meta_df)
        self.num_classes = len(class_order)
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        num_elem = len(index_array[0])
        batch_x = np.zeros((num_elem, im_size, im_size, 3),dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((num_elem, self.num_classes),dtype=K.floatx())

        b_ind = 0
        for i, j in enumerate(index_array[0]):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.meta_df.iloc[j]
                img_id = image_row["img_id"]
                pic = get_image(img_id, self.meta_df, self.file, self.image_data_generator)

                if pic is None:
                    continue

                img_cat = image_row["category"]
                img_ind = np.where(self.class_order == str(img_cat))[0]

                # Add the image and the label to the batch (one-hot encoded).
                batch_x[b_ind] = pic
                if self.with_labels:
                    batch_y[b_ind, img_ind] = 1.0

                b_ind += 1

        if self.with_labels:
            return batch_x[:b_ind,:,:,:], batch_y[:b_ind,:]
        else:
            return batch_x[:b_ind,:,:,:]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

########################################################################################################################
## Start Processing
########################################################################################################################
if __name__ == "__main__":
    ## Step 1:
    ## Extract all images classes for the some samples
    cat_df = pd.read_csv(train_metadata)
    cat_name_df = pd.read_csv(cat_names_file)

    num_classes = len(pd.unique(cat_df["category"]))
    num_imgs = len(cat_df)

    print("## Final number of categories found = {:d}".format(num_classes))
    print("## Final total number of samples = {:d}".format(num_imgs))

    ## Step 2:
    ## Split the training dataset to balanced subsamples
    train_valid, test = train_test_split(cat_df, test_size=test_size, stratify=cat_df["category"])
    train, valid = train_test_split(train_valid, test_size=valid_size, stratify=train_valid["category"])

    ## Sorting to help in random access
    train = pd.DataFrame(train.sort_values(by="img_id"), copy=True)
    valid = pd.DataFrame(valid.sort_values(by="img_id"), copy=True)
    test = pd.DataFrame(test.sort_values(by="img_id"), copy=True)

    ## Saving to files to avoid repeatation.
    train.reset_index(inplace=True,drop=True)
    valid.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)


    print("## Number of train classes = {:d}".format(len(pd.unique(train["category"]))))
    print("## Number of valid classes = {:d}".format(len(pd.unique(valid["category"]))))
    print("## Number of test classes = {:d}".format(len(pd.unique(test["category"]))))
    print("## Total number of training samples = {:d}".format(len(train)))
    print("## Total number of validation samples = {:d}".format(len(valid)))
    print("## Total number of test samples = {:d}".format(len(test)))
    print("## Done preparing data genrators")

    del cat_df
    gc.collect()

    ## Step 3:
    ## Create keras model
    print("## Creating CNN model.")
    model = create_model(num_classes,input_shape)
    print("## Done creating model.")

    ## Step 4:
    ## Train Model
    print("## Training Model")
    os.makedirs("/output/models/",exist_ok=True)
    os.makedirs("/output/logs/",exist_ok=True)
    train_filehandle = open(input_train_file, 'rb')
    f = open(class_order_file, "rb")
    class_order = np.array(pickle.load(f))
    f.close()
    lock = threading.Lock()
    model = train_finetune_model(model,models_savename,train,valid,train_filehandle,batch_size,class_order,lock)
    print("## Training is complete")

    ## Step 5:
    ## Evaluate Model
    print("## Evaluating Model on test set")
    test_generator = BSONIterator(train_filehandle, test, class_order, None, lock, batch_size=batch_size)
    metric_values = model.evaluate_generator(test_generator,steps=np.ceil(float(len(test)) / batch_size))
    print("## Test Loss = {:f} , Test Accuracy = {:f}".format(metric_values[0],metric_values[1]))
    print("## Done evaluating Model on test set")

    train_filehandle.close()
