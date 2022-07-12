########################################################################################################################
## Author: Amro Tork amtc2018@gmail.com
########################################################################################################################

from __future__ import print_function

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing import image
from keras import backend as K

import io
import bson

########################################################################################################################
## Global Variables
########################################################################################################################
batch_size = 40
num_classes = 10
epochs = 12
im_size = 180
num_dicts = 7069896 # according to data page
max_num_images = 60000 # number of images to extract.
valid_ratio = 0.1
test_ratio = 0.1
#train_ratio = 0.001
TARGET_SZ = (im_size,im_size)

if K.image_data_format() == 'channels_first':
    input_shape = (3, im_size, im_size)
else:
    input_shape = (im_size, im_size, 3)
    
########################################################################################################################
## Procedures
########################################################################################################################
def get_image_array(fimg,target_size=TARGET_SZ):
    img = image.load_img(fimg, target_size=target_size)
    x = image.img_to_array(img)
    rot_image = random.choice([True, False])
    flip_image = random.choice([True, False])

    if rot_image:
        x = np.rot90(x)

    if flip_image:
        flip_axis = random.choice([0, 1])
        x = np.flip(x, axis=flip_axis)
    x = np.expand_dims(x, axis=0)
    return x

def create_model(num_classes,input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def extract_categories_df(num_images):
    img_category = list()
    item_locs_list = list()
    items_len_list = list()
    pic_ind_list = list()

    with open('../input/train.bson', 'rb') as f:
        data = bson.decode_file_iter(f)
        last_item_loc = 0
        item_len = 0
        for c, d in enumerate(data):
            loc = f.tell()
            item_len = loc - last_item_loc
            category_id = d['category_id']

            for e, pic in enumerate(d['imgs']):
                
                img_category.append(category_id)
                item_locs_list.append(last_item_loc)
                items_len_list.append(item_len)
                pic_ind_list.append(e)
                
                if num_images is not None:
                    if len(img_category) >= num_images:
                        break
            
            last_item_loc = loc
            
            if num_images is not None:
                if len(img_category) >= num_images:
                    break
    
    f.close()
    df_dict = {
        'category': img_category,
        "img_id": range(len(img_category)),
        "item_loc": item_locs_list,
        "item_len": items_len_list,
        "pic_ind": pic_ind_list
    }
    df = pd.DataFrame(df_dict)
    df.to_csv("all_images_categories.csv", index=False)
        
    return df


def get_image(image_id, data_df, fh):
    img_info = data_df[data_df["img_id"] == image_id]
    item_loc = img_info["item_loc"].values[0]
    item_len = img_info["item_len"].values[0]
    pic_ind = img_info["pic_ind"].values[0]
    fh.seek(item_loc)
    item_data = fh.read(item_len)
    d = bson.BSON.decode(item_data)

    picture = get_image_array(io.BytesIO(d["imgs"][pic_ind]['picture']))
    return picture


def data_generator(df,fh,lb,batch_size,im_size,n_classes):
    while True:
        a = np.empty((batch_size, im_size, im_size, 3))
        y = np.empty((batch_size, n_classes))
        b_ind = 0
    
        for index, row in df.iterrows():
            img_id = row["img_id"]
            pic = get_image(img_id,df,fh)
            pic = preprocess_input(pic)
            y_pic = lb.transform([row["category"]])
            a[b_ind] = pic
            y[b_ind] = y_pic[0]
            b_ind +=1
            if b_ind >= batch_size:
                yield a, y
                a = np.empty((batch_size, im_size, im_size, 3))
                y = np.empty((batch_size, n_classes))
                b_ind = 0
        if b_ind > 0:
            yield a[:b_ind,:,:,:], y[:b_ind,:]

########################################################################################################################
## Start Processing
########################################################################################################################
## Step 1:
## Extract all images classes for the some samples
print("## Extract all categories from bson file.")
cat_df = extract_categories_df(max_num_images)
num_classes = len(pd.unique(cat_df["category"]))
num_imgs = len(cat_df)
print("## Done extraction of the categories. Number of categories found = {:d}".format(num_classes))

## Step 2:
## Prepare data generators
print("## Prepare Data generators.")
train_ratio = 1.0 - test_ratio - valid_ratio
cat_df = cat_df.sample(frac=1).reset_index(drop=True)

## Get categories that has less than 3 images and drop them
val_count = cat_df["category"].value_counts()
drop_cats = val_count[val_count < 3].index
print("## We will drop those categories as there is less than 3 images: {}".format(str(drop_cats)))
cat_df = cat_df[~cat_df.category.isin(drop_cats)]
num_classes = len(pd.unique(cat_df["category"]))
num_imgs = len(cat_df)
cat_df.to_csv("all_images_categories.csv", index=False)
print("## Final number of categories found = {:d}".format(num_classes))
print("## Final total number of samples after dropping categories with few images = {:d}".format(num_imgs))

## Split the training dataset
train_valid, test = train_test_split(cat_df, test_size=test_ratio,stratify=cat_df["category"])
train, valid = train_test_split(train_valid, test_size=valid_ratio,stratify=train_valid["category"])

## Sorting to help in random access
train.sort_values(by="img_id",inplace=True)
valid.sort_values(by="img_id",inplace=True)
test.sort_values(by="img_id",inplace=True)

## Saving to files to avoid repeatation.
train.reset_index(inplace=True,drop=True)
valid.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)

train.to_csv("train.csv", index=False)
valid.to_csv("valid.csv", index=False)
test.to_csv("test.csv",index=False)

print("## Number of train classes = {:d}".format(len(pd.unique(train["category"]))))
print("## Number of valid classes = {:d}".format(len(pd.unique(valid["category"]))))
print("## Number of test classes = {:d}".format(len(pd.unique(test["category"]))))
print("## Total number of training samples = {:d}".format(len(train)))
print("## Total number of validation samples = {:d}".format(len(valid)))
print("## Total number of test samples = {:d}".format(len(test)))
print("## Done preparing data genrators")

## Create Label binarizer
print("## Creating Label binarizer")
lb = LabelBinarizer()
lb.fit(cat_df["category"])
print("## Label binarizer created.")

## Step 3:
## Create keras model
print("## Creating CNN model.")
model = create_model(num_classes,input_shape)
print("## Done creating model.")

## Step 4:
## Train Model
train_filehandle = open('../input/train.bson', 'rb')
model.fit_generator(data_generator(train,train_filehandle,lb,batch_size,im_size,num_classes),
                    steps_per_epoch=5,
                    epochs=7,
                    verbose=1,
                    validation_data=data_generator(valid,train_filehandle,lb,batch_size,im_size,num_classes),
                    validation_steps=5)

## Step 5:
## Evaluate Model
metric_values = model.evaluate_generator(data_generator(test,train_filehandle,lb,batch_size,im_size,num_classes),
                         steps=10)
print("## Test results = {}".format(str(metric_values)))
train_filehandle.close()




