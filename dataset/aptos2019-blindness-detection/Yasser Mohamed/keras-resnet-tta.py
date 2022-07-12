import numpy as np
import pandas as pd
import os
import cv2
import PIL
import gc
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
from tqdm import tqdm
from math import ceil
import math
import sys
import gc

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from keras.activations import softmax
from keras.activations import elu
from keras.activations import relu
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm






SEED = 7
np.random.seed(SEED)
set_random_seed(SEED)
dir_path = "../input/aptos2019-blindness-detection/"
IMG_DIM = 224  # 224 399 #
BATCH_SIZE = 12
CHANNEL_SIZE = 3
NUM_EPOCHS = 60
TRAIN_DIR = 'train_images'
TEST_DIR = 'test_images'
FREEZE_LAYERS = 2  # freeze the first this many layers for training
CLASSS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


df_train = pd.read_csv(os.path.join(dir_path, "train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, "test.csv"))
NUM_CLASSES = df_train['diagnosis'].nunique()






x_train, x_test, y_train, y_test = train_test_split(df_train.id_code, df_train.diagnosis, test_size=0.2,
                                                    random_state=SEED, stratify=df_train.diagnosis)


def draw_img(imgs, target_dir, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = os.path.join(dir_path, f"{target_dir}/{row['id_code']}.png")
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axis[row, col].imshow(img)
    plt.suptitle(class_label)
    plt.show()


CLASS_ID = 0
#draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])

# %% [code]
CLASS_ID = 1
#draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])

# %% [code]
CLASS_ID = 2
#draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])

# %% [code]
CLASS_ID = 3
#draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])

# %% [code]
CLASS_ID = 4
#draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


CLASS_ID = 'Test DataSet'
#draw_img(df_test.sample(12, random_state=SEED), 'test_images', CLASS_ID)


def check_max_min_img_height_width(df, img_dir):
    max_Height , max_Width =0 ,0
    min_Height , min_Width =sys.maxsize ,sys.maxsize 
    for idx, row in df.iterrows():
        imgPath=os.path.join(dir_path,f"{img_dir}/{row['id_code']}.png") 
        img=cv2.imread(imgPath)
        H,W=img.shape[:2]
        max_Height=max(H,max_Height)
        max_Width =max(W,max_Width)
        min_Height=min(H,min_Height)
        min_Width =min(W,min_Width)
    return max_Height, max_Width, min_Height, min_Width

# %% [code]
#check_max_min_img_height_width(df_train, TRAIN_DIR)

# %% [code]
#check_max_min_img_height_width(df_test, TEST_DIR)



# %% [code]
# Display some random images from Data Set with class categories ing gray
figure = plt.figure(figsize=(20, 16))
for target_class in (y_train.unique()):
    for i, (idx, row) in enumerate(
            df_train.loc[df_train.diagnosis == target_class].sample(5, random_state=SEED).iterrows()):
        ax = figure.add_subplot(5, 5, target_class * 5 + i + 1)
        imagefile = f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
        img = cv2.imread(imagefile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_DIM, IMG_DIM))
        plt.imshow(img, cmap='gray')
        ax.set_title(CLASSS[target_class])




def draw_img_light(imgs, target_dir, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = os.path.join(dir_path, f"{target_dir}/{row['id_code']}.png")
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_DIM, IMG_DIM))
        img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , IMG_DIM/10) ,-4 ,128) # the trick is to add this line
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axis[row, col].imshow(img, cmap='gray')
    plt.suptitle(class_label)
    plt.show()

# %% [code]
#CLASS_ID = 3
#draw_img_light(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])

# %% [code]
# Image Croping
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]


# %% [code]
def crop_image_from_gray(img,tol=7):
    if img.ndim== 2:
        mask=img>tol
    elif img.ndim==3:
        gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        mask=gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if check_shape ==0: # Image was full dark and may be cropout everything.
            return img # Return original Image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            print(img1.shape,img2.shape,img3.shape)            
            img=np.stack([img1,img2,img3],axis=1)
            print(img.shape)
            return img

# %% [code]
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)S
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

# %% [code]


# %% [code]
def crop_image(img,tol=7):
    w, h = img.shape[1],img.shape[0]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.blur(gray_img,(5,5))
    shape = gray_img.shape 
    gray_img = gray_img.reshape(-1,1)
    quant = quantile_transform(gray_img, n_quantiles=256, random_state=0, copy=True)
    quant = (quant*256).astype(int)
    gray_img = quant.reshape(shape)
    xp = (gray_img.mean(axis=0)>tol)
    yp = (gray_img.mean(axis=1)>tol)
    x1, x2 = np.argmax(xp), w-np.argmax(np.flip(xp))
    y1, y2 = np.argmax(yp), h-np.argmax(np.flip(yp))
    if x1 >= x2 or y1 >= y2 : # something wrong with the crop
        return img # return original image
    else:
        img1=img[y1:y2,x1:x2,0]
        img2=img[y1:y2,x1:x2,1]
        img3=img[y1:y2,x1:x2,2]
        img = np.stack([img1,img2,img3],axis=-1)
    return img

def process_image(image, size=512):
    image = cv2.resize(image, (size,int(size*image.shape[0]/image.shape[1])))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        image = crop_image(image, tol=15)
    except Exception as e:
        image = image
        print( str(e) )
    return image






# %% [markdown]
# <a id="8"></a>
# # Data Pre-Processing
# 
#  #### Croping Images
# 
# https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv

# %% [code]
imgPath = f"../input/aptos2019-blindness-detection/train_images/cd54d022e37d.png"
img = cv2.imread(imgPath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt)
img = img[y:y + h, x:x + w]
#plt.imshow(img)


# %% [markdown]
# - Croping Images randomly for resizing.

# %% [code]
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    img = img[y:(y + dy), x:(x + dx), :]
    return img


"""Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """


def crop_generator(batches, crop_length):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[0] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

            

# %% [markdown]
# - Adding image type with image in dataframe

# %% [code]
# print("available RAM:", psutil.virtual_memory())
gc.collect()
# print("available RAM:", psutil.virtual_memory())

df_train.id_code = df_train.id_code.apply(lambda x: x + ".png")
df_test.id_code = df_test.id_code.apply(lambda x: x + ".png")
df_train['diagnosis'] = df_train['diagnosis'].astype('str')



datagenerator=ImageDataGenerator(#rescale=1./255,
#                                       validation_split=0.15, 
                                         horizontal_flip=True,
                                         vertical_flip=True, 
                                         rotation_range=40, 
                                         zoom_range=0.2, 
                                         shear_range=0.1,
                                        fill_mode='nearest')



# %% [code]
train_datagen = ImageDataGenerator(rescale=1. / 255, 
                                         validation_split=0.15, 
                                         horizontal_flip=True,
                                         vertical_flip=True, 
                                         rotation_range=40, 
                                         zoom_range=0.2, 
                                         shear_range=0.1,
                                        fill_mode='nearest')
# valid_datagen=image.ImageDataGenerator(rescale=1./255)

# %% [code]
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory="../input/aptos2019-blindness-detection/train_images/",
                                                    x_col="id_code",
                                                    y_col="diagnosis",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    subset='training',
                                                    shaffle=True,
                                                    seed=SEED,
                                                    )
valid_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory="../input/aptos2019-blindness-detection/train_images/",
                                                    x_col="id_code",
                                                    y_col="diagnosis",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    subset='validation',
                                                    shaffle=True,
                                                    seed=SEED
                                                    )
del x_train
# # del x_test
del y_train
# del y_test
gc.collect()
#  color_mode= "grayscale",


# %% [markdown]
# <a id="8"></a>
# # Model Architecture Design

# %% [code]
def design_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(2, 2), input_shape=[IMG_DIM, IMG_DIM, CHANNEL_SIZE], activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1000, activation=relu))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1000, activation=relu))
    model.add(Dropout(rate=0.2))
    model.add(Dense(5, activation='softmax'))
    return model


#model = design_model()
# model.summary()


# %% [markdown]
# ### Compile model

# %% [code]
#model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# %% [markdown]
# <a id="9"></a>
# # Keras Callback Funcations
# - Call Back functions Eraly Stoping and Learning Rate Reducing

# %% [code]
eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
# Reducing the Learning Rate if result is not improving. 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',
                              verbose=1)

# %% [code]
NUB_TRAIN_STEPS = train_generator.n // train_generator.batch_size
NUB_VALID_STEPS = valid_generator.n // valid_generator.batch_size

NUB_TRAIN_STEPS, NUB_VALID_STEPS

# %% [code]
# model.fit_generator(generator=train_generator,
#                     validation_data=valid_generator,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_steps=STEP_SIZE_TRAIN,
#                     verbose=1,
#                     callbacks=[checkpoint],
#                     use_multiprocessing=True,
#                     workers=3,
#                     shuffle=True,
#                     max_queue_size=16,
#                     epochs=NB_EPOCHS)


# %% [markdown]
# <a id="10"></a>
# # Transfer Learning 

# %% [code]
def create_resnet(img_dim, CHANNEL, n_class):
    input_tensor = Input(shape=(img_dim, img_dim, CHANNEL))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)
    base_model.load_weights('../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(n_class, activation='softmax', name="Output_Layer")(x)
    model_resnet = Model(input_tensor, output_layer)

    return model_resnet


model_resnet = create_resnet(IMG_DIM, CHANNEL_SIZE, NUM_CLASSES)


# # Layers 
# for i, lay in enumerate(model_resnet.layers):
#     print(i,lay.name)
# Training All Layers

for layers in model_resnet.layers:
    layers.trainable = True



lr = 1e-3
optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr=lr, decay=0.01) 
model_resnet.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# model.summary()
gc.collect()




history = model_resnet.fit_generator(generator=train_generator,
                                     steps_per_epoch=NUB_TRAIN_STEPS,
                                     validation_data=valid_generator,
                                     validation_steps=NUB_VALID_STEPS,
                                     epochs=NUM_EPOCHS,
                                     #                            shuffle=True,  
                                     callbacks=[eraly_stop, reduce_lr],
                                     verbose=2)
gc.collect()



history.history.keys()


accu = history.history['acc']
val_acc = history.history['val_acc']

model_resnet.save("model")

def prepare(filepath):
    img_size = 224
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)

'''
plt.plot(accu, label="Accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Acc', 'val_acc'])
plt.plot(np.argmax(history.history["val_acc"]), np.max(history.history["val_acc"]), marker="x", color="r",
         label="best model")
plt.show()




plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();



# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
(eval_loss, eval_accuracy) = tqdm(
    model_resnet.evaluate_generator(generator=valid_generator, steps=NUB_VALID_STEPS, pickle_safe=False))
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

'''


