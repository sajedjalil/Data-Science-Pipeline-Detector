

"""
The main goal of this model is to address my unease with shrinking images to a 
uniform size. It seems clear to me that at the very least one would want to 
post-process any mask that is upsampled with a non-adaptive method. Rather than 
figure that nightmare out, I decided to put it all together in a single model. 

No images are downsampled in pre-processing, but they are padded up to the 
next multiple of 32. This extra padding is simply removed when submitting the csv.

The model is input a full sized image. It is downsampled several times, merged,
and upsampled. The full sized image is concatenated onto the feature map, and 
some lighter layers are used to fine tune the upsampling. 

There are about a thousand ways to break this model, and you can't use like
half of the common cnn tools you could with uniform sized images, but have fun!
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.morphology import label

from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda, Activation
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Nadam, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Basic Imports, Determinacy, Directory Info
warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib_plugin')
seed = 139
random.seed = seed
np.random.seed = seed

data_dir = '../input/'
train_path = data_dir+'train/'
test_path = data_dir+'test/'

##==============Imports Training and Testing Data
def import_images_masks(): 
    train_ids = next(os.walk(train_path))[1]
    test_ids = next(os.walk(test_path))[1]  
    images, masks, test_images = [], [], []
    
    for ID in train_ids:
        print(ID)
        image = plt.imread(train_path+ID+"/images/"+ID+".png")
        image = np.expand_dims(image,axis=0)
        images.append(image[:,:,:,:3]) #for some reason, the fourth channel is solid ones?

        image_shape = np.shape(image)
        cmask = np.zeros((1,image_shape[1], image_shape[2],1), dtype=np.bool) #create blank mask
        for mask_file in next(os.walk(train_path+ID+'/masks'))[2]:#itterate over indivisual masks
            mask = plt.imread(train_path+ID+"/masks/"+mask_file)
            mask = np.expand_dims(mask,axis=0)
            mask = np.expand_dims(mask,axis=-1)
            cmask = np.maximum(cmask, mask)
        masks.append(cmask)
    
    for ID in test_ids:#load in test images
        print(ID)
        image = plt.imread(test_path+ID+"/images/"+ID+".png")
        image = np.expand_dims(image,axis=0)
        test_images.append(image[:,:,:,:3]) #for some reason, the fourth channel is solid 1s?
    return images, masks, test_images

##==============Pad images to the next multiple of n
def pad_images(xdata,ydata,test,n): 
    x_padded,y_padded,t_padded = [], [], []
    for i in range(len(xdata)):
        img, mask = xdata[i], ydata[i]
        h, w = np.shape(img)[1], np.shape(img)[2]
        pimg = np.zeros((1,h+(n-h%n),w+(n-w%n),3))
        pmask = np.zeros((1,h+(n-h%n),w+(n-w%n),1))
        pimg[0,:h,:w,:] = img
        pmask[0,:h,:w,:] = mask
        x_padded.append(pimg)
        y_padded.append(pmask)
        
    for i in range(len(test)):
        img = test[i]
        w = np.shape(img)[2]
        h = np.shape(img)[1]
        pimg = np.zeros((1,h+(n-h%n),w+(n-w%n),3))
        pimg[0,:h,:w,:] = img
        t_padded.append(pimg)
    return x_padded,y_padded, t_padded

##==============Shuffle X and Y data in parallel
def shuffle_split(xdata,ydata,val_rate):
    shuffle_ix = np.arange(len(xdata))
    np.random.shuffle(shuffle_ix)
    xdata = [xdata[i] for i in shuffle_ix]
    ydata = [ydata[i] for i in shuffle_ix]
    ix = int(round(len(xdata)*(1-val_rate),0))
    x_train = xdata[:ix]
    x_val = xdata[ix:]
    y_train = ydata[:ix]
    y_val = ydata[ix:]
    return x_train, x_val, y_train, y_val    

##============ Submission Building Code
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def build_submission(pred_masks_bin):
    test_ids = next(os.walk(test_path))[1]
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(pred_masks_bin[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
        
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018-1.csv', index=False)
    
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x[0] > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

##============Visualize Image and Masks
def show_img_mask(images, masks, n):
    image = images[n][0,:,:,:]
    mask = masks[n][0,:,:,:]
    stack = [image, mask[:,:,0], image*mask, image*(1-mask)]
    fig = plt.figure(figsize=(15,15))
    for i in range(4):
        sub = fig.add_subplot(2,2,i+1)
        sub.imshow(stack[i], interpolation='nearest')
    
##==============Visualize Image, Mask, Preds, and Disagreement
def show_img_mask_pred(images,targets,preds,n):
    image = images[n][0,:,:,:]
    target = targets[n][0,:,:,:]
    mask = preds[n][0,:,:,:]
    difference = np.zeros_like(image, dtype='float32')
    difference[:,:,0:1] = target-mask
    difference[:,:,2:3] = mask-target
    difference[difference<0] = 0.
    stack = [image,         target[:,:,0],      mask[:,:,0],
             difference,    image*difference,   image*(1.-difference)]
    fig = plt.figure(figsize=(12,8))
    for i in range(6):
        sub = fig.add_subplot(2,3,i+1)
        sub.imshow(stack[i], interpolation='nearest')
    fig.tight_layout()

##==============IoU metric
##==============This is not actually the eval metric....
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def data_gen(x, y,shuffle,augment):
    while True:
        shuffle_ix = np.arange(len(x))
        if shuffle:np.random.shuffle(shuffle_ix)
        for i in shuffle_ix:
            image, mask = x[i], y[i]
            if augment:
                rotate = random.randint(0,3)
                transp = random.randint(0,1)
                image = np.rot90(image,rotate,axes=(1,2))
                mask  = np.rot90(mask,rotate,axes=(1,2))
                if transp:
                    image = np.swapaxes(image,1,2)#arbitrary sized version of transpose?
                    mask  = np.swapaxes(mask ,1,2)      
            #experimental concat median of channels
            medians = np.zeros_like(image)
            medians[:,:,:,0] = np.median(image[0,:,:,0])
            medians[:,:,:,1] = np.median(image[0,:,:,1])
            medians[:,:,:,2] = np.median(image[0,:,:,2])
            yield image, mask
            
def test_gen(x):
    while True:
        for image in x:yield image
        
           
##============Begin Neural Network Stuff
def build_model():
    inputs = Input((None, None, 3))
    image = inputs
    
    half_image = AveragePooling2D(2)(image) #downsampled
    for p in [1, 2, 4, 8, 16]:
        ds = AveragePooling2D(p)(half_image)
        #image is at pooled size
        a = SeparableConv2D(32, (3,3), dilation_rate=1, activation="selu", kernel_initializer="lecun_normal", padding="same")(ds)
        b = SeparableConv2D(32, (3,3), dilation_rate=2, activation="selu", kernel_initializer="lecun_normal", padding="same")(ds)
        c = SeparableConv2D(32, (3,3), dilation_rate=3, activation="selu", kernel_initializer="lecun_normal", padding="same")(ds)
        d = SeparableConv2D(32, (5,2), dilation_rate=3, activation="selu", kernel_initializer="lecun_normal", padding="same")(ds)
        e = SeparableConv2D(32, (2,5), dilation_rate=3, activation="selu", kernel_initializer="lecun_normal", padding="same")(ds)
        x = concatenate([ds,a,b,c,d,e])
        x = Conv2D(32,1, activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        x = Conv2D(16,1, activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        x = UpSampling2D(p)(x)
        #image is at half size
        x = SeparableConv2D(16, (3,3), dilation_rate=p+1, activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        half_image = concatenate([half_image, x])
        
    
    us = UpSampling2D(2)(half_image) #upsample features 2x2
    us = concatenate([image,us]) #concat the unscaled image to begin smoothing
    us = Conv2D(32, 1, activation="selu", kernel_initializer="lecun_normal",padding="same")(us)
    us = Conv2D(16, 1, activation="selu", kernel_initializer="lecun_normal",padding="same")(us)
    #smooth out the 2x upsampling with specially chosen combination of kernel sizes
    
    x3  = SeparableConv2D(16, 3, dilation_rate=3, activation="selu", kernel_initializer="lecun_normal",padding="same")(us)
    x31 = SeparableConv2D(16, (3,1), activation="selu", kernel_initializer="lecun_normal",padding="same")(us)
    x13 = SeparableConv2D(16, (1,3), activation="selu", kernel_initializer="lecun_normal",padding="same")(us)
    x = concatenate([x3, x31, x13]) #concat smoothed features to image
    x = Conv2D(16, 1, activation="selu", kernel_initializer="lecun_normal",padding="same")(x)

    mask = Conv2D(1, 1, activation="sigmoid")(x)  
    
    outputs = mask
    model = Model(inputs=[inputs], outputs=[outputs])
    
    opt = Nadam(lr=1e-4)
    model.compile(optimizer=opt,loss = 'binary_crossentropy',metrics = [mean_iou])
    model.summary()
    return model
      
def pred_test_data():
    pred_masks = []
    for i in range(len(img_test_padded)):
        pimage = img_test_padded[i]
        shape = np.shape(img_test[i])
        pred_padded_mask = model.predict(pimage, verbose=2)
        pred_mask = pred_padded_mask[:,:shape[1],:shape[2],:]
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
        pred_masks.append(pred_mask_bin)
    return pred_masks


images, masks, img_test = import_images_masks()
images, masks, img_test_padded = pad_images(images, masks, img_test, 32) #zero pad up to nearest multiple of 32
imgs_train, imgs_val, masks_train, masks_val = shuffle_split(images, masks, 0.1)

model = build_model()

earlystop = EarlyStopping(monitor="val_mean_iou", mode="max", patience=5, verbose=0)
checkpt = ModelCheckpoint('up_down_sampling.hdf5', verbose=1, save_best_only=True, monitor="val_mean_iou", mode="max")

fit_generator_args = {"generator":data_gen(imgs_train,masks_train,True,True), "steps_per_epoch":len(imgs_train),
                      "validation_data":data_gen(imgs_val,masks_val,False,False), "validation_steps":len(imgs_val),
                      "epochs":250,"max_queue_size":16, "callbacks":[earlystop,checkpt]}
model.fit_generator(**fit_generator_args)

model.load_weights("up_down_sampling.hdf5")
preds_val = []
for image in imgs_val:
    preds_val.append(model.predict(image,verbose=2))
    
for i in range(3):
    show_img_mask_pred(imgs_val,masks_val,preds_val,random.randint(0,len(preds_val)))
    

##============ Make and Build Submission on test data
masks_pred = pred_test_data()
for x in range(5):
    i = random.randint(0,len(masks_pred))
    show_img_mask(img_test,masks_pred,i)

build_submission(masks_pred)