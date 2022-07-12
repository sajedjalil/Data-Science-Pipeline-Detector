import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from PIL import Image
import os

os.chdir(r'../input/')
train=pd.read_csv(r'train.csv')
train.set_index('id',drop=True,inplace=True)
depth=pd.read_csv(r'depths.csv')
depth.set_index('id',drop=True,inplace=True)

id_train=pd.Series(os.listdir(r'train/images')).apply(lambda x:str(x)[0:-4]).values
id_test=pd.Series(os.listdir(r'test/images')).apply(lambda x:str(x)[0:-4]).values

image_train=[]
image_test=[]
mask_train=[]
depth_train=[]
depth_test=[]

for ix,id in enumerate(id_train):
    image_train.append(np.array(Image.open(r'train/images/'+id+'.png').convert('L'),dtype=np.float32))
    depth_train.append(depth.loc[id]['z'])
    mask_train.append(np.array(Image.open(r'train/masks/'+id+'.png').convert('L'),dtype=np.float32))

for ix,id in enumerate(id_test):
    image_test.append(np.array(Image.open(r'test/images/'+id+'.png').convert('L'),dtype=np.float32))
    depth_test.append(depth.loc[id,'z'])

image_test=np.array(image_test,copy=False).reshape((-1,101,101,1))/255
image_train=np.array(image_train,copy=False).reshape((-1,101,101,1))/255
mask_train=np.array(mask_train,copy=False).reshape((-1,101,101,1))/255


depth_train=np.array(depth_train)
depth_test=np.array(depth_test)
depth_train=(depth_train-depth['z'].mean())/depth['z'].std()
depth_test=(depth_test-depth['z'].mean())/depth['z'].std()

image_train,image_validate,mask_train,mask_validate,depth_train,depth_validate=train_test_split(image_train,mask_train,depth_train,test_size=0.05)

image_height=image_train.shape[1]
image_width=image_train.shape[2]

input_image=Input(shape=(image_height,image_width,1))
input_depth=Input(shape=(1,))
conv1_down=Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu')(input_image)
pool2_down=MaxPooling2D(pool_size=(2, 2))(conv1_down)
pool2_down_drop=SpatialDropout2D(rate=0.25)(pool2_down)
conv3_down=Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(pool2_down_drop)
pool4_down=MaxPooling2D(pool_size=(2, 2))(conv3_down)
drop_down_4=SpatialDropout2D(rate=0.25)(pool4_down)
conv5_down=Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu')(drop_down_4)
depth_vec=RepeatVector(50*50)(input_depth)
depth_channel=Reshape((50,50,1))(depth_vec)
unpool6_up=Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5_down),conv3_down,depth_channel])
unpool6_up_drop=SpatialDropout2D(rate=0.25)(unpool6_up)
conv7_up=Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(unpool6_up_drop)
#unpool8_up=Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7_up),conv1_down])
unpool8_up=Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,padding='valid')(conv7_up)
unpool8_up_cat=Concatenate(axis=3)([unpool8_up,conv1_down])
unpool8_up_cat_drop=SpatialDropout2D(rate=0.25)(unpool8_up_cat)
conv9_up=Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu')(unpool8_up_cat_drop)
conv10_up=Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid')(conv9_up)

unet=Model(inputs=[input_image,input_depth],outputs=[conv10_up])
unet.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
unet.fit(x=[image_train,depth_train],y=mask_train,epochs=2,batch_size=32,validation_data=([image_validate,depth_validate],mask_validate))
os.chdir(r'../working/')
unet.save('model.h5')
print((unet.predict(x=[image_test[:100],depth_test[:100]])>0.5).sum())
print('done')
#mask_test=unet.predict(x=[image_test,depth_test])



# def RLenc(img, order='F', format=True):
#     bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
#     runs = []  ## list of run lengths
#     r = 0  ## the current run length
#     pos = 1  ## count starts from 1 per WK
#     for c in bytes:
#         if (c == 0):
#             if r != 0:
#                 runs.append((pos, r))
#                 pos += r
#                 r = 0
#             pos += 1
#         else:
#             r += 1
#     if r != 0:
#         runs.append((pos, r))
#         pos += r
#         r = 0
#     if format:
#         z = ''
#         for rr in runs:
#             z += '{} {} '.format(rr[0], rr[1])
#         return z[:-1]
#     else:
#         return runs


#submission=pd.DataFrame({'id':id_test,'rle_mask':[RLenc(x) for x in (mask_test>0.5).astype(np.uint8)]})
#os.chdir(r'../working/')
#submission.to_csv('mysubmission.csv',index=False,index_label=False)
#print('done')
