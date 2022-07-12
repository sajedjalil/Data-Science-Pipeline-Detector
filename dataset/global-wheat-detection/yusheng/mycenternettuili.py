# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# /kaggle/input/final-weights100/final_weights100.h5
# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

        

# coding: utf-8

# # Centernet Keras Inference
# In this notebook I am going to make the inference sample for a simple centernet based model.
# 
# The training code can be found in [this notebook](https://www.kaggle.com/nvnnghia/keras-centernet-training)
# 
# The post processing parta are borrowed from [this notebook](https://www.kaggle.com/kmat2019/centernet-keypoint-detector)

# In[1]:


import cv2, os
import numpy as np 
from keras.utils import Sequence
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from PIL import Image
from keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from keras.models import Model

import pandas as pd

from matplotlib import pyplot as plt

# import tensorflow as tf
# from keras import backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True  #设置gpu显存 不然不够
# sess = tf.Session(config=config)
# K.set_session(sess)
# # Define the network

# In[2]:


###
category_n=1
output_layer_n=category_n+4

##########MODEL#############

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = BatchNormalization()(x_deep)   
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x=Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)   
    x = LeakyReLU(alpha=0.1)(x)
    return x
  


def cbr(x, out_layer, kernel, stride):
    x=Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def resblock(x_in,layer_n):
    x=cbr(x_in,layer_n,3,1)
    x=cbr(x,layer_n,3,1)
    x=Add()([x,x_in])
    return x  


#I use the same network at CenterNet
def create_model(input_shape, aggregation=True):
    input_layer = Input(input_shape)
    
    #resized input
    input_layer_1=AveragePooling2D(2)(input_layer)
    input_layer_2=AveragePooling2D(2)(input_layer_1)

    #### ENCODER ####

    x_0= cbr(input_layer, 16, 3, 2)#512->256
    concat_1 = Concatenate()([x_0, input_layer_1])

    x_1= cbr(concat_1, 32, 3, 2)#256->128
    concat_2 = Concatenate()([x_1, input_layer_2])

    x_2= cbr(concat_2, 64, 3, 2)#128->64
    
    x=cbr(x_2,64,3,1)
    x=resblock(x,64)
    x=resblock(x,64)
    
    x_3= cbr(x, 128, 3, 2)#64->32
    x= cbr(x_3, 128, 3, 1)
    x=resblock(x,128)
    x=resblock(x,128)
    x=resblock(x,128)
    
    x_4= cbr(x, 256, 3, 2)#32->16
    x= cbr(x_4, 256, 3, 1)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
 
    x_5= cbr(x, 512, 3, 2)#16->8
    x= cbr(x_5, 512, 3, 1)
    
    x=resblock(x,512)
    x=resblock(x,512)
    x=resblock(x,512)
    
    #### DECODER ####
    x_1= cbr(x_1, output_layer_n, 1, 1)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_2= cbr(x_2, output_layer_n, 1, 1)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3= cbr(x_3, output_layer_n, 1, 1)
    x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n) 
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

    x_4= cbr(x_4, output_layer_n, 1, 1)

    x=cbr(x, output_layer_n, 1, 1)
    x= UpSampling2D(size=(2, 2))(x)#8->16 

    x = Concatenate()([x, x_4])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#16->32

    x = Concatenate()([x, x_3])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#32->64 

    x = Concatenate()([x, x_2])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#64->128 

    x = Concatenate()([x, x_1])
    x=Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
    out = Activation("sigmoid")(x)
    
    model=Model(input_layer, out)
    
    return model


# # Function for post processing

# In[3]:


def NMS_all(predicts,category_n, pred_out_h, pred_out_w, score_thresh,iou_thresh):
    y_c=predicts[...,category_n]+np.arange(pred_out_h).reshape(-1,1)
    x_c=predicts[...,category_n+1]+np.arange(pred_out_w).reshape(1,-1)
    height=predicts[...,category_n+2]*pred_out_h
    width=predicts[...,category_n+3]*pred_out_w

    count=0
    for category in range(category_n):
        predict=predicts[...,category]
        mask=(predict>score_thresh)
        #print("box_num",np.sum(mask))
        if mask.all==False:
            continue
        box_and_score=NMS(predict[mask],y_c[mask],x_c[mask],height[mask],width[mask],iou_thresh,pred_out_h, pred_out_w)
        box_and_score=np.insert(box_and_score,0,category,axis=1)#category,score,top,left,bottom,right
        if count==0:
            box_and_score_all=box_and_score
        else:
            box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
        count+=1
    score_sort=np.argsort(box_and_score_all[:,1])[::-1]
    box_and_score_all=box_and_score_all[score_sort]
    #print(box_and_score_all)

    _,unique_idx=np.unique(box_and_score_all[:,2],return_index=True)
    #print(unique_idx)
    return box_and_score_all[sorted(unique_idx)]
  
def NMS(score,y_c,x_c,height,width,iou_thresh,pred_out_h, pred_out_w,merge_mode=False):
    if merge_mode:
        score=score
        top=y_c
        left=x_c
        bottom=height
        right=width
    else:
        #flatten
        score=score.reshape(-1)
        y_c=y_c.reshape(-1)
        x_c=x_c.reshape(-1)
        height=height.reshape(-1)
        width=width.reshape(-1)
        size=height*width


        top=y_c-height/2
        left=x_c-width/2
        bottom=y_c+height/2
        right=x_c+width/2

        inside_pic=(top>0)*(left>0)*(bottom<pred_out_h)*(right<pred_out_w)
        outside_pic=len(inside_pic)-np.sum(inside_pic)
        #if outside_pic>0:
        #  print("{} boxes are out of picture".format(outside_pic))
        normal_size=(size<(np.mean(size)*20))*(size>(np.mean(size)/20))
        score=score[inside_pic*normal_size]
        top=top[inside_pic*normal_size]
        left=left[inside_pic*normal_size]
        bottom=bottom[inside_pic*normal_size]
        right=right[inside_pic*normal_size]
  

    

  #sort  
    score_sort=np.argsort(score)[::-1]
    score=score[score_sort]  
    top=top[score_sort]
    left=left[score_sort]
    bottom=bottom[score_sort]
    right=right[score_sort]

    area=((bottom-top)*(right-left))

    boxes=np.concatenate((score.reshape(-1,1),top.reshape(-1,1),left.reshape(-1,1),bottom.reshape(-1,1),right.reshape(-1,1)),axis=1)

    box_idx=np.arange(len(top))
    alive_box=[]
    while len(box_idx)>0:
  
        alive_box.append(box_idx[0])

        y1=np.maximum(top[0],top)
        x1=np.maximum(left[0],left)
        y2=np.minimum(bottom[0],bottom)
        x2=np.minimum(right[0],right)

        cross_h=np.maximum(0,y2-y1)
        cross_w=np.maximum(0,x2-x1)
        still_alive=(((cross_h*cross_w)/area[0])<iou_thresh)
        if np.sum(still_alive)==len(box_idx):
            print("error")
            print(np.max((cross_h*cross_w)),area[0])
        top=top[still_alive]
        left=left[still_alive]
        bottom=bottom[still_alive]
        right=right[still_alive]
        area=area[still_alive]
        box_idx=box_idx[still_alive]
    return boxes[alive_box]#score,top,left,bottom,right

def visualize(box_and_score,img):
    boxes = []
    scores = []
    colors= [(0,0,255), (255,0,0), (0,255,255), (0,127,127), (127,255,127), (255,255,0)]
    classes = ["car", "motor", "person", "bus", "truck", "bike"]
    number_of_rect=np.minimum(500,len(box_and_score))

    for i in reversed(list(range(number_of_rect))):
        predicted_class, score, top, left, bottom, right = box_and_score[i,:]


        top = np.floor(top + 0.5).astype('int32')
        left = np.floor(left + 0.5).astype('int32')
        bottom = np.floor(bottom + 0.5).astype('int32')
        right = np.floor(right + 0.5).astype('int32')

        predicted_class = int(predicted_class)

        label = '{:.2f}'.format(score)
        #print(label)
        #print(top, left, right, bottom)
        cv2.rectangle(img, (left, top), (right, bottom), colors[predicted_class], 3)
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX ,  
                       0.5, (255,255,255), 2, cv2.LINE_AA) 
        boxes.append([left, top, right-left, bottom-top])
        scores.append(score)
    
    return np.array(boxes), np.array(scores)


# Convert to submission format
# Borrowed from [here](https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train)

# In[4]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


# In[5]:


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

# DIR_TRAIN = f'../train'
# DIR_TEST = f'../test'


# DIR_WEIGHTS = '/kaggle/input/kerascenter'

# WEIGHTS_FILE = f'./atemp254-1.556.hdf5'
WEIGHTS_FILE = f'/kaggle/input/final-weights100/final_weights100.h5'

print(os.path.exists(WEIGHTS_FILE))
imagenames = os.listdir(DIR_TEST)

#def predict_image(imagenames,input_size=320, weights_file=''):
input_size = 768

pred_out_h=int(input_size/4)
pred_out_w=int(input_size/4)

model=create_model(input_shape=(input_size,input_size,3))

model.load_weights(WEIGHTS_FILE,by_name=True, skip_mismatch=False)


# In[6]:


imagenames


# # Inference

# In[7]:


results = []
fig, axes = plt.subplots(10, 1,figsize=(320,160))
for count, name in enumerate(imagenames):
    ids = name.split('.')[0] 
    imagepath = '%s/%s.jpg'%(DIR_TEST,ids)
    imgcv = cv2.imread(imagepath)
    img = cv2.resize(imgcv, (input_size, input_size))
    predict0 = model.predict((img[np.newaxis])/255).reshape(pred_out_h,pred_out_w,(category_n+4))
    #print(img.shape)
    print_h, print_w = imgcv.shape[:2]
    #print(predict.shape)
    '''img90 =np.rot90(img)
    predict90 = model.predict((img90[np.newaxis])/255).reshape(pred_out_h,pred_out_w,(category_n+4))
    predict90 = np.rot90(predict90, 3)
    
    img180 = np.rot90(img, 2)
    predict180 = model.predict((img180[np.newaxis])/255).reshape(pred_out_h,pred_out_w,(category_n+4))
    predict180 = np.rot90(predict180, 2)
    
    img270 = np.rot90(img, 3)
    predict270 = model.predict((img270[np.newaxis])/255).reshape(pred_out_h,pred_out_w,(category_n+4))
    predict270 = np.rot90(predict270)
    
    predict1 = np.add(predict0, predict90)/2
    predict2 = np.add(predict180, predict270)/2
    predict = np.add(predict1, predict2)/2'''
    
    box_and_score=NMS_all(predict0,category_n, pred_out_h, pred_out_w, score_thresh=0.25,iou_thresh=0.5)
    if len(box_and_score)==0:
        print('no boxes found!!')
        #return
        result = {
                'image_id': ids,
                'PredictionString': ''
            }

        results.append(result)
    else:

        #heatmap=predict[:,:,2]

        box_and_score=box_and_score*[1,1,print_h/pred_out_h,print_w/pred_out_w,print_h/pred_out_h,print_w/pred_out_w]
        # img=draw_rectangle(box_and_score[:,2:],img,"red")
        # img=draw_rectangle(true_boxes,img,"blue")
        preds, scores = visualize(box_and_score,imgcv)

        result = {
                'image_id': ids,
                'PredictionString': format_prediction_string(preds, scores)
            }

        results.append(result)
    

    
    # #axes[0].set_axis_off()
    if count <10:
        axes[count].imshow(imgcv)
    # #axes[1].set_axis_off()
    # axes[1].imshow(heatmap)#, cmap='gray')
    # #axes[2].set_axis_off()
    # #axes[2].imshow(heatmap_1)#, cmap='gray')
# plt.show()
    #break


# In[8]:


print(results[:5])

# In[9]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()


# In[10]:


test_df.to_csv('submission.csv', index=False)


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session