# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from tqdm import tqdm 
import tensorflow as tf 
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from google.cloud import storage 
from io import BytesIO
import time
import cv2
start = time.time()


model = ResNet50(weights='imagenet', pooling=max, include_top = False) 

####### GENERATING FEATURES
featues = open('train_featues.txt', 'w+') #to save features
train_dir = '.../train'
train_files = os.listdir(train_dir)

 
start = time.time()

for file in  tqdm(train_files):
    try:
        f = v2.imread(os.path.join(train_dir, file))
        img = image.load_img(f, target_size=(224, 224)) 
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x) 
        features = model.predict(x) 
        features_reduce = features.squeeze() 
        train_featues.write(' '.join(str(x) for x in features.squeeze()) + '\n')
    except: # to skip not files in a wrong format
         pass

print(i)
end = time.time()
print('\n\ntime spend: ' , (end - start)/60 , ' minutes \n\n')

train_filenames.close()
