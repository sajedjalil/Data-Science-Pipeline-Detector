# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import time
import cv2
from tqdm import tqdm 
import os
import pandas as pd
import pickle
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


            

import cv2
from tqdm import tqdm_notebook as tqdm
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

HEIGHT = 137
WIDTH = 236

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

path='/kaggle/input/bengaliai-cv19/'
t=time.time()
images=[]
SIZE=64
for i in range(4):
    print('Getting batch{}'.format(i))
    file=path+'test_image_data_'+str(i)+'.parquet'
    df=pd.read_parquet(file)
    df=df.iloc[:,1:].values
    # df=df.to_numpy()
    # labels=df[:,0]
    # df=df[:,1:].astype('float32')
    for j in tqdm(range(len(df))):
        image=df[j].reshape(137,236).astype(np.uint8)
        image = 255 - image
        image=crop_resize(image,size=SIZE)
        image=image.reshape((SIZE,SIZE,1)).astype('float16')/255
        images.append(image)
    del df  
images=np.asarray(images,dtype='float16')
#pickle.dump(np.asarray(images,dtype='float16'),open('test_data.h5','wb+'))
print(time.time()-t)

  
input_shape=(64,64,1)
num_classes=[168,11,7]
labels=['grapheme_root','vowel_diacritic','consonant_diacritic']
predictions=[]
#for i in range(len(labels)):
 #   model_path='/kaggle/input/models2/model{}'.format(i)
  #  model=tf.keras.models.load_model(model_path)
  #  predictions.append(np.argmax(model.predict(images,batch_size=64),axis=1))

image_size=64
input_shape=(None,image_size,image_size,1)

    
for i in range(len(labels)):
    model_path='/kaggle/input/bengali-test10/model'+str(i)
    model=tf.keras.models.load_model(model_path)
    #model=ResNet(input_shape,stacks=3,start_num_filters=16,num_res_blocks=8,
               #batch_norm=True,l2_constant=1e-5,num_classes=num_classes[i])
    #model.build(input_shape)
    #weights_path='/kaggle/input/weights/weights/'+str(i)+'.h5'
    #model.load_weights(weights_path)
    predictions.append(np.argmax(model.predict(images,batch_size=32),axis=1))    
    
df=[]
for i in range(len(images)):            
    for j in range(len(labels)):
        row_id='Test_'+str(i)+'_'+labels[j]
        target=str(predictions[j][i])
        df.append([row_id,target])

df=pd.DataFrame(df,columns=['row_id','target'])
df.to_csv('submission.csv',index=False)