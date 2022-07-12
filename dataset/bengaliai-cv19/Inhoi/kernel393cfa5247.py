# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import tensorflow.compat.v1 as tf #modyficatin for tensorflow 2.1 might follow soon
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import io
from matplotlib.image import imsave
import csv
import os
import time
import gc
import cv2

def make_predict_batch(img,export_path):
    """
    INPUT
        -`img` list of bytes representing the images to be classified
        
    OUTPUT
        -dataframe containing the probabilities of the labels and the la
        els as columnames
    """
    
    
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], export_path)
        graph = tf.get_default_graph()
        
        feed_dict={'Placeholder:0':img}
        y_pred=sess.run(['Softmax:0','Tile:0'],feed_dict=feed_dict)
        
        if len(img)==1:
            labels=[label.decode() for label in y_pred[1]]
        else:
            labels=[label.decode() for label in y_pred[1][0]]
        
    return pd.DataFrame(data=y_pred[0],columns=labels)


HEIGHT = 137
WIDTH = 236
SIZE = 128

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
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

def make_submit(images,height=137,width=236):
    """
    
    """
    consonant_path='../input/automl-vision'
    root_path='../input/automl-vision5'
    vowel_path='../input/automl-vision5'
    num=images.shape[0]
    #transform the images from a dataframe to a list of images and then bytes
    image_id=images.image_id
    #images=images.iloc[:, 1:].values.reshape(-1, height, width)
    imagebytes=[]
    data = 255 - images.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)  
    for i in range(num):
        img = (data[i] * (255.0 / data[i].max())).astype(np.uint8)
        img = crop_resize(img)
        imageBytearray=io.BytesIO()
        imsave(imageBytearray,img,format='png')
        imagebytes.append(imageBytearray.getvalue())
    
    #get the predictions from the three models - passing the bytes_list
    start_pred=time.time()
    prediction_root=make_predict_batch(imagebytes,export_path=root_path)
    prediction_consonant=make_predict_batch(imagebytes,export_path=consonant_path)
    prediction_vowel=make_predict_batch(imagebytes,export_path=vowel_path)
    end_pred=time.time()
    print('Prediction took {} seconds.'.format(end_pred-start_pred))
    
    start_sub=time.time()
    p0=prediction_root.idxmax(axis=1)
    p1=prediction_vowel.idxmax(axis=1)
    p2=prediction_consonant.idxmax(axis=1)
        
    row_id = []
    target = []
    for i in range(len(image_id)):
        row_id += [image_id.iloc[i]+'_grapheme_root', image_id.iloc[i]+'_vowel_diacritic',image_id.iloc[i]+'_consonant_diacritic']
        target += [p0[i].split('_')[0], p1[i].split('_')[1], p2[i].split('_')[2]]
        
    submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
    #submission_df.to_csv(name, index=False)
        
    end_sub=time.time()
    print('Writing the submission_df took {} seconds'.format(end_sub-start_sub))
    return submission_df

with open('submission.csv','w') as sub:
    writer=csv.writer(sub)
    writer.writerow(['row_id','target'])

batchsize=1000

start = time.time()
for i in range(4):
    start1 = time.time()
    name=f'test_image_data_{i}.parquet'
    print('start with '+name+'...')
    test_img = pd.read_parquet('../input/bengaliai-cv19/'+name)
    print('starting prediction')
    start1 = time.time()
    #split into smaler filesl
    for r in range(np.ceil(test_img.shape[0]/batchsize).astype(int)):
            
        df=make_submit(test_img[r*batchsize:np.minimum((r+1)*batchsize,test_img.shape[0]+1)])
        df.to_csv('submission.csv',mode='a',index=False,header=False)
    
    end1 = time.time()
    print(end1 - start1)
    del test_img

end = time.time()
print(end - start)