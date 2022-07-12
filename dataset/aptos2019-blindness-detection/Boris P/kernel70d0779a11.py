# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import load_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#print(os.listdir("../input/ep_512_05.hdf5"))
model = load_model('../input/working/ep_512_07.hdf5')
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
#model.load_weights('../input/best-qwk-resnet50-keras-agnos-2019/best_qwk.h5')
predicted = []
SIZE = 512
def flatten_fn(y):
    for i in range(len(y)):
        if y[i] <0.7:
            y[i] = 0
        elif y[i]>0.7 and y[i]<1.5:
            y[i] = 1
        elif y[i]>1.5 and y[i]< 2.5:
            y[i] = 2
        elif y[i] >2.5 and y[i]<3.5:
            y[i] = 3
        elif y[i]> 3.5:
            y[i] = 4
    return int(y[0][0])
def bens_processing(img,IMG_SIZE):
    sigma = 50
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigma) ,-4 ,128)
    return image
def normalization_fn(img):
    img = np.divide(img,255)
    mean_value  = [0.485,0.456,0.406]
    std_value = [0.229,0.224,0.225]
    std_after = []
    mean_after = []
    for i in range(3):
        norm_tmp = np.subtract(img[:,:,i],np.mean(img[:,:,i]))
        std_mult = np.divide(std_value[i],np.std(img[:,:,i], dtype=np.float64))
        img[:,:,i] = np.add(mean_value[i],np.multiply(norm_tmp,std_mult))
        #mean_after.append(np.mean(img[:,:,i]))
        #std_after.append(np.std(img[:,:,i]))
    return img

for i, name in tqdm(enumerate(submit['id_code'])):
    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')
    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE, SIZE))
    img = bens_processing(image,SIZE)
    img = normalization_fn(img)
    X = np.array((img[np.newaxis]))
    val_predict = model.predict(X)
    #print(val_predict)
    #label_predict = flatten_fn(val_predict)
    #X = np.array((image[np.newaxis])/255)
    score_predict=((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    label_predict = np.argmax(score_predict)
    predicted.append(str(label_predict))
submit['diagnosis'] = predicted    
submit.to_csv('submission.csv',index=False)
#submit.to_csv('submission.csv', index=False)
submit.sample(20)
# Any results you write to the current directory are saved as output.