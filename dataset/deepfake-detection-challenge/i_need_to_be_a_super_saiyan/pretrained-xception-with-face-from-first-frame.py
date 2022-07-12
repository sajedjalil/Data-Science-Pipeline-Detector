import numpy as np 
import pandas as pd 
import os 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm 
from PIL import Image 
from tensorflow.keras.applications.xception import decode_predictions
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import *
from keras.callbacks import ModelCheckpoint, EarlyStopping


# install MTCNN 
os.system('pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl') 

# reference for MTCNN in facenet_pytorch: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py 
from facenet_pytorch import MTCNN


# import model 
model_path = '../input/pretrained-xception-prototype2/pretrained_xception_prototype2.h5' 
model = tf.keras.models.load_model(model_path)
model.summary() 

# extract face from a captured frame using facenet_pytorch 
def ExtractFaces(video_path): 
    v_cap = cv2.VideoCapture(video_path) 
    success, frame = v_cap.read()
    face_arr = [] 
    if success: 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        mtcnn = MTCNN(image_size=244,margin=40,keep_all=True,select_largest=False,post_process=False) 
        faces = mtcnn(frame)   
        if faces is None:
            return face_arr # empty array 
        for face in faces: 
            face = face.permute(1,2,0).int().numpy() 
            face_arr.append(face) 
        return face_arr 
    return face_arr # this will be empty  

# process single data point for real time prediction 
def processSingleData(image): 
    image = image.reshape(-1,244,244,3) 
    image = image.astype('float64')
    image = image/255.0 
    return image 

# make predictions by taking the maximum of the probability for all faces 
# we may also try other methods like averaging the probabilities for all faces
def makePredictions(video_paths): 
    predictions = [] 
    trial = 1 
    undetected = 0
    for video_path in video_paths: 
        print("Trial = {}...".format(trial))
        faces = ExtractFaces(video_path)  
        if not faces: # no face was detected 
            predictions.append(0.5) 
            undetected = undetected+1 
            trial = trial+1
            continue 
        fake_probability = 0.0 
        for face in faces: 
            image = processSingleData(face)
            single_prediction = model.predict_on_batch(image)[0]
            single_prediction = np.asarray(single_prediction) 
            fake_probability = np.maximum(fake_probability,single_prediction[0])
        predictions.append(fake_probability)
        trial = trial+1 
    #print("{} videos undetected".format(undetected))
    return predictions 
    
# get test video directory 
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/' 
test_video_files = [test_dir + x for x in os.listdir(test_dir)]
test_video_files.sort() # submission is in alphabetical order I believe 

# make predictions 
predictions = makePredictions(test_video_files) 

# save to output csv file 
ss = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')
ss['label'] = predictions 
ss.to_csv('submission.csv',index=False)

ss.head() 