import numpy as np 
import pandas as pd 
import os , time
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm 
from PIL import Image 
import keras
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.notebook import tqdm
import random
import torch

import subprocess

'''
bashCommand = "for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(output)
print(error)
'''

# install MTCNN 
os.system('pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl') 

# reference for MTCNN in facenet_pytorch: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py 
from facenet_pytorch import MTCNN

face_shape = (244,244,3)

# not used in current implementation
def examples_generator(video_files, batch_size, detector, n_frames):
    batch_start = 0
    batch_end = batch_size
    while True:
        test_examples = []
        limit = min(batch_end, len(video_files))
        for video in video_files[batch_start:limit]:
            num_faces = 0
            v_cap = cv2.VideoCapture(video)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_stack = np.linspace(0, v_len - 1, n_frames).astype(int)
            video_name = os.path.basename(video)

            stacked_faces = []
            num_empty_stacked_faces = 0

            for j in range(v_len):
                v_cap.grab()
                if j in sample_stack:
                    success, frame = v_cap.retrieve()
                    if not success:
                        stacked_faces.append(np.zeros(face_shape))
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)

                    faces = detector(frame)

                    if faces is None:
                        stacked_faces.append(np.zeros(face_shape))
                        num_empty_stacked_faces += 1
                        continue
                    if len(faces) > 1:
                        idx = random.randint(0, len(faces) - 1)
                        stacked_faces.append(faces[idx].permute(1, 2, 0).int().numpy())
                        continue
                    stacked_faces.append(faces[0].permute(1, 2, 0).int().numpy())

            v_cap.release()
            test_examples.append(np.array(stacked_faces))
        print(np.array(test_examples).shape)
        yield np.array(test_examples)

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(video_files):
            batch_start = 0
            batch_end = batch_size

# this is the function I use to make predictions            
def batch_predict(video_files, batch_size, detector, n_frames, num_batches, model):
    predictions = []
    videos = []
    batch_start = 0
    batch_end = batch_size
    for i in range(num_batches):
        test_examples = []
        limit = min(batch_end, len(video_files))
        for video in video_files[batch_start:limit]:
            v_cap = cv2.VideoCapture(video)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_stack = np.linspace(0, v_len - 1, n_frames).astype(int)
            videos.append(os.path.basename(video))

            stacked_faces = []
            num_empty_stacked_faces = 0

            for j in range(v_len):
                v_cap.grab()
                if j in sample_stack:
                    success, frame = v_cap.retrieve()
                    if not success:
                        stacked_faces.append(np.zeros(face_shape))
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)

                    faces = detector(frame)

                    if faces is None:
                        stacked_faces.append(np.zeros(face_shape))
                        num_empty_stacked_faces += 1
                        continue
                    if len(faces) > 1:
                        idx = random.randint(0, len(faces) - 1)
                        stacked_faces.append(faces[idx].permute(1, 2, 0).int().numpy())
                        continue
                    stacked_faces.append(faces[0].permute(1, 2, 0).int().numpy())

            v_cap.release()
            test_examples.append(np.array(stacked_faces))
        print("batch start: " + str(batch_start) + ", batch end: " + str(limit))
        print(np.array(test_examples).shape)

        batch_start += batch_size
        batch_end += batch_size

        predictions.extend(model.predict_on_batch(np.array(test_examples)))

    predictions = np.array(predictions).flatten()

    return pd.DataFrame(data={'filename': videos[:], 'label': predictions[:]})


# process single data point for real time prediction 
def processSingleData(image): 
    #image = image.reshape(-1,244,244,3) 
    image = image.astype('float64')
    image = image/255.0 
    return image 

# import model 
model_path = '../input/cnnlstmmodel/CNN_LSTM.h5' 
model = keras.models.load_model(model_path)

n_frames = 40
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = MTCNN(image_size=244, margin=100, keep_all=True, thresholds=[0.7, 0.95, 0.85], post_process=False, device=device)

# get test video directory 
videos_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/' 
video_files = [videos_dir + x for x in os.listdir(videos_dir) if '.mp4' in x]
video_files.sort() # submission is in alphabetical order I believe 

num_batches = int(np.ceil(len(video_files) / batch_size))
print('# of batches: ' + str(num_batches))

#predictGen = examples_generator(video_files, batch_size, detector, n_frames)

print('predicting')

# make predictions

#predictions = model.predict_generator(predictGen, steps=num_batches, workers=1)

#df = pd.DataFrame(data={'filename': video_predictions[:, 0], 'label': video_predictions[:, 1]})
df = batch_predict(video_files, batch_size, detector, n_frames, num_batches, model)
print(df.head())
#pd.DataFrame(data={'filename': video_files[:], 'label': predictions[:]})
df.sort_values(by='filename', ascending=True, inplace=True)
df.to_csv('submission.csv', index=False)