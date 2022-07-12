#!pip install tensorflow==2.2.0-rc2

import os , time
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm 
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image 
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split
import random
import torch
import json
print(tf.__version__)

# install MTCNN 
os.system('pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl')

# reference for MTCNN in facenet_pytorch: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py 
from facenet_pytorch import MTCNN


face_shape = (244,244,3)

class native_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, frames, batch_size, detector, n_frames, videos_dir, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.frames = frames
        self.data_dir = videos_dir
        self.shuffle = shuffle
        self.size = len(self.frames)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        x_batch = self.__data_generation(index)

        return x_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.frames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples'
        current_indices = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]
        x_batch = []
        print('\npredicting on indices: ' + str(current_indices))
        for idx in current_indices:
            video = self.data_dir + self.frames[idx]
            v_cap = cv2.VideoCapture(video)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_stack = np.linspace(0, v_len - 1, n_frames).astype(int)
            video_name = os.path.basename(video)
            stacked_faces = []

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
                        continue
                    if len(faces) > 1:
                        idx = random.randint(0, len(faces) - 1)
                        stacked_faces.append(faces[idx].permute(1, 2, 0).int().numpy())
                        continue
                    stacked_faces.append(faces[0].permute(1, 2, 0).int().numpy())

            v_cap.release()

            x_batch.append(np.stack(stacked_faces, axis=3).reshape(face_shape[0], face_shape[1], -1))

        return np.array(x_batch)
    
        

# import model 
model_path = '../input/xception-model/Xception_0.h5' 

model = keras.models.load_model(model_path)

n_frames = 40
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = MTCNN(image_size=244, margin=100, keep_all=True, thresholds=[0.7, 0.95, 0.85], post_process=False)

# get test video directory 
videos_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/' 
video_files = sorted([x for x in os.listdir(videos_dir) if x[-4:] == ".mp4"])
video_files.sort() # submission is in alphabetical order I believe 

num_batches = int(np.ceil(len(video_files) / batch_size))
print('# of batches: ' + str(num_batches))

#predictGen = examples_generator(video_files, batch_size, detector, n_frames, videos_dir)
predictGen = native_generator(video_files, batch_size, detector, n_frames, videos_dir)

print('predicting')

# make predictions

predictions = model.predict_generator(predictGen, steps=num_batches)
predictions = np.array(predictions).flatten()
#df = batch_predict(video_files, batch_size, detector, n_frames, num_batches, model) 
df = pd.DataFrame(data={'filename': video_files[:], 'label': predictions[:]})
print(df.head())
df.sort_values(by='filename', ascending=True, inplace=True)
df.to_csv('submission.csv', index=False)