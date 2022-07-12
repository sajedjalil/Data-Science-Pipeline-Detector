# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
import numpy as np
from IPython.display import YouTubeVideo


video_files = os.listdir("../input/video_level/")
frame_files = os.listdir("../input/video_level/")
video_lvl_record = "../input/video_level/train-1.tfrecord"
frame_lvl_record = "../input/frame_level/train-1.tfrecord"

# now, let's read the frame-level data
# due to execution time, we're only going to read the first video

feat_rgb = []
feat_audio = []
targets = []

num_labels = 4716

def createTargetVec(labels):
    out = np.zeros((1, num_labels))
    for label in labels:
        out[0,label] = 1
    return out

samples = 15
nfea=1024
base = np.empty((samples, 100, nfea))
tars = np.empty((samples, num_labels))
# there are 4716 different labels. therefore we need to predict a vector
# with probabilities. the 5 labels with the highest probability will then be selected 
# for submission.
k = 0
sess = tf.InteractiveSession()
for example in tf.python_io.tf_record_iterator(frame_lvl_record):        
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    labels = tf_seq_example.context.feature['labels'].int64_list.value
    n_frames = len(tf_seq_example.feature_lists.feature_list['rgb'].feature)
    rgb_frame = []
    audio_frame = []
    frame = np.zeros((100, nfea))
    for i in range(100):
        #rgb_frame.append(tf.cast(tf.decode_raw(
        #        tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
        #               ,tf.float32).eval())
        f = tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval()
        frame[i] = f
    base[k] = frame
    tars[k] = createTargetVec(labels[:])
    k += 1
    progress = (k / samples) * 100
    if int(progress) % 10 == 0:
        print("Progress", progress, "%")
    if k >= samples:
        break
sess.close()
    
feat_audio_ = np.asarray(feat_audio)
targets_ = np.asarray(tars)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(base, tars, test_size=0.2, random_state=42)
    
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
    

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(128, input_shape=X_train.shape[1:], return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=64)














