# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf
import cv2
from tqdm import tqdm
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
 #       print(os.path.join(dirname, filename))
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
model = tf.keras.models.load_model('../input/uioiol/aptos_models1.07-0.41.hdf5')
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

predicted = []
for i, name in tqdm(enumerate(submit['id_code'])):
    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')
    image = cv2.imread(path)
    image = cv2.resize(image, (299,299))
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img1=np.expand_dims(img1,axis=0)
    
    
    score_predict=model.predict(img1)
    label_predict = np.argmax(score_predict)
    predicted.append(label_predict)
submit['diagnosis'] = predicted
submit.to_csv('submission.csv', index=False)
