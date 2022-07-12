import numpy as np 
import pandas as pd 
import os 
from sklearn.utils import class_weight,shuffle,class_weight
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical, Sequence
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import cv2


WORKERS = 2
CHANNEL = 3

import warnings 
warnings.filterwarnings("ignore")
SIZE = 300
NUM_CLASSES = 5
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

x = train_df['id_code']
y = train_df['diagnosis']

x, y = shuffle(x,y,random_state=8)
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
y = to_categorical(y,num_classes=NUM_CLASSES)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,stratify=y, random_state=8)
class My_Generator(Sequence):

    def __init__(self, image_filenames, labels,
                 batch_size, is_train=False,
                 mix=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('../input/train_images/' + sample + '.png')
            img = cv2.resize(img, (SIZE, SIZE))
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        if(self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_y)
        return batch_images, batch_y

    def valid_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('../input/train_images/' + sample + '.png')
            img = cv2.resize(img, (SIZE, SIZE))
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y

num_epochs = 30
batch_size = 32
train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False)
valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)

def build_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = ResNet50(include_top=False,weights=None,input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(1e-3),metrics=['accuracy'])
    return model

model = build_model(input_shape=(SIZE,SIZE,3),n_out=NUM_CLASSES)

model.fit_generator(
    train_mixup,
    class_weight=class_weights,
    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),
    validation_data=valid_generator,
    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),
    epochs=num_epochs,
    verbose=1,
    workers=1, use_multiprocessing=False)

submit = pd.read_csv('../input/sample_submission.csv')
predicted = []

for i, name in tqdm(enumerate(submit['id_code'])):
    image = cv2.imread('../input/test_images/' + name + '.png')
    image = cv2.resize(image,(SIZE, SIZE))
    score_predict = model.predict((image[np.newaxis])/255)
    label_predict = np.argmax(score_predict)
    predicted.append(str(label_predict))
    
submit['diagnosis'] = predicted
submit.to_csv('submission.csv', index=False)
submit.head()


