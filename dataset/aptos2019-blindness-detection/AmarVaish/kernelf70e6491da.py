# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
data_dir = Path("../input/aptos2019-blindness-detection")
# Get a pandas dataframe from the data we have in our list 
train_data = "../input/aptos2019-blindness-detection/train.csv"
train_metadata = pd.read_csv(train_data) 

# Get the list of all the images
train_images_dir = data_dir / 'train_images'
train_images = train_images_dir.glob('*.png')
# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []

# Go through all the normal cases. The label for these cases will be 0
for img in train_images:
    train_data.append((img,img.name.split('.')[0]))
    
# print(train_data[0])   

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'id_code'],index=None)

train_data.head()
train_metadata.head()
final_data = pd.merge(train_data,train_metadata,on='id_code')
renamed_data = final_data.rename(index=str, columns={"diagnosis": "label"})
renamed_data.head()

from sklearn.model_selection import train_test_split

train, test = train_test_split(renamed_data, test_size=0.2)
renamed_data = train
renamed_data.head()
# Get the counts for each class
cases_count = renamed_data['label'].value_counts()
print(cases_count)

from keras.models import  Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint

densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model

model = build_model()
model.summary()

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) 
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) 

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return
    
kappa_metrics = Metrics()    


import imgaug.augmenters as iaa
import cv2
from keras.utils import to_categorical

# Augmentation sequence 
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness

def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,5), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=5)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))
            
            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            # generating more samples of the undersampled class
            
            if label==1 and count < batch_size-4:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img3 = seq.augment_image(img)
                aug_img4 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img3 = cv2.cvtColor(aug_img3, cv2.COLOR_BGR2RGB)
                aug_img4 = cv2.cvtColor(aug_img4, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.
                aug_img3 = aug_img3.astype(np.float32)/255.
                aug_img4 = aug_img4.astype(np.float32)/255.
                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                batch_data[count+3] = aug_img3
                batch_labels[count+3] = encoded_label
                batch_data[count+4] = aug_img4
                batch_labels[count+4] = encoded_label
                count +=4
            elif label==2 and count < batch_size-1:
                aug_img1 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                count +=1  
            elif label==3 and count < batch_size-5:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img3 = seq.augment_image(img)
                aug_img4 = seq.augment_image(img)
                aug_img5 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img3 = cv2.cvtColor(aug_img3, cv2.COLOR_BGR2RGB)
                aug_img4 = cv2.cvtColor(aug_img4, cv2.COLOR_BGR2RGB)
                aug_img5 = cv2.cvtColor(aug_img5, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.
                aug_img3 = aug_img3.astype(np.float32)/255.
                aug_img4 = aug_img4.astype(np.float32)/255.
                aug_img5 = aug_img5.astype(np.float32)/255.
                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                batch_data[count+3] = aug_img3
                batch_labels[count+3] = encoded_label
                batch_data[count+4] = aug_img4
                batch_labels[count+4] = encoded_label
                batch_data[count+5] = aug_img5
                batch_labels[count+5] = encoded_label
                count +=5
            elif label==4 and count < batch_size-4:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img3 = seq.augment_image(img)
                aug_img4 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img3 = cv2.cvtColor(aug_img3, cv2.COLOR_BGR2RGB)
                aug_img4 = cv2.cvtColor(aug_img4, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.
                aug_img3 = aug_img3.astype(np.float32)/255.
                aug_img4 = aug_img4.astype(np.float32)/255.
                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                batch_data[count+3] = aug_img3
                batch_labels[count+3] = encoded_label
                batch_data[count+4] = aug_img4
                batch_labels[count+4] = encoded_label
                count +=4
            else:
                count+=1
             
#             count+=1
            if count==batch_size-1:
                break
            
        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0

from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
# opt = RMSprop(lr=0.0001, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-5)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)

batch_size = 16
nb_epochs = 20

# Define the number of training steps
nb_train_steps = renamed_data.shape[0]//batch_size

# Get a train data generator
train_data_gen = data_gen(data=renamed_data, batch_size=batch_size)
# print(train_data_gen)
X = test["image"].values
Y = test["label"].values

validation_X = []
valid_labels = []
count=0
for img in X:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    validation_X.append(img)
    label = to_categorical(Y[count], num_classes=5)
    valid_labels.append(label)
    count+=1
    
validation_X = np.array(validation_X)
valid_labels = np.array(valid_labels)

# # Fit the model
history = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                               validation_data=(validation_X, valid_labels),
#                                callbacks=[es, chkpt],
                               callbacks=[kappa_metrics]
#                                class_weight={0:1.0, 1:4.0, 2:2.0, 3:7.0, 4:5}
                             )
# model.load("model.h5")
from tqdm import tqdm
test_data = "../input/aptos2019-blindness-detection/test.csv"
test_metadata = pd.read_csv(test_data) 

def preprocess_image(imagePath):
    img = cv2.imread(str(imagePath))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    return img

N = test_metadata.shape[0]    
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_metadata['id_code'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    )
 
# print(test_data2[0])    
prediction = model.predict(x_test)
# print(prediction)
y_classes = prediction.argmax(axis=-1)
# print(y_classes[124])

test_metadata['diagnosis'] = y_classes
test_metadata.to_csv('submission.csv',index=False)