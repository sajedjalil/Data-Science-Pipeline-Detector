import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing import image
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

from IPython.display import Image
print(os.listdir("../input"))


train_img = '../input/aerial-cactus-identification/train/train/' 
test_img = '../input/aerial-cactus-identification/test/test/' 
train = '../input/aerial-cactus-identification/train.csv'
df_test = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

all_label = pd.read_csv(train)
all_label['has_cactus'] = all_label['has_cactus'].astype(str)
train_label, val_label = train_test_split(all_label, test_size = 0.2, random_state = 0)



train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.1,
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size=128

img_size = 100
train_generator = train_datagen.flow_from_dataframe(dataframe=train_label,
                                                    directory=train_img,
                                                    x_col='id',
                                                    y_col='has_cactus',
                                                    class_mode='binary',
                                                    batch_size=batch_size,
                                                    target_size=(img_size,img_size))

val_generator = test_datagen.flow_from_dataframe(dataframe=val_label,
                                                 directory=train_img,
                                                 x_col='id',
                                                 y_col='has_cactus',
                                                 class_mode='binary',
                                                 batch_size=50,
                                                 target_size=(img_size,img_size))

                                              
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5), padding = 'same', activation='relu',input_shape=(img_size,img_size,3)))
model.add(layers.Conv2D(32,(5,5), padding = 'same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64,(3,3), padding = 'same', activation='relu'))
model.add(layers.Conv2D(64,(3,3), padding = 'same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(128,(3,3),padding = 'same', activation = 'relu'))
model.add(layers.Conv2D(128,(3,3), padding = 'same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1,activation='sigmoid'))


#lr = 0.001
#adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, verbose=1, baseline=None, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5)

epochs=200
history=model.fit_generator(train_generator,
                            steps_per_epoch=train_label.shape[0]//batch_size,
                            epochs=epochs,
                            validation_data=val_generator,
                            validation_steps=val_label.shape[0]//50,
                            callbacks=[es, reduce_lr],
                            verbose=2,
                           )

plt.figure()
acc_train = history.history['acc']
acc_val = history.history['val_acc']

epochs_ = range(0, len(acc_train))
plt.plot(epochs_, acc_train, label='training accuracy')
plt.plot(epochs_, acc_val, label="validation accuracy")
plt.xlabel('no of epochs')
plt.ylabel('accuracy')

plt.title("no of epochs vs accuracy")
plt.legend()
#plt.savefig('acc_20190627.png')

plt.figure()
loss_train = history.history['loss']
loss_val = history.history['val_loss']

epochs_ = range(0, len(loss_train))
plt.plot(epochs_, loss_train, label='training loss')
plt.plot(epochs_, loss_val, label="validation loss")
plt.xlabel('No of epochs')
plt.ylabel('loss')

plt.title('no of epochs vs loss')
plt.legend()
#plt.savefig('loss_20190627.png')

# Save Model
#model.save("cnn_20190627.h5")

# Load Model
#model = load_model('../input/cactus-model/cnn_20190627.h5')

def batch_predict(directory,samples, batch_size, df):
    prob = np.zeros(shape=(samples, 1))
    generator=test_datagen.flow_from_dataframe(dataframe=df,
                                            directory=directory,
                                            x_col='id',
                                            y_col='has_cactus',
                                            class_mode='other',
                                            batch_size=batch_size,
                                            shuffle=False,
                                            target_size=(img_size,img_size))
    i=0
    for input_batch,_ in generator:
        prob_batch = model.predict_proba(input_batch)
        prob[i*batch_size:(i+1)*batch_size] = prob_batch
        i += 1
        if(i*batch_size > samples):
            break
    
    return prob

df_test.has_cactus = df_test.has_cactus.astype(str)
test_prob = batch_predict(test_img, df_test.shape[0], 10000, df_test)
df = pd.DataFrame({'id':df_test['id'] })
df['has_cactus']= test_prob
df.to_csv("submission.csv",index=False)