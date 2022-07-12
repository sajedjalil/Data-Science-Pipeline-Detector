
import os
import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
# from PIL import ImageFile
from keras.applications.resnet50 import ResNet50

# img_rows, img_cols, img_channel = 224, 224, 3
# input_tensor_shape=(img_rows, img_cols, img_channel)

def get_data(img_size, split_rate):
    # get train data
    train_label = pd.read_csv("../../data/train_labels.csv")
    img_path = "../../data/train/"
    
    file_paths = []
    y = []
    for i in range(len(train_label)):
        file_paths.append( img_path + str(train_label.iloc[i][0]) +'.jpg' )
        y.append(train_label.iloc[i][1])
    y = np.array(y)

    x = []
    for i, img_path in enumerate(file_paths):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = image.img_to_array(img)   
        x.append(img)
    x = np.array(x)
    
    # data shuffle
    random_index = np.random.permutation(len(y))
    x_shuffle = []
    y_shuffle = []
    for i in range(len(y)):
        x_shuffle.append(x[random_index[i]])
        y_shuffle.append(y[random_index[i]])

    x = np.array(x_shuffle) 
    y = np.array(y_shuffle)
    
    # data split
    val_split_num = int(round(split_rate*len(y)))
    x_train = x[val_split_num:]
    y_train = y[val_split_num:]
    x_test = x[:val_split_num]
    y_test = y[:val_split_num]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return x_train, y_train, x_test, y_test


def build_ResNet50(input_tensor_shape):
    '''
    # reference 
        https://keras.io/applications/#vgg16
        https://www.tensorflow.org/api_docs/python/tf/contrib/keras/applications/ResNet50
    # model defination
        https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/keras/python/keras/applications/resnet50.py
        
    # Arguments
        include_top: whether to include the fully-connected layer at the top of the network.
     
    '''
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape= input_tensor_shape)
    
    x_model = base_model.output
    
    x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model)
    
    x_model = Dense(1024, activation='relu',name='fc1_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_1')(x_model)
    
    x_model = Dense(256, activation='relu',name='fc2_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_2')(x_model)
    predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model
           

# model_save_path = './model.json'
def save_model_to_json(model,model_save_path):
    model_json = model.to_json()
    with open(model_save_path, 'w') as json_file:
        json_file.write(model_json)
    print('model saved')
    

img_size = 224
split_rate = 0.1
# get data
(x_train, y_train, x_test, y_test) = get_data(img_size, split_rate)

# get model
img_rows, img_cols, img_channel = 224, 224, 3
input_tensor_shape=(img_rows, img_cols, img_channel)

model = build_ResNet50(input_tensor_shape)

model_save_path = './model.json'
save_model_to_json(model,model_save_path)

# for i, layer in enumerate(model.layers):
#     if i < 175:
#         print(i, layer.name)

# the following is option. 
# If you want to re-trian the model rather than fine-tune it just feel free to comment it.
# frozen the first 15 layers
for layer in model.layers[:175]:
    layer.trainable = False
for layer in model.layers[175:]:
    layer.trainable = True

# compile the model
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# set train Generator
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(x_train)

# trainning process
nb_epoch = 5
batch_size = 32
checkpointer = ModelCheckpoint(filepath= './ResNet50_weights.hdf5', verbose=1, monitor='val_acc',save_best_only=True, save_weights_only=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch = x_train.shape[0],
                    epochs=nb_epoch,
                    validation_data = (x_test, y_test),
                    callbacks=[checkpointer])
'''
The following part is prediction part.
You can save this part as a separete file to run independently
'''
###############
# load json and create model
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./ResNet50_weights.hdf5')

# get test data
test_no = pd.read_csv("../../data/sample_submission.csv")
test_img_path = "../../data/test/"

test_file_paths = []
test_img_nos = []
for i in range(len(test_no)):
    test_file_paths.append( test_img_path + str(int(test_no.iloc[i][0])) +'.jpg' )
    test_img_nos.append(int(test_no.iloc[i][0]))
test_nos = np.array(test_img_nos)

test = []
for i, img_path in enumerate(test_file_paths):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = image.img_to_array(img)   
    test.append(img)
test = np.array(test)

test_images = test.astype('float32')
test_images /= 255

# make a prediction
predictions = model.predict(test_images)
# write results into csv
sample_submission = pd.read_csv('../../data/sample_submission.csv')
for i, no in enumerate(test_nos):
    sample_submission.loc[sample_submission['name'] == no, 'invasive'] = predictions[i]
sample_submission.to_csv('./submition_ResNet_with_5_epoch.csv', index=False)
#############









