import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import random
import sys
import cv2
import json
import PIL, PIL.Image
from imageio import imread
import numpy as np

configuration_path = sys.argv[1]
with open(configuration_path) as f:
    configuration = json.load(f)
    
# path to the model weights file.
weights_path = configuration['weightsPath']
top_model_weights_path = configuration['topModelWeightsPath']
#weights_path = '../keras/examples/vgg16_weights.h5'
#top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = configuration['trainFolder']
validation_data_dir = configuration['validationFolder']
nb_train_samples = configuration["nb_train_samples"]
nb_validation_samples = configuration["nb_validation_samples"]
nb_epoch = configuration['nb_epoch']
train_data_shape = (1, 512, 7, 7)
        
def generate_arrays_from_bottleneck_folder(path):
    '''
    Generator that reads the precomputed weights from the files
    and gives it to the trainer.
    
    It shuffles the entries on every epoch.
    '''
    labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    labels.sort()
    
    labels_map = {}
    for i in range(len(labels)):
        label = labels[i]
        labels_map[label] = i
    
    all_images = []
    
    for i in range(len(labels)):
        label = labels[i]
        images = [str(a) for a in os.listdir(os.path.join(path, label)) if os.path.isfile(os.path.join(path, label, a))]
        images.sort()
        for image in images:
            all_images.append((label, image))
    
    while 1:
        
        random.shuffle(all_images)
        for i in range(len(all_images)):
            entry = all_images[i]
            label = entry[0]
            image = entry[1]
            
            x = np.load(open(os.path.join(path, label, image)))
            y = np.zeros((1, len(labels)))
            y[0, labels_map[label]] = 1

            yield (x, y)
                    
def write_bottlenecks_to_file(bottlenecks_folder, model, images_path):
    '''
    Precomputes the values of the weights for all images
    and saves them in a folder in order to save time.
    '''
    
    #To use with RGB images
    #mean_pixel = [123.68, 116.779, 103.939]
    
    #To use with BGR images
    mean_pixel = [103.939, 116.779, 123.68]
        
    labels = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
    labels.sort()
    line_count = 0
    
    for i in range(len(labels)):
        label = labels[i]
        if not os.path.exists(os.path.join(bottlenecks_folder, label)):
            os.makedirs(os.path.join(bottlenecks_folder, label))
            
        images = [str(a) for a in os.listdir(os.path.join(images_path, label)) if os.path.isfile(os.path.join(images_path, label, a))]
        images.sort()
        for image in images:
            print(label + ": " + image)
            image_path = os.path.join(images_path, label, image)
                
            im = imageio.imread(image_path)
            pil_img = PIL.Image.fromarray(im)
            resized = pil_img.resize((224, 224))
            im = np.asarray(resized).astype(np.float32)
            
            #Needed here to transform RGB into BGR
            np.roll(im, 1, axis = -1)
            for c in range(3):
                im[:, :, c] = im[:, :, c] - mean_pixel[c]
            im = im.astype(np.float32, copy=False)
            im = im.transpose((2, 0, 1))
            im = np.expand_dims(im, axis = 0)
                
            prediction = model.predict(im)
            if line_count == 0:
                global train_data_shape
                train_data_shape = prediction.shape
            
            np.save(open(os.path.join(bottlenecks_folder, label, image + ".npy"), 'w+'), prediction)
            line_count += 1
            
def save_bottlebeck_features():

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    
    bottlenecksFolder = configuration['bottlenecksFolder']
    if not os.path.exists(bottlenecksFolder):
        os.makedirs(bottlenecksFolder)
    
    if not os.path.exists(os.path.join(bottlenecksFolder, 'train')):
        os.makedirs(os.path.join(bottlenecksFolder, 'train'))
    if not os.path.exists(os.path.join(bottlenecksFolder, 'validation')):
        os.makedirs(os.path.join(bottlenecksFolder, 'validation'))
    
    write_bottlenecks_to_file(os.path.join(bottlenecksFolder, 'train'), 
                              model, 
                              train_data_dir)
    
    write_bottlenecks_to_file(os.path.join(bottlenecksFolder, 'validation'), 
                              model, 
                              validation_data_dir)

def train_top_model():
    '''
    Trains the last layers of the vgg-16 model
    '''
    bottlenecksFolder = configuration['bottlenecksFolder']
    train_path = os.path.join(bottlenecksFolder, "train")
    validation_path = os.path.join(bottlenecksFolder, "validation")
    
    #1. Last layers of the model.
    model = Sequential()
    model.add(Flatten(input_shape=train_data_shape[1:]))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #2. Training.
    print("Fitting model")
    checkpointsFolder = configuration['checkpointsFolder']
    checkpointer = ModelCheckpoint(filepath=checkpointsFolder + "weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)
    model.fit_generator(generator = generate_arrays_from_bottleneck_folder(train_path), 
                        samples_per_epoch = nb_train_samples, 
                        nb_epoch = nb_epoch, 
                        validation_data = generate_arrays_from_bottleneck_folder(validation_path),
                        nb_val_samples = nb_validation_samples,
                        max_q_size = 10,
                        callbacks = [checkpointer])
    
    model.save_weights(top_model_weights_path, overwrite = True)

#save_bottlebeck_features()
train_top_model()