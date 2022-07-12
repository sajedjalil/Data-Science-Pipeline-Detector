__author__ = 'Carlos V: https://kaggle.com/carlosvqubits'

import numpy as np
np.random.seed(2017)
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import os
import glob
import cv2
import datetime
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#from keras.layers import BatchNormalization
#from keras.optimizers import SGD
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import __version__ as keras_version

def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 64), cv2.INTER_LINEAR)
    return img
    
def load_train():
    
    X_train = []
    X_train_id = []
    y_train = []
    b_y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join("..", 'input', "train", fld, "*.jpg")
        print(path)
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
            if fld=="NoF": 
                index_bin=0
            else: 
                index_bin=1
            b_y_train.append(index_bin)
            
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    gc.collect()
    return X_train, y_train, X_train_id, b_y_train
    
def load_test():
    path = os.path.join("..",'input', 'test_stg1', '*.jpg')
    start_time = time.time()    
    
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)
    
    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    gc.collect()
    return X_test, X_test_id
    
def read_and_normalize_train_data():
    start_time = time.time()
    train_data, train_target, train_id, b_y_train = load_train()
    

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    b_y_train = np.array(b_y_train, dtype=np.uint8)      

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)
    #b_y_train = np_utils.to_categorical(b_y_train, 2)
    
    print('Train shape:', train_data.shape)  
    print(train_data.shape[0], 'train samples')
    
    print('Read and process train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train_data, train_target, train_id, b_y_train


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    print('Convert to numpy...')
    test_data = np.array(test_data, dtype=np.uint8)
    
    print('Convert to float...')
    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    
    result1.to_csv(sub_file, index=False)
    print("Saved as: ", sub_file)
    


def create_classifier():
    classifier = Sequential()
            
    classifier.add(ZeroPadding2D((1, 1), input_shape=(64, 128, 3)))
    classifier.add(Convolution2D(8, 3, 3, activation='relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(8, 3, 3, activation = 'relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 64, activation = 'relu'))
    classifier.add(Dropout(0.5))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(Dense(output_dim = 64, activation = 'relu'))
    classifier.add(Dropout(0.5))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(Dense(output_dim = 7, activation = 'softmax'))
    
    #sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=4e-5)
    #adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)     
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return classifier
    
def create_binary_classifier():
    classifier = Sequential()
            
    classifier.add(ZeroPadding2D((1, 1), input_shape=(64, 128, 3)))
    classifier.add(Convolution2D(8, 3, 3, activation='relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(8, 3, 3, activation = 'relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 64, activation = 'relu'))
    classifier.add(Dropout(0.5))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(Dense(output_dim = 64, activation = 'relu'))
    classifier.add(Dropout(0.5))
    #classifier.add(BatchNormalization(axis=-1))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
    #sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=4e-5)
    #adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)    
    classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()
    
def run_cross_validation_create_models(nfolds=10, augmentation=False):
    # input image dimensions
    batch_size = 16
    nb_epoch = 300
    nb_epoch_b = 100
    random_state = 42
    
    #class weights when zooming on train fish:
    #bin_dict ={0: 9.4, 1: 1.0}
    #cat_dict = {0: 1.00, 1: 7.91, 2: 17.77, 3: 20.78, 4: 6.93, 5: 12.02, 6: 3.16}

    #class weights when using original train photos:    
    bin_dict ={0: 7.14, 1: 1.0}
    cat_dict = {0: 1.00, 1: 8.60, 2: 14.69, 3: 25.65, 4: 5.75, 5: 9.77, 6: 2.34}
    
    train_data, train_target, train_id, bin_target = read_and_normalize_train_data()

    #yfull_train = dict()
    #b_yfull_train = dict()
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    b_sum_score = 0
    models = []
    binary_models=[]
    
    for train_index, test_index in kf.split(train_data, bin_target):
        
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        B_train = bin_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        B_valid = bin_target[test_index]
                    
        num_fold += 1
        print('Start Train KFold number {} from {}'.format(num_fold, nfolds))
        #################################################
        bin_model = create_binary_classifier()
                
        print("Binary model split: ")
        print('Split train: ', len(X_train), len(B_train))
        print('Split valid: ', len(X_valid), len(B_valid))
                 
        callbacks_b = [
                EarlyStopping(monitor='val_loss', patience=30, verbose=0),
            ]
        
        if augmentation:
            
            print("Using image augmentation to fit binary model...")            
            
            train_datagen_b = ImageDataGenerator(rotation_range=20,
                                       channel_shift_range=0.15,
                                       zoom_range=0.2,
                                       shear_range=0.1,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       vertical_flip=True,
                                       horizontal_flip = True)
    
            training_set_b = train_datagen_b.flow(X_train, B_train,
                                              batch_size = batch_size,
                                              seed=random_state)
    
    
            test_datagen_b = ImageDataGenerator()
       
            test_set_b = test_datagen_b.flow(X_valid, B_valid,
                                         batch_size = batch_size,
                                         seed=random_state)
            
            
            bin_model.fit_generator(training_set_b,
                                samples_per_epoch=6000,
                                nb_val_samples = 3000,
                                nb_epoch=nb_epoch_b,
                                validation_data=test_set_b,
                                class_weight=bin_dict,
                                verbose=1, 
                                callbacks=callbacks_b)
                                
        else:      
            
            print("Fitting binary model...")
            bin_model.fit(X_train, B_train, batch_size=batch_size, nb_epoch=nb_epoch_b,
                  shuffle=True, verbose=1, validation_data=(X_valid, B_valid),
                  callbacks=callbacks_b)
        
        #print('Saving weights of binary model number {} from {}'.format(num_fold, nfolds))          
        #bin_model.save_weights("weights/binary_model_"+str(num_fold)+".h5")
          
        print('Making predictions for binary model number {} from {}'.format(num_fold, nfolds))          
        
        b_predictions_valid = bin_model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        bscore = log_loss(B_valid, b_predictions_valid)
        print('Binary Score log_loss: ', bscore)
        b_sum_score += bscore*len(test_index)

        # Store valid predictions
        #for i in range(len(test_index)):
           # b_yfull_train[test_index[i]] = b_predictions_valid[i]

        binary_models.append(bin_model)
        
     #################################################

        model = create_classifier()
        
        X_train_fish = X_train[B_train==1]
        Y_train_fish = Y_train[B_train==1][:,[0,1,2,3,5,6,7]]
        X_valid_fish = X_valid[B_valid==1]
        Y_valid_fish = Y_valid[B_valid==1][:,[0,1,2,3,5,6,7]]
        
        print("Categorical model split: ")
        print('Split train: ', len(X_train_fish), len(Y_train_fish))
        print('Split valid: ', len(X_valid_fish), len(Y_valid_fish))
        
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=50, verbose=0),
            ]
        
        if augmentation:
            
            print("Using image augmentation to fit categorical model...")            
            
            train_datagen_fish = ImageDataGenerator(rotation_range=20,
                                       channel_shift_range=0.15,
                                       vertical_flip=True,
                                       horizontal_flip = True)
    
            training_set_fish = train_datagen_fish.flow(X_train_fish, Y_train_fish,
                                              batch_size = batch_size,
                                              seed=random_state)
    
    
            test_datagen_fish = ImageDataGenerator()
       
            test_set_fish = test_datagen_fish.flow(X_valid_fish, Y_valid_fish,
                                         batch_size = batch_size,
                                         seed=random_state)
            
            
            model.fit_generator(training_set_fish,
                                samples_per_epoch=6000,
                                nb_val_samples = 3000,
                                nb_epoch=nb_epoch,
                                validation_data=test_set_fish,
                                class_weight=cat_dict,
                                verbose=1, 
                                callbacks=callbacks)
                                
        else:      
            
            print("Fitting model...")
            model.fit(X_train_fish, Y_train_fish, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=1, validation_data=(X_valid_fish, Y_valid_fish),
                  callbacks=callbacks)
        
        #print('Saving weights of categorical model number {} from {}'.format(num_fold, nfolds))          
        #model.save_weights("weights/categorical_model_"+str(num_fold)+".h5")
          
        print('Making predictions for categorical model number {} from {}'.format(num_fold, nfolds))          
        
        predictions_valid = model.predict(X_valid_fish.astype('float32'), batch_size=batch_size, verbose=2)
        cscore = log_loss(Y_valid_fish, predictions_valid)
        print('Categorical Score log_loss: ', cscore)
        sum_score += cscore*len(test_index)

        # Store valid predictions
        #for i in range(len(test_index)):
          #  yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    b_score = b_sum_score/len(train_data)
    print("Binary log_loss train independent avg: ", b_score)
    
    score = sum_score/len(train_data)
    print("Categorical log_loss train independent avg: ", score)
    
    

    info_string = 'bin_loss_' + str(b_score) +'_cat_loss_' + str(score) + '_folds_' + str(nfolds) + '_bin_ep_' + str(nb_epoch_b) +'_cat_ep_' + str(nb_epoch)
    return info_string, models, binary_models

def run_cross_validation_process_test(info_string, models, binary_models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)
    
    test_data, test_id = read_and_normalize_test_data()

    for i in range(nfolds):
        model = models[i]
        bin_model = binary_models[i]
        num_fold += 1
        print('Start Test KFold number {} from {}'.format(num_fold, nfolds))
        
        print("Binary Model Prediction... ")
        bin_test_prediction = bin_model.predict(test_data, batch_size=batch_size, verbose=2)
        
        print("Categorical Model Prediction... ")
        cat_test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        
        print("Overall Prediction...")        
        test_prediction = np.zeros((len(test_data), 8))
        test_prediction[:, 4] = (1-bin_test_prediction.flatten())
        test_prediction[:, 0] = bin_test_prediction.flatten() * cat_test_prediction[:, 0]
        test_prediction[:, 1] = bin_test_prediction.flatten() * cat_test_prediction[:, 1]
        test_prediction[:, 2] = bin_test_prediction.flatten() * cat_test_prediction[:, 2]
        test_prediction[:, 3] = bin_test_prediction.flatten() * cat_test_prediction[:, 3]
        test_prediction[:, 5] = bin_test_prediction.flatten() * cat_test_prediction[:, 4]
        test_prediction[:, 6] = bin_test_prediction.flatten() * cat_test_prediction[:, 5]
        test_prediction[:, 7] = bin_test_prediction.flatten() * cat_test_prediction[:, 6]
        
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    print(info_string)
    print("Creating submission's file...")
    create_submission(test_res, test_id, info_string)
    
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    #augmentation=False
    augmentation=True
    
    info_string, models, bin_models = run_cross_validation_create_models(nfolds=num_folds, augmentation=augmentation)   
    run_cross_validation_process_test(info_string, models, bin_models)