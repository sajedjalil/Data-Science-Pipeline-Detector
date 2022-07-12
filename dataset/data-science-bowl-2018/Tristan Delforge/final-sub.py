import pandas as pd
import glob
from PIL import Image
import numpy as np
import random
import functools
import operator
import keras.utils
import scipy.misc
import traceback
import multiprocessing
from scipy.ndimage import gaussian_gradient_magnitude, morphology
from scipy.ndimage.morphology import binary_opening
from keras import optimizers
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from keras.layers.merge import concatenate
from keras import backend as K
from keras.layers import Input
import math
from sklearn.svm import SVC
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
import os
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import h5py
import configparser
from sklearn import preprocessing
import ast
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer
import shutil
import gc

max_images = 1000
files_loc = '../input/'
min_nuclei_size = 2
confidence_threshold = .5
full_image_read_size = (64,64)
max_training_iter = 5


def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    res = 0
    intersection = K.sum(y_true_f * y_pred_f)
    return 2 * (intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


def normalize_image(np_image):

    flat_image = np_image.flatten()
    if np.median(flat_image) > 128:
        flip_f = np.vectorize(lambda t: 255-t)
        flat_image = flip_f(flat_image)
    flat_image = np.reshape(flat_image, (-1, 1))

    scaler2 = MinMaxScaler(feature_range=(0, 255))
    flat_image = scaler2.fit_transform(flat_image)

    # scaler1 = QuantileTransformer(n_quantiles= 25)
    # flat_image = scaler1.fit_transform(flat_image)
    #
    # scaler3 = MinMaxScaler(feature_range=(0, 255))
    # flat_image = scaler3.fit_transform(flat_image)



    flat_image = np.rint(flat_image)

    np_image_2 = np.reshape(flat_image, (np_image.shape[0], np_image.shape[1]))

    # scipy.misc.imsave('location_image_1.jpg', np_image)
    # scipy.misc.imsave('location_image_2.jpg', np_image_2)

    # np_image_scaled = np_image

    return np_image_2


#taken from https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
def get_cnn():
    inputs = Input((full_image_read_size[0], full_image_read_size[1], 2))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    adam = optimizers.Adam(lr=.0001, decay=1e-6)
    model.compile(optimizer=adam, loss=IOU_calc_loss, metrics=['acc'])
    return model




#Section: Generate training sets
def generate_input_image_and_masks():
    folders = glob.glob(files_loc + 'stage1_train/*/')
    random.shuffle(folders)

    for count, folder in enumerate(folders):
        try:
            image_location = glob.glob(folder + 'images/*')[0]
            mask_locations = glob.glob(folder + 'masks/*')
            start_image = Image.open(image_location).convert('LA')
            np_image = np.array(start_image.getdata())[:,0]
            np_image = np_image.reshape(start_image.size[1], start_image.size[0])

            masks = []
            for i in mask_locations:
                mask_image = Image.open(i)
                np_mask = np.array(mask_image.getdata())
                np_mask = np_mask.reshape(start_image.size[1], start_image.size[0])
                masks.append(np_mask)
        except OSError:
            continue
        # if count > 10:
        #     break

        yield np_image, masks

#augmentation function
#TODO: add forms of augmentation with changed lighting or a few randomly sligtly altered pixels
def get_subimages(input_image, gradient, input_mask, transpose = False, rotation = 0, mask_non_zero = True, step_size = 60):
    if transpose:
        input_image = np.transpose(input_image)
        input_mask = np.transpose(input_mask)
        gradient = np.transpose(gradient)
    input_image = np.rot90(input_image, rotation)
    input_mask = np.rot90(input_mask, rotation)
    input_gradient = np.rot90(gradient, rotation)

    max_x_subimages  = (input_image.shape[0])//full_image_read_size[0]
    max_y_subimages = (input_image.shape[1]) // full_image_read_size[1]

    x_index = 0
    output = []

    while x_index + full_image_read_size[0] <= input_image.shape[0]:
        y_index = 0
        while y_index + full_image_read_size[1] < input_image.shape[1]:
            x1 = x_index
            x2 = x_index + full_image_read_size[0]
            y1 = y_index
            y2 = y_index + full_image_read_size[1]

            if np.mean(input_mask[x1:x2, y1:y2]) > 0:
                next_input = {'input': np.dstack((np.expand_dims(input_image[x1:x2, y1:y2], axis=2),
                                                  np.expand_dims(input_gradient[x1:x2, y1:y2], axis=2))),
                              'output': np.expand_dims(input_mask[x1:x2, y1:y2], axis=2)}
                output.append(next_input)

            y_index += step_size
        x_index += step_size

    return output


#TODO: merge input creation functions
def get_image_arrays_for_full_location_training(tuple_input):
    # input_image = normalize_image(input_image)
    input_image, masks = tuple_input
    input_image = normalize_image(input_image)

    gradient = gaussian_gradient_magnitude(input_image, sigma=.4)

    mask_sum = functools.reduce(operator.add, masks)
    mask_sum = mask_sum > 0
    mask_sum = mask_sum.astype(int)
    # vectorized = np.vectorize(lambda t: 1 if t>0 else 0)
    # mask_sum = vectorized(mask_sum)

    output = []
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=0))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=1))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=2))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=3))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=0))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=1))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=2))
    output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=3))

    return pd.DataFrame(output)


def get_image_arrays_for_full_edge_training_with_contact(tuple_input):
    input_image, masks = tuple_input
    input_image = normalize_image(input_image)
    print(input_image.shape)

    gradient = gaussian_gradient_magnitude(input_image, sigma=.4)

    mask_sum = input_image.copy()
    mask_sum[:] = 0

    vectorized = np.vectorize(lambda t: 1 if t > 0 else 0)
    vectorized2 = np.vectorize(lambda t: 1 if t > 1 else 0)
    for m in masks:
        temp_edge = gaussian_gradient_magnitude(m, sigma=.4)
        temp_edge = temp_edge > 0
        mask_sum = np.add(temp_edge.astype(int), mask_sum)

    mask_sum = mask_sum > 1
    mask_sum = mask_sum.astype(int)

    output = []

    print(np.mean(mask_sum))

    if np.mean(mask_sum) > 0:
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=0))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=1))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=2))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=3))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=0))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=1))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=2))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=3))

    return pd.DataFrame(output)


def get_image_arrays_for_full_edge_training_without_contact(tuple_input):
    input_image, masks = tuple_input
    input_image = normalize_image(input_image)
    print(input_image.shape)

    gradient = gaussian_gradient_magnitude(input_image, sigma=.4)

    mask_sum = input_image.copy()
    mask_sum[:] = 0

    mask_sum = input_image.copy()
    mask_sum[:] = 0

    vectorized = np.vectorize(lambda t: 1 if t > 0 else 0)
    vectorized2 = np.vectorize(lambda t: 1 if t > 1 else 0)
    for m in masks:
        temp_edge = gaussian_gradient_magnitude(m, sigma=.4)
        temp_edge = temp_edge > 0
        mask_sum = np.add(temp_edge.astype(int), mask_sum)

    mask_sum = mask_sum > 1
    mask_sum = mask_sum.astype(int)

    output = []

    print(np.mean(mask_sum))

    if np.mean(mask_sum) > 0:
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=0))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=1))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=2))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=False, rotation=3))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=0))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=1))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=2))
        output.extend(get_subimages(input_image, gradient, mask_sum, transpose=True, rotation=3))

    return pd.DataFrame(output)


def get_dataframes_for_training_location():
    gen = generate_input_image_and_masks()
    location_dfs = []

    # for count, _ in enumerate(range(max_images)):
    #     print(count)
    #     try:
    #         images, masks = next(gen)
    #         location_dfs.append(get_image_arrays_for_full_location_training(images, masks))
    #     except StopIteration:
    #         traceback.print_exc()
    #         break
    p = multiprocessing.Pool(processes=4)
    location_df = pd.concat(list(p.imap(get_image_arrays_for_full_location_training, gen, chunksize=1)))

    print('images read')
    # location_df = pd.concat(location_dfs, ignore_index=True)
    location_df = location_df.sample(frac=1)

    return location_df


def get_dataframes_for_training_edge_with_contact():
    gen = generate_input_image_and_masks()

    p = multiprocessing.Pool(processes=4)
    edge_df = pd.concat(list(p.imap(get_image_arrays_for_full_edge_training_with_contact, gen, chunksize=1)))
    # get_image_arrays_for_full_edge_training_with_contact(gen.__next__)
    print('images read')
    edge_df = edge_df.sample(frac=1)

    return edge_df


def get_dataframes_for_training_edge_without_contact():
    gen = generate_input_image_and_masks()

    p = multiprocessing.Pool(processes=4)
    edge_df = pd.concat(list(p.imap(get_image_arrays_for_full_edge_training_without_contact, gen, chunksize=1)))

    print('images read')
    edge_df = edge_df.sample(frac=1)

    return edge_df


def get_model_inputs(df, x_labels, test_size = 0.05):
    print('testing inputs: {0}'.format(x_labels))
    print(df.shape)
    x, y = [], []

    #TODO: vectorize
    for _, i in df.iterrows():
        x.append(np.hstack([i[x_label] for x_label in x_labels]))
        y.append(i['output'])

    del df

    gc.collect()
    x = np.array(x)
    y = np.array(y)
    #x = np.nan_to_num(x)

    print('arrays processed')

    print(x.shape, y.shape)
    x_train = x[0:int((1-test_size)*x.shape[0])]
    x_test = x[int((1-test_size)*x.shape[0]):]
    y_train = y[0:int((1-test_size)*x.shape[0])]
    y_test = y[int((1-test_size)*x.shape[0]):]
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    print('inputs preprocessed')

    return x_train, x_test, y_train, y_test



#Section: Train models
def get_loc_model():
    try:
        loc_model = load_model(files_loc + 'dsb2018models/cnn_full_loc2.h5', custom_objects={'IOU_calc_loss':IOU_calc_loss})
    except:
        traceback.print_exc()
        df_loc = get_dataframes_for_training_location()

        x_train, x_test, y_train, y_test = get_model_inputs(df_loc, x_labels=['input'])

        epochs = max_training_iter//x_train.shape[0]
        loc_model = get_cnn()
        loc_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)
        loc_model.save(files_loc + 'cnn_full_loc2.h5')
    return loc_model


def get_edge_model_with_contact():
    try:
        print(glob.glob(files_loc + 'dsb2018models/*'))
        edge_model = load_model(files_loc + 'dsb2018models/cnn_full_edge_connected_only2.h5', custom_objects={'IOU_calc_loss':IOU_calc_loss})
    except:
        traceback.print_exc()

        df_edge = get_dataframes_for_training_edge_with_contact()
        x_train, x_test, y_train, y_test = get_model_inputs(df_edge, x_labels=['input'])
        epochs = max_training_iter // x_train.shape[0]
        edge_model = get_cnn()
        edge_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)
        edge_model.save(files_loc + 'cnn_full_edge_connected_only2.h5')
    return edge_model


def get_edge_model_without_contact():
    try:
        edge_model = load_model(files_loc + 'dsb2018models/cnn_full_edge_not_connected_only.h5', custom_objects={'IOU_calc_loss':IOU_calc_loss})
    except:
        traceback.print_exc()

        df_edge = get_dataframes_for_training_edge_without_contact()
        x_train, x_test, y_train, y_test = get_model_inputs(df_edge, x_labels=['input'])
        epochs = max_training_iter // x_train.shape[0]
        edge_model = get_cnn()
        edge_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)
        edge_model.save(files_loc + 'cnn_full_edge_not_connected_only.h5')
    return edge_model



#Section: Format chaning functions
def get_outputs_from_flat_array(a):
    on_label = False

    image_list = []
    temp_image_list = []
    for count, i in enumerate(a):
        if i != 0 and not on_label:
            temp_image_list = []
            temp_image_list.append(count)
            on_label = True
        elif i != 0 and on_label:
            temp_image_list.append(count)
        elif i == 0 and on_label:
            on_label = False
            image_list.append(temp_image_list)

    res_str = ''
    for i in image_list:
        res_str += str(int(i[0]) + 1)
        res_str += ' '
        res_str += str(len(i))
        res_str += ' '
    res_str = res_str[:-1]

    return res_str


def to_output_format(label_dict, np_image, image_name):
    output_dicts = []
    for i in label_dict.keys():

        image_copy = np_image.copy()
        image_copy.fill(0)
        for j in label_dict[i]:
            image_copy[j[0], j[1]] = 1
        image_copy = np.transpose(image_copy)
        flat_image = image_copy.flatten()

        output_dict = dict()
        output_dict['ImageId'] = image_name
        output_dict['EncodedPixels'] = get_outputs_from_flat_array(flat_image)
        output_dicts.append(output_dict)

    return output_dicts




#Section: Generate testing files
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def run_tests(loc_model, edge_model_with_contact, edge_model_without_contact):
    folders = glob.glob(files_loc + 'stage1_train/*/')

    output_dir = create_directory('training_output/')
    random.shuffle(folders)

    output_dicts = []

    for folder in folders:
        image_location = glob.glob(folder + 'images/*')[0]
        mask_locations = glob.glob(folder + 'masks/*')
        pre_image = Image.open(image_location)
        start_image = Image.open(image_location).convert('LA')
        image_id = os.path.basename(image_location).split('.')[0]

        image_output_dir = create_directory(output_dir + image_id + '/')
        src_image_output_loc = create_directory(image_output_dir+ 'input' + '/')
        src_image_output_loc2 = src_image_output_loc+ image_id + '.png'
        pre_image.save(src_image_output_loc + 'im.png', format = 'png')
        #shutil.copy2(image_location, src_image_output_loc)

        mask_output_dir = create_directory(image_output_dir  + 'masks' + '/' )
        for count, m in enumerate(mask_locations):
            try:
                mask_id = os.path.basename(m).split('.')[0]
                mask_output_loc = mask_output_dir
                mask_im = Image.open(m)
                mask_im.save( mask_output_loc + str(count) + '.png', format = 'png')
                #shutil.copy(m, mask_output_loc)
            except:
                pass

        np_image = np.array(start_image.getdata())[:, 0]
        np_image = np_image.reshape(start_image.size[1], start_image.size[0])

        np_image = normalize_image(np_image)
        output, clusters, nuclie_predictions, edge_predictions_with_contact, edge_predictions_without_contact = \
            predict_image(loc_model, edge_model_with_contact, edge_model_without_contact, np_image, image_id)

        pred_output_dir1 = create_directory(image_output_dir  + 'predicted_touching_edges' + '/')
        pred_output_dir2 = create_directory(image_output_dir  + 'predicted_non_touching_edges' + '/')
        pred_output_dir3 = create_directory(image_output_dir  + 'predicted_nuclei_locations' + '/')
        pred_output_dir4 = create_directory(image_output_dir  + 'predicted_clusters' + '/')

        print(os.path.exists(pred_output_dir1))
        print(os.path.exists(pred_output_dir2))
        print(os.path.exists(pred_output_dir3))
        print(os.path.exists(pred_output_dir4))

        scipy.misc.imsave(pred_output_dir1 + 'im.png', edge_predictions_with_contact)
        scipy.misc.imsave(pred_output_dir2 + 'im.png', edge_predictions_without_contact)
        scipy.misc.imsave(pred_output_dir3+ 'im.png', nuclie_predictions)
        with open(pred_output_dir4 + 'predicted_clusters.plk', 'wb') as output_plk:
            pickle.dump(clusters, output_plk)




#Section: Generate output
#removes pixels that are alone or ina  cluster smaller than the minimum size
def get_valid_pixels(locations):
    location_set = set(locations)
    prediction_n_locations = location_set
    valid_locations = set()

    counter = 0
    while len(prediction_n_locations) > 0:
        starting_location = prediction_n_locations.pop()
        prediction_n_locations.add(starting_location)

        temp_neucli_locations = set([starting_location])
        failed_locations = set()

        while True:
            location_added = False

            search_set = temp_neucli_locations - failed_locations

            for n_loc in search_set:
                if (n_loc[0] + 1, n_loc[1]) in prediction_n_locations and (n_loc[0] + 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0] + 1, n_loc[1]))
                    location_added = True
                if (n_loc[0] - 1, n_loc[1]) in prediction_n_locations and (n_loc[0] - 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0] - 1, n_loc[1]))
                    location_added = True
                if (n_loc[0], n_loc[1] + 1) in prediction_n_locations and (n_loc[0], n_loc[1] + 1) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0], n_loc[1] + 1))
                    location_added = True
                if (n_loc[0], n_loc[1] - 1) in prediction_n_locations and (n_loc[0], n_loc[1] - 1) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0], n_loc[1] - 1))
                    location_added = True
                failed_locations.add(n_loc)
            if not location_added:
                break
        prediction_n_locations = prediction_n_locations - temp_neucli_locations
        if len(temp_neucli_locations) >= min_nuclei_size or len(valid_locations) == 0:
            valid_locations.update(temp_neucli_locations)
            counter += 1

    return valid_locations

#extract clusters from set of locations
def get_nuclei_from_predictions(locations, image_id):
    location_set = set(locations)
    prediction_n_locations = location_set
    nuclei_predictions = dict()

    counter = 0
    while len(prediction_n_locations) > 0:
        starting_location = prediction_n_locations.pop()
        prediction_n_locations.add(starting_location)

        temp_neucli_locations = set([starting_location])
        failed_locations = set()

        while True:
            location_added = False

            search_set = temp_neucli_locations - failed_locations

            for n_loc in search_set:
                if (n_loc[0] + 1, n_loc[1]) in prediction_n_locations and (n_loc[0] + 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0] + 1, n_loc[1]))
                    location_added = True
                if (n_loc[0] - 1, n_loc[1]) in prediction_n_locations and (n_loc[0] - 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0] - 1, n_loc[1]))
                    location_added = True
                if (n_loc[0], n_loc[1] + 1) in prediction_n_locations and (n_loc[0], n_loc[1] + 1) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0], n_loc[1] + 1))
                    location_added = True
                if (n_loc[0], n_loc[1] - 1) in prediction_n_locations and (n_loc[0], n_loc[1] - 1) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0], n_loc[1] - 1))
                    location_added = True
                failed_locations.add(n_loc)
            if not location_added:
                break
        prediction_n_locations = prediction_n_locations - temp_neucli_locations
        # if len(temp_neucli_locations) > min_nuclei_size or len(nuclei_predictions.keys()) == 0:
        #     nuclei_predictions[counter] = temp_neucli_locations
        #     counter += 1
        if len(temp_neucli_locations) > 0:
            nuclei_predictions[counter] = temp_neucli_locations
            counter += 1

        print('clusters found:', len(nuclei_predictions), ' pixels_left:', len(prediction_n_locations),image_id)
    return nuclei_predictions


def prediction_image_to_location_list(prediction_image):
    output = []
    for i in range(prediction_image.shape[0]):
        for j in range(prediction_image.shape[1]):
            if prediction_image[i,j] > 0:
                output.append((i,j))
    return output


#values to classify loose pixels
def point_to_cluster_classification_model_features(t):
    return [t[0], t[1], t[0] / (t[1] + 1), (t[1])/(t[0] + 1), t[0]+t[1], (t[0]+t[1])**2, (t[0]+t[1])/((t[0]+t[1] + 1)**2)]


#classify loose pixels into clusters
def train_cluster_model(clusters, v_locations):
    x = []
    y = []

    current_points = functools.reduce(operator.or_, [i for _, i in clusters.items()])
    points_to_predict = set(v_locations) - current_points

    print('classifying unclustered pixels:', len(points_to_predict), len(current_points))

    for i in clusters.keys():

        #duplicating record for consistent sample size, also
        if len(clusters[i]) == 1:
            for j in clusters[i]:
                x.append(np.array(point_to_cluster_classification_model_features(j)))
                y.append(np.array([int(i)]))
        for j in clusters[i]:
            x.append(np.array(point_to_cluster_classification_model_features(j)))
            y.append(np.array([int(i)]))
    x = np.array(x)
    y = np.array(y)
    y = np.ravel(y)

    #x1,x2,y1,y2 = train_test_split(x, y, shuffle=True)

    pred_x = []
    for i in points_to_predict:
        pred_x.append(np.array(point_to_cluster_classification_model_features(i)))
    pred_x = np.array(pred_x)

    clf = ExtraTreesClassifier(n_jobs=-1)

    # clf.fit(x1, y1)
    # print(clf.score(x2,y2))

    clf.fit(x, y)

    try:
        predictions = clf.predict(pred_x)
        for i, j in zip(points_to_predict, predictions):
            clusters[j].add(i)
            current_points.add(i)
    except:
        traceback.print_exc()
        print(pred_x.shape)

    return clusters


def locations_to_np_array(locations, np_image):
    np_image = np_image.copy()
    np_image[:] = 0
    for l in locations:
        np_image[l[0], l[1]] = 1
    return np_image


def split_clusters(clusters, edges1, edges2, np_image, image_id):

    adj_cluster_list = []

    #scaling for size, very arbitrary at the moment
    average_cluster_size = sum([len(i) for _, i in clusters.items()])/len(clusters.keys())
    if average_cluster_size>300:
        opening_split_size = 3
    elif average_cluster_size>150:
        opening_split_size = 2
    else:
        opening_split_size = 1


    for _, c in clusters.items():
        try:
            cluster_sub_edges_locations = c - set(edges1)
            adj_cluster1 = get_nuclei_from_predictions(cluster_sub_edges_locations, image_id)
            cluster1_locations = functools.reduce(operator.or_, [i for _, i in adj_cluster1.items()])
        except:
            traceback.print_exc()
            adj_cluster_list.append({0: c})
            continue

        # try:
        #     cluster_sub_edges_locations2 = cluster_sub_edges_locations - set(edges2)
        #     adj_cluster2 = get_nuclei_from_predictions(cluster_sub_edges_locations2, image_id)
        #     cluster2_locations = functools.reduce(operator.or_, [i for _, i in adj_cluster2.items()])
        #
        #     adj_cluster_count1 = len(adj_cluster1.keys())
        #     adj_cluster_count2 = len(adj_cluster2.keys())
        #
        #     if adj_cluster_count2 > adj_cluster_count1:
        #         adj_cluster = adj_cluster2
        #         cluster_locations = cluster1_locations
        #     else:
        #         adj_cluster = adj_cluster1
        #         cluster_locations = cluster2_locations
        # except:
        #     traceback.print_exc()
        #     adj_cluster = adj_cluster1
        #     cluster_locations = cluster1_locations

        cluster_locations = cluster1_locations

        image_without_edges = locations_to_np_array(cluster_locations, np_image)
        image_without_edges = binary_opening(image_without_edges, iterations=opening_split_size)
        n_locations = prediction_image_to_location_list(image_without_edges)
        adj_cluster= get_nuclei_from_predictions(n_locations, image_id)
        adj_cluster_count= len(adj_cluster.keys())

        if adj_cluster_count > 1:
            adj_cluster_list.append(adj_cluster)
        else:
            adj_cluster_list.append({0:c})

    adj_clusters = dict()

    cluster_id = 0
    for i in adj_cluster_list:
        for k, v in i.items():
            adj_clusters[cluster_id] = v
            cluster_id += 1
    return adj_clusters


def get_outputs(input_dict):
    output, edges_with_contact, edges_without_contact, np_image, image_id = input_dict['output_n'], input_dict['edges_with_contact'], input_dict['edges_without_contact'], input_dict['np_image'], input_dict['image_id']

    v_locations = prediction_image_to_location_list(output)
    v_locations = get_valid_pixels(v_locations)
    edge_with_contact_locations = prediction_image_to_location_list(edges_with_contact)
    edge_without_contact_locations = prediction_image_to_location_list(edges_without_contact)
    clusters = get_nuclei_from_predictions(v_locations, image_id)

    while True:
        clusters_split = split_clusters(clusters, edge_with_contact_locations, edge_without_contact_locations, np_image, image_id)
        if len(clusters.keys()) == len(clusters_split.keys()):
            break
        else:
            clusters = clusters_split
    clusters = train_cluster_model(clusters, v_locations)
    formated_output = to_output_format(clusters, np_image, image_id)
    return formated_output, clusters


def predict_subimages(input_image, gradient, transpose, rotation, model):
    if transpose:
        input_image = np.transpose(input_image)
        gradient = np.transpose(gradient)
    input_image = np.rot90(input_image, rotation)
    input_gradient = np.rot90(gradient, rotation)


    x_index = 0
    outputs = []
    step_size = 16

    #input_dict = {}

    while x_index + full_image_read_size[0] < input_image.shape[0]:
        y_index = 0

        row_list = []
        while y_index + full_image_read_size[1] < input_image.shape[1]:
            x1 = x_index
            x2 = x_index + full_image_read_size[0]
            y1 = y_index
            y2 = y_index + full_image_read_size[1]
            #input_dict[(x1, x2, y1, y2)] = input_image[x1:x2,y1:y2]
            temp_image = input_image[x1:x2,y1:y2]
            temp_gradient = input_gradient[x1:x2,y1:y2]
            model_input = np.dstack([np.expand_dims(temp_image, axis=2), np.expand_dims(temp_gradient, axis=2)])
            model_input = np.expand_dims(model_input, axis = 0)
            prediction = np.squeeze(model.predict(model_input))
            pad = np.zeros(input_image.shape)
            pad[:] = np.nan
            pad[x1:x2, y1:y2] = prediction
            rotated_output = np.rot90(pad, 4 - rotation)
            row_list.append(rotated_output)
            #outputs.append(rotated_output)


            y_index += step_size
        outputs.append(np.nanmean(np.dstack(row_list), 2))
        x_index += step_size


    # max_x_subimages  = (input_image.shape[0])//full_image_read_size[0]
    # max_y_subimages = (input_image.shape[1]) // full_image_read_size[1]
    #
    # output = []
    # for i in range(max_x_subimages):
    #     predictions = []
    #     for j in range(max_y_subimages):
    #         x1 = i * full_image_read_size[0]
    #         x2 = (1 + i) * full_image_read_size[0]
    #         y1 = j * full_image_read_size[0]
    #         y2 = (1 + j) * full_image_read_size[0]
    #         temp_image = input_image[x1:x2,y1:y2]
    #         temp_gradient = input_gradient[x1:x2,y1:y2]
    #         model_input = np.dstack([np.expand_dims(temp_image, axis=2), np.expand_dims(temp_gradient, axis=2)])
    #         model_input = np.expand_dims(model_input, axis = 0)
    #         prediction = model.predict(model_input)
    #         prediction = np.squeeze(prediction)
    #         predictions.append(prediction)
    #     output.append(np.concatenate(predictions, 1))
    # output = np.concatenate(output, 0)
    # pad = np.zeros(input_image.shape)
    # pad[:] = np.nan
    # pad[:output.shape[0],:output.shape[1]] = output
    # output = pad
    #
    # rotated_output = np.rot90(output, 4-rotation)
    return outputs


def predict_image(loc_model, edge_model_with_contact, edge_model_without_contact, np_image, image_id):
    image_gradient = gaussian_gradient_magnitude(np_image, sigma=.4)
    results = []
    results.extend(predict_subimages(np_image, image_gradient, False, 0, loc_model))
    results.extend(predict_subimages(np_image, image_gradient, False, 1, loc_model))
    results.extend(predict_subimages(np_image, image_gradient, False, 2, loc_model))
    results.extend(predict_subimages(np_image, image_gradient, False, 3, loc_model))
    result_array = np.dstack(results)

    result_mean = np.nanmean(result_array, 2)
    print(result_mean.shape)

    prediction_f = np.vectorize(lambda t: 1 if t > confidence_threshold else 0)
    nuclie_predictions = prediction_f(result_mean)

    edges_with_contact = []
    edges_with_contact.extend(predict_subimages(np_image, image_gradient, False, 0, edge_model_with_contact))
    edges_with_contact.extend(predict_subimages(np_image, image_gradient, False, 1, edge_model_with_contact))
    edges_with_contact.extend(predict_subimages(np_image, image_gradient, False, 2, edge_model_with_contact))
    edges_with_contact.extend(predict_subimages(np_image, image_gradient, False, 3, edge_model_with_contact))
    edge_array_with_contact = np.dstack(edges_with_contact)

    edges_without_contact = []
    # edges_without_contact.extend(predict_subimages(np_image, image_gradient, False, 0, edge_model_without_contact))
    # edges_without_contact.extend(predict_subimages(np_image, image_gradient, False, 1, edge_model_without_contact))
    # edges_without_contact.extend(predict_subimages(np_image, image_gradient, False, 2, edge_model_without_contact))
    # edges_without_contact.extend(predict_subimages(np_image, image_gradient, False, 3, edge_model_without_contact))
    edge_array_without_contact = np.dstack(edges_with_contact)

    edge_mean_with_contact = np.nanmean(edge_array_with_contact, 2)
    print(edge_mean_with_contact.shape)
    edge_predictions_with_contact = prediction_f(edge_mean_with_contact)

    edge_mean_without_contact = np.nanmean(edge_array_without_contact, 2)
    print(edge_mean_without_contact.shape)
    edge_predictions_without_contact = prediction_f(edge_mean_without_contact)
    # edge_predictions_without_contact =np_image.copy()
    # edge_predictions_without_contact[:] = 0

    input_dict = {'output_n':nuclie_predictions, 'edges_with_contact': edge_predictions_with_contact, 'edges_without_contact': edge_predictions_without_contact, 'image_id':image_id, 'np_image':np_image}

    output_dicts, clusters = get_outputs(input_dict)
    #output_dicts.extend()
    return output_dicts, clusters, nuclie_predictions, edge_predictions_with_contact, edge_predictions_without_contact


def run_predictions(loc_model, edge_model_with_contact, edge_model_without_contact):
    folders = glob.glob(files_loc + 'stage1_test/*/')
    random.shuffle(folders)

    output_dicts = []

    for folder in folders:
        image_location = glob.glob(folder + 'images/*')[0]
        start_image = Image.open(image_location).convert('LA')
        image_id = os.path.basename(image_location).split('.')[0]
        if len(image_id) < 5:
            print('here')
        np_image = np.array(start_image.getdata())[:, 0]
        np_image = np_image.reshape(start_image.size[1], start_image.size[0])

        np_image = normalize_image(np_image)
        output, _, _, _, _ = predict_image(loc_model, edge_model_with_contact,
                                           edge_model_without_contact, np_image, image_id)
        output_dicts.extend(output)

    df = pd.DataFrame.from_dict(output_dicts)
    df = df[['ImageId', 'EncodedPixels']]
    df.to_csv('output.csv', index = False)


def main():
    edge_model_with_contact = get_edge_model_with_contact()
    edge_model_without_contact = get_edge_model_without_contact()
    #edge_model_without_contact = None
    loc_model = get_loc_model()
    print('loc model loaded')

    print('edge model loaded')
    #run_tests(loc_model, edge_model_with_contact, edge_model_without_contact)
    run_predictions(loc_model, edge_model_with_contact, edge_model_without_contact)


if __name__ == '__main__':
    main()
    # print(glob.glob('../*'))
    # print(glob.glob('../*/*'))
    # print(glob.glob('../*/*/*'))