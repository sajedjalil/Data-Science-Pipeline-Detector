import shutil
import os
import numpy as np
import pandas as pd
from random import sample
from itertools import chain
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

import matplotlib
import matplotlib.pyplot as plt

from skimage import data, io, exposure, feature
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity
from skimage.filters import threshold_otsu, threshold_local, threshold_yen
from skimage.filters import sobel
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.morphology import reconstruction

import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# gets all .png files from specified folder in the same directory
# this avoids the issue of reading in hidden folders accidentally

def sample_files(folder = '../input/aptos2019-blindness-detection/train_images', numfiles = 10):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            files.append(file)
    print('All file names loaded with length', len(files))
    if numfiles <= len(files):
        print('Sampling', numfiles, 'files out of', len(files), 'files')
        sampled_files = sample(files, numfiles)
    print('the file names are:', sampled_files)
    return sampled_files

def get_files(folder = '../input/aptos2019-blindness-detection/train_images'):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            files.append(file)
    print('All file names loaded with length', len(files))
    return files

# get_training_images imports all training images rescaled to 256px x 256px 
# and the result is output as a numpy array of dimension m x 256 x 256 x 3
# where m is the length of the training set and each image has 3 rgb channels

def get_images(files, directory = '../input/aptos2019-blindness-detection/train_images/'):
    images = []
    print('Reading in training images')
    for index, file in enumerate(files):
        raw = io.imread(directory + file)
        print('Importing image', (index + 1), ':', file) 
        images.append(resize(raw,(784,784), mode = 'constant'))
    return np.asarray(images)

# get_training_labels by checking the file names against all of the training labels
# each file name has a .png at the end so omit the last four characters from each file name
def get_labels(files):
    all_labels = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
    labels = []
    file_names = [file[0:-4] for file in files]
    all_files = pd.Index(all_labels )
    for file in file_names:
        file_index = pd.Index(all_labels.iloc[:,0]).get_loc(file)
        # print('found', file, 'at index', file_index)
        labels.append(all_labels.iloc[file_index,1])
    return np.asarray(labels)

# the following function takes in a list of images, performs basic image processing techniques
# (grayscale, color extraction, reconstruction, thresholding) to extract the relevant features (veins, exudates)
# and returns a list of 
def get_veins_exudates(before_images):
    exudate_images = []
    vein_images = []
    for image in before_images:
        red=image.copy()
        red[:,:,1] = 0
        red[:,:,2] = 0

        green=image.copy()
        green[:,:,0] = 0
        green[:,:,2] = 0

        # subtract out red only in order to darken the veins against a relatively lighter background
        vein_image = equalize_adapthist(rgb2gray(image - red))
        # equalized.append(image)
        exudate_image = equalize_adapthist(rgb2gray(red + green))
        # discs.append(get_disc(image))
        # equalized.append(image)

        # fill holes which are dark areas surrounded by lighter areas
        # subtract out holes from original image to obtain peaks
        seed = np.copy(exudate_image)
        seed[1:-1, 1:-1] = exudate_image.min()
        mask = exudate_image
        rec = reconstruction(seed, mask, method='dilation')
        peaks = exudate_image - rec

        exudate_thresh = threshold_yen(peaks)
        exudate_binary = peaks > exudate_thresh

        ## apply the disc as a mask
        exudate_binary[get_disc(exudate_image)] = False
        exudate_images.append(exudate_binary)

        selem = disk(4)
        vein_image = opening(vein_image, selem)
        # opened.append(image)

        vein_image = sobel(vein_image)
		# sobeled.append(image)

		# opening gets rid of salt "small bright spots" and connects dark cracks 

		# contrast stretching to enhance the darkness of the veins
        p2, p98 = np.percentile(vein_image, (2, 98))
        vein_image = rescale_intensity(vein_image, in_range=(p2, p98))
		# stretched.append(image)

        selem = disk(2)
        vein_image = opening(vein_image, selem)

        vein_thresh = threshold_otsu(vein_image)
        vein_binary = vein_image > vein_thresh
        vein_images.append(vein_binary)
	# return [np.asarray(before_images), np.asarray(exudate_images), np.asarray(vein_images)]
	# displayimages.create_plots([np.asarray(before_images), np.asarray(exudate_images), np.asarray(vein_images)])
    return score_images(vein_images), score_images(exudate_images)

# incoming image is already yellow extracted, greyscaled, and adaptively equalized
# detects the disc/retina as a mask to be subtracted from the exudate detector 
def get_disc(image):
    # get rid of small bright patches
    selem = disk(10)
    image = opening(image, selem)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    rec = reconstruction(seed, mask, method='dilation')
    peaks = image - rec

    thresh = threshold_otsu(image)
    binary = peaks > thresh
    return binary

def score_images(images):
    scores = []
    for image in images:
        scores.append(image.sum())
    return scores

# due to memory constraints, try processing the images in batches
def train_batch_processing(batch_size = 100):
    print('starting timer')
    start_time = time.time()
    featuresDF = pd.DataFrame(columns=['label', 'veinScores', 'exudateScores'])
    train_files = get_files(folder = '../input/aptos2019-blindness-detection/train_images/')
    # train_files = get_files(folder = 'train_images', numfiles = 100)
    batch_train_files = [train_files[x:x+batch_size] for x in range(0, len(train_files), batch_size)]
    for train_files in batch_train_files:
        train_labels = get_labels(train_files)
        train_images = get_images(train_files)
        vein_scores, exudate_scores = get_veins_exudates(train_images)
        print('Creating features and labels dataframe')
        featuresDF = featuresDF.append(pd.DataFrame(dict(label = train_labels, veinScores = vein_scores, exudateScores = exudate_scores)))
        print('Added', featuresDF.shape[0], 'images to train dataframe')
    print("Scored", len(batch_train_files), "images in", "--- %s seconds ---" % (time.time() - start_time))
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    print('Standardizing exudate and vein scores')
    featuresDF[['exudateScores', 'veinScores']] = scaler.fit_transform(featuresDF[['exudateScores', 'veinScores']])
    return featuresDF

def save_file(featuresDF):
    featuresDF.to_csv('featureData.csv', encoding='utf-8', index=False)

def load_file(filename = 'featureData.csv'):
    featuresDF = pd.read_csv(filename)
    return featuresDF

def test_batch_processing(batch_size = 100):
    test_files = get_files(folder = '../input/aptos2019-blindness-detection/test_images')
    # test_files = get_files(folder = 'test_images', numfiles = 100)
    test_featuresDF = pd.DataFrame(columns=['fileNames', 'veinScores', 'exudateScores'])
    batch_test_files = [test_files[x:x+batch_size] for x in range(0, len(test_files), batch_size)]
    for test_files in batch_test_files:
        test_images = get_images(test_files, directory = '../input/aptos2019-blindness-detection/test_images/')
        vein_scores, exudate_scores = get_veins_exudates(test_images)
        print('Creating features dataframe')
        fileNames = [file[0:-4] for file in test_files]
        test_featuresDF = test_featuresDF.append(pd.DataFrame(dict(fileNames = fileNames, veinScores = vein_scores, exudateScores = exudate_scores)))
        print('Added', test_featuresDF.shape[0], 'images to test dataframe')
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    print('Standardizing exudate and vein scores')
    test_featuresDF[['exudateScores', 'veinScores']] = scaler.fit_transform(test_featuresDF[['exudateScores', 'veinScores']])
    return test_featuresDF

# splitting the training data into training/validation sets in order to get an idea of how well the model is working
def logistic_model(featuresDF):
    # vein scores alone as feature label
    # X_train, X_test, y_train, y_test = train_test_split(featuresDF['veinScores'], featuresDF['label'], test_size = 0.20, random_state = 42)
    # model = LogisticRegression(random_state = 4, solver = 'lbfgs', multi_class = 'ovr').fit(np.asarray(X_train).reshape(-1,1), y_train)
    # print('Test accuracy =', sum(model.predict(np.asarray(X_test).reshape(-1, 1)) == y_test) / len(y_test))

    # choose between ovr or multinomial
    print('Creating logistic model from training set')
    X_train, X_test, y_train, y_test = train_test_split(featuresDF.iloc[:,1:3], featuresDF['label'], test_size = 0.20, random_state = 42)
    model = LogisticRegression(random_state = 4, solver = 'lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
    print('Test accuracy =', sum(model.predict(X_test) == y_test) / len(y_test))

# train the logistic model on the training images, and make predictions on the test images, 
# outputting the final predictions as an array
def final_logistic_model(featuresDF, test_featuresDF):
    print('Creating final logistic model from training set')
    X_train, y_train = featuresDF.iloc[:,1:3], featuresDF['label']
    y_train = y_train.astype('int')
    model = LogisticRegression(random_state = 4, solver = 'lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
    test_predictions = pd.DataFrame(dict(id_code = test_featuresDF['fileNames'], diagnosis = model.predict(test_featuresDF.iloc[:,1:3])))
    return test_predictions

def save_predictions(test_predictions):
    test_predictions.to_csv('submission.csv', index = False)

featuresDF = train_batch_processing()
test_featuresDF = test_batch_processing()
test_predictions = final_logistic_model(featuresDF, test_featuresDF)
save_predictions(test_predictions)