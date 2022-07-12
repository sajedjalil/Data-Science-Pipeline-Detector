#!/usr/bin/env python

"""image_classification.py: Classify images to classify fish types"""

import sys
import os
import glob
import joblib
import cv2
import numpy as np
from scipy.cluster import vq

import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


__author__ = "Pradeep Kumar A.V."


CLASSES = {
    'ALB': 1,
    'BET': 2,
    'DOL': 3,
    'LAG': 4,
    'NoF': 5,
    'OTHER': 6,
    'SHARK': 7,
    'YFT': 8
}

CLASSES_REV = {value: key for key, value in CLASSES.items()}


# Helper functions
def _load_img(path):
    """
    :param path: path of image to be loaded.
    :return: cv2 image object
    """
    img = cv2.imread(path)
    # Convert the image from cv2 default BGR format to RGB (for convenience)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _pretty_print(msg):
        print()
        print('=' * len(msg))
        print(msg)
        print('=' * len(msg))


def _detect_and_describe(image, method='ORB'):
    """
    :param image: Input RGB color image
    :return: keypoints and features tuple
    """
    # detect and extract features from the image
    if method == 'SIFT':
        descriptor = cv2.xfeatures2d.SIFT_create()
    else:
        descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    features = np.float32(features)

    # return a tuple of keypoints and features
    return kps, features


def _kmeans_clustering(data, k=7):
    """
    :param data: input data
    :param k: K value
    :return: k-Means clusters
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, label, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    return centers


#  Main wrapper methods

def extract_img_features(img_data_dir, type='train'):
    """
    :param img_data_dir: directory path where the images reside.
     The training images should reside in class named folders
    :return:
    """
    if type == 'train':
        files = glob.glob("%s/*/*" % img_data_dir)
    else:
        files = glob.glob("%s/*" % img_data_dir)
    dataset_size = len(files)
    resp = np.zeros((dataset_size, 1))
    ctr = 0
    print("\nProcessing images, and generating descriptors..\n")
    des_list = []
    for f in files:
        print("Processing image %s" % f)
        img = _load_img(f)
        kpts, des = _detect_and_describe(img)
        des_list.append((f, des))
        if type == 'train':
            resp[ctr] = CLASSES[f.split('/')[-2]]
            ctr += 1

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    k = 3
    print("\nClustering the descriptors to form BOVW dictionary..\n")
    centers = _kmeans_clustering(descriptors, k)
    im_features = np.zeros((dataset_size, k), "float32")
    for i in range(dataset_size):
        words, distance = vq.vq(des_list[i][1], centers)
        for w in words:
            im_features[i][w] += 1

    # Scaling the values of features
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    resp = np.float32(resp)
    return files, im_features, resp


def train_classifier(train_data, train_resp):
    """
    :param train_data: training data array
    :param train_resp: training data labels
    :return: trained classifier object
    """
    model = GaussianNB()
    model.fit(train_data, train_resp)
    return model


def test_classifier(model, test_data):
    """
    :param model: trained kNN classifier object
    :param test_data: test data array
    :return: predicted classes
    """
    result = model.predict_proba(test_data)
    return result


def main():
    """
    Main wrapper to call the classifier
    :return: None
    """
    use_model = False
    generate_model = True
    training_data_dir = '../input/train'
    testing_data_dir = '../input/test_stg1'

    if use_model:
        model = joblib.load('model.pkl')
    else:
        # Extract features and train the classifier
        _pretty_print("Extracting training image features")
        train_files, train_data, train_resp = \
            extract_img_features(training_data_dir)
        _pretty_print("Training the classifier")
        model = train_classifier(train_data, train_resp)
        if generate_model:
            joblib.dump(model, 'model.pkl')
    
    if os.path.exists(testing_data_dir):
        # Extract features and test the classifier
        _pretty_print("Extracting testing image features")
        test_files, test_data, test_resp = \
            extract_img_features(testing_data_dir, type='test')
        _pretty_print("Testing the classifier")
        predictions = test_classifier(model, test_data)
        columns = [CLASSES_REV[int(entry)] for entry in model.classes_]
        submission1 = pd.DataFrame(predictions, columns=columns)
        images = [f.split('/')[-1] for f in test_files]
        submission1.insert(0, 'image', images)
        submission1.head()
        submission1.to_csv("Naive_bayes_ORB_submission.csv", index=False)


if __name__ == '__main__':
    main()