import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
from scipy import sum, average
from skimage import feature
from skimage.transform import resize
from sklearn import datasets, svm, metrics
from skimage.color import rgb2gray
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)

import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle

from glob import glob
basepath = '../input/train/'

def Ra_space(img, Ra_ratio, a_threshold):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)
            
    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])
 
    return Ra

mask_color = [0, 0, 0]
 
# a channel saturation threshold
a_threshold = 300

all_cervix_images = []

for path in sorted(glob(basepath + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)

train_data = []
train_target = []
test_data = []
test_target = []
raw_data = []
feature_names = ['image_array']

all_samples = 5
train_samples = 2

plt_counter = 1
fig = plt.figure(figsize=(50, 50))

for t in all_cervix_images['type'].unique():
    i = 1
    
    for i in range(all_samples):
        image_name = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[i]
        image = imread(image_name)
        # creating the R-a feature for the image
        Ra_array = Ra_space(image, 1.0, a_threshold)
 
        # k-means gaussian mixture model
        g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
        image_array_sample = shuffle(Ra_array, random_state=0)[:1000]
        g.fit(image_array_sample)
        labels = g.predict(Ra_array)
 
        # creating the mask array and assign the correct cluster label
        boolean_image_mask = np.array(labels).reshape(image.shape[0], image.shape[1])
        outer_cluster_label = boolean_image_mask[0,0]
 
        new_image = image.copy()
 
        for i in range(boolean_image_mask.shape[0]):
            for j in range(boolean_image_mask.shape[1]):
                if boolean_image_mask[i, j] == outer_cluster_label:
                    new_image[i, j] = mask_color

        try:
            gray_image = rgb2gray(imread(image_name))
        except:
            pass

        ax = fig.add_subplot(all_samples, 10, plt_counter)
        ax.imshow(gray_image)
        
        ax = fig.add_subplot(all_samples, 10, plt_counter+1)
        ax.imshow(new_image)
        
        plt_counter += 2

plt.show()