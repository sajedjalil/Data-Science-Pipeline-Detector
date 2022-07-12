import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
from scipy import sum, average
from skimage import feature
from skimage.transform import resize
from sklearn import mixture
from sklearn.utils import shuffle
from sklearn import datasets, svm, metrics
from skimage.color import rgb2gray
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)

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

def crop_image(img,tol=0):
    mask = img > tol

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    return resize(img[x0:x1, y0:y1],(256,256))

mask_color = [0, 0, 0]
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

all_samples = 100
train_samples = 90

for t in all_cervix_images['type'].unique():
    for a in range(all_samples):
        image_name = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[a]
        #image = resize(imread(image_name), (200, 200))
        try:
            image = cv2.resize(imread(image_name), dsize=(512,512))
        except:
            pass
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
            gray_image = rgb2gray(image) #.6
            #gray_image = crop_image(rgb2gray(new_image)) #.45
            #gray_image = crop_image(rgb2gray(image)) #.5
        except:
            pass

        image_array = gray_image #resize(gray_image, (200, 200))
    
        if a > train_samples:
            test_data.append(image_array.flatten())
            test_target.append(t)
        else:
            train_data.append(image_array.flatten())
            train_target.append(t)
    
print(len(train_data))
print(len(test_data))

random_forest = RandomForestClassifier(n_estimators=30)
random_forest.fit(train_data, train_target)

random_forest_predicted = random_forest.predict(test_data)
random_forest_probability = random_forest.predict_proba(test_data)

print(metrics.classification_report(test_target, random_forest_predicted))
print(metrics.confusion_matrix(test_target, random_forest_predicted))
print(test_target)
print(random_forest_predicted)
print(random_forest_probability)