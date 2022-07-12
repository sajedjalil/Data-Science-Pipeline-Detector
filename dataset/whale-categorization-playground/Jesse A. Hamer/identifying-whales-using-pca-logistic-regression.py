"""
Kaggle Competition: Humpback Whale Identification Challenge

Author: Jesse A. Hamer

DESCRIPTION:
Preamble:
This is my first Kaggle kernel and competition, so if you have any helpful feedback,
please don't hesitate to provide it! I basically chose this dataset at random--
it looked interesting and challenging and I can't say much more about my motivation.
My algorithm proceeds through three phases: image normalization (data cleaning),
dimensionality reduction, and then the model learning (logistic regression). Most of
the initial data analysis was done as needed in my personal Python IDE, but I have
taken care to include any relevant conclusions from this analysis.

1. Normalizing the images.
First, the pictures need to be manipulated into a more managable form. In particular, 
the picture types are not consistent: some are RGB, while others are grayscale. 
Since grayscale is the simpler picture type, we first convert each picture into 
its grayscale approximation using weights borrowed from the MATLAB RGB->grayscale
conversion algorithm. Next, the pictures come in varying shapes and sizes, and so
these qualities must be uniformized as well. Unfortunately we lose a good deal of
resolution on several of the pictures during this process. One point of improvement
would be to discern a more efficient way of performing this compression, and
compressing in such a way that a minimum of information is lost from each picture.

2. Dimensionality reduction.
Even after compressing, the image feature vectors will consist of several thousand
features. To remedy this, we perform PCA in order to extract out the most useful
features.

3. The model.
The model is a two-step pipeline, fitted with a GridSearchCV across several param-
eters. The first step in the pipeline is the abovementioned PCA transformation;
we will use the GridSearchCV to determine the optimal number of components as
well as whether whitening is appropriate. The second step of the pipeline
is a logistic regression model. We will use the GridSearchCV to set the regularization
parameter C.

Again, please let me know if you have any suggestions, I am very new to this and
looking to improve!

NOTE: Currently this kernel takes about an hour and a half to run (so it can't actually
be done on Kaggle's site), takes up a ton of memory, and only scores about 10%. 
Needless to say, there are some improvements to be made. I want to play around 
with the number of PCA components as well as the classifier.

"""

import os
import time
import math

import numpy as np
import pandas as pd

import matplotlib.image as mpimg

from sklearn.decomposition.pca import PCA
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline

start_time = time.clock()


#8381 of the images are RGB, so they will need to be converted to greyscale.
def rgb_to_gray(img):
    if len(img.shape) == 2: #already gray
        return(img)
    grayImage = np.zeros(img.shape)
    #These scaling coefficients are evidently standard for RGB->Grayscale conversion
    R = img[:,:,0]*0.299
    G = img[:,:,1]*0.587
    B = img[:,:,2]*0.114
    
    grayImage = R + G + B
    
    return(grayImage)
    
#We will also need to group together nearby pixels in order to 
#reduce image complexity and uniformize image size (so that every image has
#the same number of features).
    
def img_compress(img, x_bins=100,y_bins=100):
    x_splits = np.linspace(0,img.shape[1]-1,x_bins+1, dtype = int)
    y_splits=np.linspace(0,img.shape[0]-1,y_bins+1, dtype = int)
    
    compressed = np.zeros((y_bins,x_bins))
    
    for i in range(y_bins):
        for j in range(x_bins):
            temp = np.mean(img[y_splits[i]:y_splits[i+1],
                                      x_splits[j]:x_splits[j+1]])
            if math.isnan(temp):
                if y_splits[i]==y_splits[i+1]:
                    compressed[i,j]=compressed[i-1,j]
                else:
                    compressed[i,j] = compressed[i,j-1]
            else:
                compressed[i,j] = int(temp)
    return(compressed)


train_dir = '../input/train'
test_dir = '../input/test'
#Convert .jpg images into pixel arrays
imgs_train = [rgb_to_gray(mpimg.imread(train_dir + '/' + file, format = 'JPG')) for file in os.listdir(train_dir)]
#rows_train = [img.shape[0] for img in imgs_train]
#cols_train = [img.shape[1] for img in imgs_train]
print("Training images read.", "Time:", time.clock()-start_time)

min_cols = 138
min_rows = 54



#Get indices of those pictures which aren't too small
good_pics = [i for i in range(len(imgs_train)) if (imgs_train[i].shape[0]>=min_rows)and(imgs_train[i].shape[1]>=min_cols)]
#Select only pictures that are sufficiently large
imgs_train = [imgs_train[i] for i in good_pics]


#Compress images. Unfortunately we have to distort the aspect ratio, which may
#lose valuable information. But if we do not do this then features will not
#have consistent meaning across pictures, even if we ensure that all pictures
#get compressed to the same resolution (pixel count)

#We need to find the smallest dimension across all images (train and test) in
#order to properly compress; otherwise the compressing algorithm will generate
#NaN-values.

compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
print("Training images compressed.", "Time:", time.clock()-start_time)
del imgs_train



#Extract filenames to later associate to whale IDs
filenames = [file for file in os.listdir(train_dir)]
filenames = [filenames[i] for i in good_pics]
filenames_test = [file for file in os.listdir(test_dir)]
#Convert images into pandas dataframe format for learning the model
df1 = pd.DataFrame(data = [img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data = filenames, columns = ['FileName'])
data1 = pd.concat([df2,df1],axis = 1)
#Obtain filenames, indexed by whale IDs
data2 = pd.read_csv('./train.csv',names = ['FileName','WhaleID'])
#Map whale IDs to images via filenames (index of data1 is matched to value of data2)
data = data2.merge(data1, on = 'FileName')
data = data.drop('FileName',axis = 1)
del compressed_train_imgs, df1, df2, data1, data2

#Extract training examples and labels, and get test set.
X_train = data.iloc[:,1:]
y_train = data['WhaleID']


#Perform principal component analysis for dimensionality reduction
#Try playing around with n_components to get better scores
pca = PCA(random_state = 42, n_components = 100, whiten = True)
pca.fit(X_train)
print("PCA fitting complete", "Time:", time.clock()-start_time)

#Try other classifiers for better scores.
logreg = LogisticRegression(C = 1e-2)

#Define our pipeline. First we transform the data into its principal components,
#then learn the logistic regression model
clf = Pipeline([('pca',pca),
                ('logreg',logreg),
                ])

#Fit model
clf.fit(X_train,y_train)
print('Estimator fitting complete.', "Time:", time.clock()-start_time)
#Score model on training data
print("Score on training set:", clf.score(X_train,y_train))
#We no longer need the training data. Delete to free up memory.
del data, X_train, y_train

#Read and clean test set
imgs_test = [rgb_to_gray(mpimg.imread(test_dir + '/' + file)) for file in os.listdir(test_dir)]
print("Test images read.", "Time:", time.clock()-start_time)
#Smallest number of columns and rows across all test pictures
compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
print("Test images compressed.", "Time:", time.clock()-start_time)
del imgs_test
X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs

#Extract the top 5 predictions for each test example

y_preds = clf.predict_proba(X_test)
print("Predictions made on test set.", "Time:", time.clock()-start_time)


#Extract top 5 results
results = pd.DataFrame(data = [clf.classes_[np.argsort(y_preds[i,:])[-5:]] for i in range(y_preds.shape[0])],
                       index = filenames_test, header = False)
def list_to_str(L):
    string = ""
    for word in L:
        string = string + word + " "
    return(string)
        
results2 = pd.DataFrame(data = [list_to_str(results.iloc[i].values) for i in range(results.shape[0])],
                        index = filenames_test,columns = ['Id'])
 
results2.to_csv('submission.csv', sep = ',', index_label = 'Image', header = True)

print("Done.", "Time:", time.clock()-start_time)