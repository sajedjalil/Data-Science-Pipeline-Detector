import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import LinearSVC

cactus_label = pd.read_csv('../input/train.csv')

train_img = []
train_lb = []
for i in range(len(cactus_label)):
    row = cactus_label.iloc[i]
    fileName = row['id']
    train_lb.append(row['has_cactus'])
    path = "../input/train/train/{}".format(fileName)
    im = mpimg.imread(path)
    train_img.append(im)

X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb)
X_train = np.array(X_train)
X_test = np.array(X_test)


def imageToFeatureVector(images):
    flatten_img = []
    for img in images:
        data = np.array(img)
        flattened = data.flatten()
        flatten_img.append(flattened)
    return flatten_img

start = time.time()

X_train_flatten = imageToFeatureVector(X_train)
X_test_flatten = imageToFeatureVector(X_test)


test_img = []
sample = pd.read_csv('../input/sample_submission.csv')
folder = '../input/test/test/'

for i in range(len(sample)):
    row = sample.iloc[i]
    fileName = row['id']
    path = folder + fileName
    img = mpimg.imread(path)
    test_img.append(img)

test_img = np.asarray(test_img)

scaler = preprocessing.StandardScaler()
start = time.time()
scaler.fit(X_train_flatten)
X_test_normalized = scaler.transform(X_test_flatten)
X_train_normalized = scaler.transform(X_train_flatten)
test_flatten = imageToFeatureVector(test_img)
test_normalized = scaler.transform(test_flatten)
linearKernel = LinearSVC().fit(X_train_normalized, y_train)
predictions = linearKernel.predict(test_normalized)
sample['has_cactus'] = predictions
sample.head()

sample.to_csv('out.csv', index=False)
score = linearKernel.score(X_test_normalized,y_test)
end = time.time()

print("The run time of Linear SVC with normalized features is {:.2f} seconds".format(end-start))
print("Linear SCV with normalized features has test score of: {:.3f}".format(score))