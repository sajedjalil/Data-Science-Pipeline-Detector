# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import svm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

NUM_SPECIES = 99
FEATURE_SIZE = 192

# Location of data
data_dir = "../input"

# Get data
train_data = pd.read_csv(os.path.join(data_dir,"train.csv"))
test_data = pd.read_csv(os.path.join(data_dir,"test.csv"))

# Map species name to integer labels
species_classes = dict(zip(train_data['species'].unique(), range(NUM_SPECIES)))

# Convert data to data matrix and corresponding labels
def convert_data(data, typeData, featureSize, species2class):
    X = np.zeros((data.shape[0],featureSize))
    y = np.zeros((data.shape[0],),dtype=np.int)
    
    for index, image in data.iterrows():
        if typeData == "train":
            X[index,:] = np.array(image[2:])
            y[index] = species2class[image['species']]
        else:
            X[index,:] = np.array(image[1:])
    
    return X, y

# Get matrices and labels for train and test data
X_train, y_train = convert_data(train_data, "train", FEATURE_SIZE, species_classes)
X_test, _ = convert_data(test_data, "test", FEATURE_SIZE, species_classes)

print("Train 0: ",X_train.shape, y_train.shape)
print("Test 0: ", X_test.shape)

# Build SVM classifier
clf = svm.SVC()
print(clf.fit(X_train, y_train))
# Any results you write to the current directory are saved as output.