import csv, os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

os.system("ls ../input")

features = pd.read_csv("../input/train.csv")
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print(train.head())
features = features.drop('id', axis=1)

# Extract target and Encode it to make it manageable by ML algo
labels = features.target.values
labels = LabelEncoder().fit_transform(labels)

# Remove target from train, else it's too easy ...
features = features.drop('target', axis=1)

#features = preprocessing.normalize(features)
scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
features = scaler.fit_transform(features, labels)
#print features