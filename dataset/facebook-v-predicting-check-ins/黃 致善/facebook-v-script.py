import h5py
import pandas as pd
import zipfile

# scikit-learn
from sklearn import cross_validation, preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

# keras
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.objectives import sparse_categorical_crossentropy
from keras.utils import np_utils

F_TRAIN = '../input/train.csv'
F_TEST  = '../input/test.csv'
df = pd.read_csv(F_TRAIN)
dt = df.sample(frac=0.01)