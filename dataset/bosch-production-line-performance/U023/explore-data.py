import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

train_categorical = pd.read_csv('../input/train_categorical.csv', nrows = 1000)
test_categorical = pd.read_csv('../input/test_categorical.csv', nrows = 1000)

print(train_categorical.head())
print(test_categorical.head())

train_numeric = pd.read_csv('../input/train_numeric.csv', nrows = 1000)
test_numeric = pd.read_csv('../input/test_numeric.csv', nrows = 1000)

print(train_numeric.head())
print(test_numeric.tail())

train_date = pd.read_csv('../input/train_date.csv', nrows = 1000)
test_date = pd.read_csv('../input/test_date.csv', nrows = 1000)

print(train_date.head())
print(test_date.tail())