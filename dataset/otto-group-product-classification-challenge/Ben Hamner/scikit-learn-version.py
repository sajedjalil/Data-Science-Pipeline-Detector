import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.metrics import log_loss

# Checking to make sure this updated correctly
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.calibration import CalibratedClassifierCV
