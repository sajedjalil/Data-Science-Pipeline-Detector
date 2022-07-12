import os
os.system("ls ../input")
os.system("echo \n\n")
os.system("head ../input/*")

import pandas as pd 
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

# load training and test datasets
train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
tube_data = pd.read_csv('../input/tube.csv')
bill_of_materials_data = pd.read_csv('../input/bill_of_materials.csv')
specs_data = pd.read_csv('../input/specs.csv')