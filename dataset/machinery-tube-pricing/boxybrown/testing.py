#######
## https://www.kaggle.com/andrewmatteson/caterpillar-tube-pricing/beating-the-benchmark-v1-0/run/20747
#######

import os

#os.system("ls ../input")
#os.system("echo \n\n")
#os.system("head ../input/*")

import csv
import pandas as pd
import numpy as np
import datetime
from sklearn import ensemble, preprocessing
import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

### grab that data
train = pd.read_csv("../input/train_set.csv", parse_dates=[2,])
test = pd.read_csv("../input/test_set.csv", parse_dates=[3,])

tubes = pd.read_csv('../input/tube.csv')
tube_end = pd.read_csv('../input/tube_end_form.csv')

train = pd.merge(train,tubes,on='tube_assembly_id',how='inner')
test = pd.merge(test,tubes,on='tube_assembly_id',how='inner')

train = pd.merge(train,tube_end, left_on='end_a', right_on = 'end_form_id',how='left')
test = pd.merge(test,tube_end,left_on='end_a', right_on = 'end_form_id',how='left')

train['material_id'].fillna('SP-9999',inplace=True)
test['material_id'].fillna('SP-9999',inplace=True)
