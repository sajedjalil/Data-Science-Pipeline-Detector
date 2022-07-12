import numpy as np
import pandas as pd

from xgboost.sklearn import XGBClassifier

store_file = "../input/store.csv"
train_file = "../input/train.csv"
test_file = "../input/test.csv"

store = pd.read_csv(store_file)
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
