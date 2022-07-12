import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
import xgboost as xgb

train = pd.read_csv('../input/train.csv')