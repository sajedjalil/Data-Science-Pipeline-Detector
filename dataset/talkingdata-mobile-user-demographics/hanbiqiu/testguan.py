import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
datadir = '../input'
gatrain = pd.read_csv(os.path.join(datadir, 'gender_age_train.csv'),
                      index_col='device_id')
gatrain.group.value_counts().sort_index(ascending=False).plot(kind='barh')