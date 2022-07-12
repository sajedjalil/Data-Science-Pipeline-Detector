# Get the packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn import mixture
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
import time
import itertools
import operator

# Get the data

allData = pd.read_csv('../input/data.csv')
data = allData[allData['shot_made_flag'].notnull()].reset_index()

# What does it look like?

data.info()

